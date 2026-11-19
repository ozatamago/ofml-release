"""gpn_smc.factorised_independent -- S1 of the GPN improvement programme.

Factorised independent-protocol GPN backend: Q independent SMC particle
systems (one per protocol), each with its OWN logweights, old_loglik,
observation list, ESS test, resampling and pCN step size.

Why this is exact (REVIEW.md "main mathematical checks", item 1): the
independent arm has no shared types, so the joint posterior factorises,
pi(u_1..u_Q | D) = prod_p pi_p(u_p | D_p), where D_p is protocol p's own
observation set.  Observing a point of protocol p changes only the
likelihood of u_p, so only system p needs a weight update, a resampling
test and a rejuvenation.  The other Q-1 systems are not touched.

Differences from ``gpn_smc.backends.IndependentProtocolGpnBackend``
(left unmodified; this file only imports from the package):

* state per protocol is ``(n_devices, local_particles, 2G)`` (depth 3 only);
* each protocol is padded to ITS OWN bucket ``n_max_for(n_p)`` taken from
  ``n_max_buckets_pp`` (default 8/16/32/64; a protocol holds at most G=64
  distinct points under the no-re-observation UCB rule), not to the total
  observation bucket;
* the pCN kernel moves ONE block (one protocol); in the segment mode the
  step size beta of each protocol adapts from that protocol's own
  population acceptance via ``_segment_beta_update`` (same rule the old
  segment-blocked kernel applies per block);
* kernels are compiled once per bucket and shared by all protocols (the
  first-layer Cholesky is a runtime argument, so protocols with different
  ``protocol_layer1_cholesky`` entries share the same compiled kernel);
* the per-protocol posterior moments are cached and recomputed only for
  the protocol touched by the latest observation (exact: the other
  protocols' latent, weights and observations did not change).

Random streams: ``derive_stream_seed(inference_seed, n_initial,
len(observed_total), purpose)`` with ``purpose = 2 + 2p`` (resampling) and
``3 + 2p`` (rejuvenation) for protocol p; purposes 0/1 of the joint
backends are not used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .constants import GRID_SIZE_PER_PROTOCOL
from .library import ProtocolLibrarySpec
from .numerics import (
    compute_chunk_size,
    derive_stream_seed,
    normalize_logweights,
    rbf_covariance,
    stable_cholesky,
    systematic_resample,
)
from .pcn import PcnBetaState, SegmentRejuvenationInfo, _segment_beta_update
from .pushforward import pushforward_h
from .terminal_rb import build_terminal_rb_functions

FACTORISED_PCN_MODES = ("segment-blocked", "segment", "fixed")


def resample_purpose(p: int) -> int:
    return 2 + 2 * int(p)


def rejuvenate_purpose(p: int) -> int:
    return 3 + 2 * int(p)


class FactorisedIndependentGpnBackend:
    """Q independent depth-3 whitened GPN stacks, one particle system each.

    Interface naming mirrors ``IndependentProtocolGpnBackend`` (``n_particles``,
    ``n_devices``, ``local_particles``, ``n_protocols``, ``n_candidates``,
    ``n_max_for``, ``initialise``, ``loglik``, ``rejuvenate``,
    ``posterior_moments``, ``host_particles``, ``shard_particles``,
    ``sample_prior_h``) with a protocol index ``p`` added where the state is
    per protocol.
    """

    def __init__(
        self,
        warped_x: np.ndarray,
        library_spec: ProtocolLibrarySpec,
        noise_variance: float,
        kernel_jitter: float,
        n_particles: int,
        pcn_steps: int,
        *,
        pcn_mode: str = "segment-blocked",
        pcn_segment_length: int | None = None,
        n_protocols: int = 16,
        protocol_depths: tuple[int, ...] | None = None,
        protocol_layer1_cholesky: tuple[Any, ...] | None = None,
        n_max_buckets_pp: tuple[int, ...] = (8, 16, 32, 64),
        chunk_target_bytes: int = 256 * 1024 * 1024,
        adapt_beta: bool = True,
    ) -> None:
        devices = jax.devices("gpu")
        if not devices:
            raise RuntimeError("No JAX GPU devices are visible")
        if n_particles % len(devices):
            raise ValueError(f"particles={n_particles} must be divisible by devices={len(devices)}")
        if pcn_mode not in FACTORISED_PCN_MODES:
            raise ValueError(f"pcn_mode={pcn_mode!r} must be one of {FACTORISED_PCN_MODES}")
        if n_protocols < 1:
            raise ValueError(f"n_protocols={n_protocols} must be >= 1")
        if protocol_depths is not None and any(int(d) != 3 for d in protocol_depths):
            raise ValueError(
                f"FactorisedIndependentGpnBackend supports depth 3 only, got protocol_depths={protocol_depths}"
            )
        G = int(warped_x.shape[0])
        if G != GRID_SIZE_PER_PROTOCOL:
            raise ValueError(f"expected a {GRID_SIZE_PER_PROTOCOL}-point grid, got {G}")
        buckets = tuple(sorted(int(b) for b in n_max_buckets_pp))
        if not buckets or any(b <= 0 for b in buckets) or len(set(buckets)) != len(buckets):
            raise ValueError(f"n_max_buckets_pp={n_max_buckets_pp} must be distinct positive ints")
        self.n_max_buckets_pp = buckets
        self.n_max = buckets[-1]
        self.devices = devices
        self.n_devices = len(devices)
        self.n_particles = int(n_particles)
        self.local_particles = self.n_particles // self.n_devices
        self.library_spec = library_spec
        self.G = G
        self.n_protocols = int(n_protocols)
        self.protocol_depths = tuple([3] * self.n_protocols)
        self.depth = 3
        self.state_width = 2 * G  # per-protocol whitened state: layer-1 (G) + layer-2 (G)
        self.n_candidates = self.n_protocols * G
        self.n_pcn_blocks = self.n_protocols  # interface parity (one beta per protocol)
        self.pcn_steps = int(pcn_steps)
        self.pcn_mode = pcn_mode
        self.segment_mode = True  # every supported mode runs the segment scan
        # S0c additive kwarg: adapt_beta=False switches adaptation off (default True = old behaviour).
        self.adapt_beta = pcn_mode in ("segment-blocked", "segment") and bool(adapt_beta)
        from .constants import PCN_SEGMENT_LENGTH

        seg = PCN_SEGMENT_LENGTH if pcn_segment_length is None else int(pcn_segment_length)
        if seg <= 0 or self.pcn_steps % seg != 0:
            raise ValueError(f"pcn_steps={pcn_steps} must be a positive multiple of pcn_segment_length={seg}")
        self.pcn_segment_length = seg
        self.n_pcn_segments = self.pcn_steps // seg
        self.particle_sharding = NamedSharding(Mesh(np.asarray(devices), ("device",)), P("device"))

        layer1_cov = rbf_covariance(warped_x, library_spec.amplitude, library_spec.layer1_lengthscale)
        l1_np, self.layer1_jitter_used = stable_cholesky(layer1_cov, kernel_jitter)
        self._layer1_cholesky = jnp.asarray(l1_np, dtype=jnp.float64)
        if protocol_layer1_cholesky is None:
            self._protocol_layer1_cholesky = tuple([self._layer1_cholesky] * self.n_protocols)
        else:
            self._protocol_layer1_cholesky = tuple(
                jnp.asarray(c, dtype=jnp.float64) for c in protocol_layer1_cholesky
            )
            if len(self._protocol_layer1_cholesky) != self.n_protocols:
                raise ValueError("protocol_layer1_cholesky must have n_protocols entries")
        self._jitter_const = float(kernel_jitter)
        self._noise_variance = float(noise_variance)
        self._inner_amplitudes = jnp.asarray([library_spec.amplitude], dtype=jnp.float64)
        self._inner_lengthscales = jnp.asarray([library_spec.layer2_lengthscale], dtype=jnp.float64)
        self._chunk_target_bytes = int(chunk_target_bytes)
        self.chunk_size_local = compute_chunk_size(self.local_particles, G, chunk_target_bytes)
        self._bundles: dict[int, dict] = {}

        def initialise_local(key: jax.Array) -> jax.Array:
            return jax.random.normal(key, (self.local_particles, self.state_width), dtype=jnp.float64)

        self._init = jax.pmap(initialise_local, devices=self.devices)

    # ------------------------------------------------------------------ buckets
    def n_max_for(self, n_obs: int) -> int:
        """Smallest per-protocol bucket that holds ``n_obs`` observations."""
        for bucket in self.n_max_buckets_pp:
            if bucket >= n_obs:
                return bucket
        raise ValueError(f"n_obs={n_obs} exceeds every bucket in n_max_buckets_pp={self.n_max_buckets_pp}")

    def _bundle(self, n_max: int) -> dict:
        if n_max not in self._bundles:
            if n_max not in self.n_max_buckets_pp:
                raise ValueError(f"n_max={n_max} is not in n_max_buckets_pp={self.n_max_buckets_pp}")
            self._bundles[n_max] = self._build_bundle(n_max)
        return self._bundles[n_max]

    def _build_bundle(self, n_max: int) -> dict:
        G = self.G
        spec = self.library_spec
        jitter_const = self._jitter_const
        inner_amplitudes, inner_lengthscales = self._inner_amplitudes, self._inner_lengthscales
        chunk_size_local = self.chunk_size_local
        rb_chunk = compute_chunk_size(self.local_particles, max(G, n_max), self._chunk_target_bytes)
        ll_chunk = compute_chunk_size(self.local_particles, n_max, self._chunk_target_bytes)
        loglik_from_h, moments_from_h, _ = build_terminal_rb_functions(
            n_candidates=G,
            n_max=n_max,
            amplitude=spec.amplitude,
            lengthscale=spec.terminal_lengthscale,
            noise_variance=self._noise_variance,
            jitter=jitter_const,
            chunk_size=rb_chunk,
            loglik_chunk_size=ll_chunk,
        )

        def push(state_p: jax.Array, chol: jax.Array) -> jax.Array:
            return pushforward_h(
                state_p, chol, inner_amplitudes, inner_lengthscales, jitter_const, G, 3, chunk_size_local
            )

        def loglik_local(state, obs_idx, y_padded, active, chol):
            return loglik_from_h(push(state, chol), obs_idx, y_padded, active)

        n_segments = self.n_pcn_segments
        segment_length = self.pcn_segment_length
        n_particles = self.n_particles
        adapt = self.adapt_beta

        def pcn_local(state, loglik, key, obs_idx, y_padded, active, chol, beta0):
            def pcn_step(carry, _):
                current, current_ll, state_key, beta = carry
                state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                white = jax.random.normal(normal_key, current.shape, dtype=jnp.float64)
                shrink = jnp.sqrt(1.0 - beta * beta)
                proposal = shrink * current + beta * white
                proposal_ll = loglik_from_h(push(proposal, chol), obs_idx, y_padded, active)
                accepted = jnp.log(
                    jax.random.uniform(uniform_key, (current.shape[0],), dtype=jnp.float64)
                ) < (proposal_ll - current_ll)
                current = jnp.where(accepted[:, None], proposal, current)
                current_ll = jnp.where(accepted, proposal_ll, current_ll)
                return (current, current_ll, state_key, beta), jnp.sum(accepted)

            def segment_body(carry, _):
                current, current_ll, state_key, beta = carry
                (current, current_ll, state_key, _), accepted_per_step = jax.lax.scan(
                    pcn_step, (current, current_ll, state_key, beta), None, length=segment_length
                )
                global_accepted = jax.lax.psum(jnp.sum(accepted_per_step), axis_name="device")
                seg_acceptance = global_accepted / (segment_length * n_particles)
                new_beta = _segment_beta_update(beta, seg_acceptance) if adapt else beta
                return (current, current_ll, state_key, new_beta), (beta, seg_acceptance)

            (state, final_ll, _, beta_final), (beta_trace, acceptance_trace) = jax.lax.scan(
                segment_body, (state, loglik, key, beta0), None, length=n_segments
            )
            return state, final_ll, beta_final, beta_trace, acceptance_trace

        def moments_local(state, local_weights, obs_idx, y_padded, active, chol):
            return moments_from_h(push(state, chol), local_weights, obs_idx, y_padded, active)

        return {
            "loglik_from_h": loglik_from_h,
            "moments_from_h": moments_from_h,
            "push": push,
            "_loglik": jax.pmap(loglik_local, in_axes=(0, None, None, None, None), devices=self.devices),
            "_pcn": jax.pmap(
                pcn_local,
                in_axes=(0, 0, 0, None, None, None, None, None),
                axis_name="device",
                devices=self.devices,
            ),
            "_moments": jax.pmap(moments_local, in_axes=(0, 0, None, None, None, None), devices=self.devices),
        }

    # ------------------------------------------------------------------ state
    def initialise(self, seed: int) -> list[jax.Array]:
        """Q prior states, each ``(n_devices, local_particles, 2G)``."""
        keys_p = jax.random.split(jax.random.PRNGKey(seed), self.n_protocols)
        out = []
        for p in range(self.n_protocols):
            res = self._init(jax.random.split(keys_p[p], self.n_devices))
            res.block_until_ready()
            out.append(res)
        return out

    def host_particles(self, latent_p: jax.Array) -> np.ndarray:
        return np.asarray(latent_p).reshape(self.n_particles, self.state_width)

    def shard_particles(self, particles: np.ndarray) -> jax.Array:
        shaped = np.asarray(particles, dtype=np.float64).reshape(
            self.n_devices, self.local_particles, self.state_width
        )
        return jax.device_put(shaped, self.particle_sharding)

    # ------------------------------------------------------------- observations
    def protocol_observations(
        self, p: int, observed_flat: list[int] | np.ndarray, noisy_truth_flat: np.ndarray,
        n_max: int | None = None, *, observed_y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Protocol p's own padded ``(local_index, y, active)`` triple, in
        observation order, padded to ``n_max`` (default: its own bucket).
        Returns ``(indices, y, active, n_p)``.

        2026-10-31 additive ``observed_y`` (default ``None`` = old behaviour,
        y = ``noisy_truth_flat[flat_p]``): per-observation y aligned with
        ``observed_flat``; used when an index can be observed repeatedly with
        a fresh noise draw per visit. Duplicated indices are fine (rows are
        copied in observation order; no uniqueness assumption)."""
        obs = np.asarray(observed_flat, dtype=np.int64)
        mask = (obs // self.G) == p
        flat_p = obs[mask]
        n_p = int(flat_p.shape[0])
        if n_max is None:
            n_max = self.n_max_for(n_p)
        if n_p > n_max:
            raise ValueError(f"protocol {p} has {n_p} observations > n_max={n_max}")
        idx = np.zeros(n_max, dtype=np.int32)
        y = np.zeros(n_max, dtype=np.float64)
        act = np.zeros(n_max, dtype=np.float64)
        idx[:n_p] = (flat_p % self.G).astype(np.int32)
        if observed_y is None:
            y[:n_p] = np.asarray(noisy_truth_flat, dtype=np.float64)[flat_p]
        else:
            y_all = np.asarray(observed_y, dtype=np.float64)
            if y_all.shape != obs.shape:
                raise ValueError(f"observed_y shape {y_all.shape} != observed_flat shape {obs.shape}")
            y[:n_p] = y_all[mask]
        act[:n_p] = 1.0
        return idx, y, act, n_p

    # ------------------------------------------------------------------ kernels
    def loglik(self, p: int, latent_p: jax.Array, obs_idx: np.ndarray, y: np.ndarray, active: np.ndarray) -> jax.Array:
        """``(n_devices, local_particles)`` loglik of protocol p's own observations."""
        bundle = self._bundle(int(y.shape[0]))
        res = bundle["_loglik"](latent_p, obs_idx, y, active, self._protocol_layer1_cholesky[p])
        res.block_until_ready()
        return res

    def rejuvenate(
        self, p: int, latent_p: jax.Array, loglik_p: jax.Array, seed: int,
        obs_idx: np.ndarray, y: np.ndarray, active: np.ndarray, beta: float | np.ndarray,
    ) -> tuple[jax.Array, jax.Array, SegmentRejuvenationInfo]:
        """``pcn_steps`` single-block pCN sweeps on protocol p's system.
        ``loglik_p`` must be the current loglik of ``latent_p``. ``beta`` is
        the scalar starting step size of this protocol."""
        bundle = self._bundle(int(y.shape[0]))
        keys = jax.random.split(jax.random.PRNGKey(seed), self.n_devices)
        latent_p, loglik_p, beta_final, beta_trace, acc_trace = bundle["_pcn"](
            latent_p, loglik_p, keys, obs_idx, y, active,
            self._protocol_layer1_cholesky[p], np.asarray(float(beta), dtype=np.float64),
        )
        loglik_p.block_until_ready()
        acc = np.asarray(acc_trace)[0]
        bt = np.asarray(beta_trace)[0]
        info = SegmentRejuvenationInfo(
            last_segment_acceptance=float(acc[-1]),
            first_segment_acceptance=float(acc[0]),
            beta_used_final_segment=float(bt[-1]),
            beta_next=float(np.asarray(beta_final)[0]),
            beta_trace=bt,
            acceptance_trace=acc,
        )
        return latent_p, loglik_p, info

    def protocol_moments(
        self, p: int, latent_p: jax.Array, weights_p: np.ndarray, obs_idx: np.ndarray,
        y: np.ndarray, active: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mean, second, within)``, each ``(G,)``: weighted RB moments of protocol p."""
        bundle = self._bundle(int(y.shape[0]))
        local_weights = np.asarray(weights_p).reshape(self.n_devices, self.local_particles)
        res = bundle["_moments"](
            latent_p, local_weights, obs_idx, y, active, self._protocol_layer1_cholesky[p]
        )
        return tuple(np.sum(np.asarray(v), axis=0) for v in res)  # type: ignore[return-value]

    def posterior_moments(
        self,
        latents: list[jax.Array],
        weights: list[np.ndarray],
        observed_flat: list[int] | np.ndarray,
        noisy_truth_flat: np.ndarray,
        *,
        observed_y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenated ``(mean, variance, within)``, each ``(Q*G,)``, protocol-major."""
        means, variances, withins = [], [], []
        for p in range(self.n_protocols):
            idx, y, act, _ = self.protocol_observations(
                p, observed_flat, noisy_truth_flat, observed_y=observed_y
            )
            m, s, w = self.protocol_moments(p, latents[p], weights[p], idx, y, act)
            means.append(m)
            variances.append(np.maximum(s - m * m, 0.0))
            withins.append(w)
        return np.concatenate(means), np.concatenate(variances), np.concatenate(withins)

    def sample_prior_h(self, key: jax.Array, n_samples: int | None = None) -> np.ndarray:
        """Test utility: ``(n, Q*G)`` penultimate values from fresh prior draws."""
        n = self.n_particles if n_samples is None else int(n_samples)
        keys = jax.random.split(key, self.n_protocols)
        chunk = compute_chunk_size(n, self.G, 256 * 1024 * 1024)
        parts = []
        for p in range(self.n_protocols):
            st = jax.random.normal(keys[p], (n, self.state_width), dtype=jnp.float64)
            parts.append(pushforward_h(
                st, self._protocol_layer1_cholesky[p], self._inner_amplitudes, self._inner_lengthscales,
                self._jitter_const, self.G, 3, chunk,
            ))
        return np.asarray(jnp.concatenate(parts, axis=1))


@dataclass
class FactorisedState:
    """Everything the Q systems carry between observations."""

    latents: list[jax.Array]
    logweights: list[np.ndarray]
    old_loglik: list[np.ndarray]
    resampling_count_pp: np.ndarray
    pcn_states: list[PcnBetaState]
    ess_pp: np.ndarray
    # exact cache of per-protocol (mean, variance, within); invalidated for
    # the touched protocol only
    _moments_cache: list[Any] = field(default_factory=list)
    last_acceptance: float | None = None
    last_pcn_beta: float | None = None

    @classmethod
    def fresh(cls, backend: FactorisedIndependentGpnBackend, inference_seed: int, pcn_mode: str,
              pcn_beta: float) -> "FactorisedState":
        Q, Pn = backend.n_protocols, backend.n_particles
        # one scalar beta per protocol: PcnBetaState in the scalar mode of the same family
        scalar_mode = "segment" if pcn_mode in ("segment-blocked", "segment") else "fixed"
        return cls(
            latents=backend.initialise(inference_seed),
            logweights=[np.full(Pn, -math.log(Pn)) for _ in range(Q)],
            old_loglik=[np.zeros(Pn, dtype=np.float64) for _ in range(Q)],
            resampling_count_pp=np.zeros(Q, dtype=np.int64),
            pcn_states=[PcnBetaState(mode=scalar_mode, fixed_beta=pcn_beta, n_blocks=1) for _ in range(Q)],
            ess_pp=np.full(Q, float(Pn)),
            _moments_cache=[None] * Q,
        )

    def weights(self, p: int) -> np.ndarray:
        return normalize_logweights(self.logweights[p])[0]

    def current_ess_pp(self) -> np.ndarray:
        return np.array([normalize_logweights(lw)[1] for lw in self.logweights])


def assimilate_observation(
    backend: FactorisedIndependentGpnBackend,
    state: FactorisedState,
    observed_flat: list[int],
    noisy_truth_flat: np.ndarray,
    ess_fraction: float,
    inference_seed: int,
    n_initial: int,
    *,
    observed_y: np.ndarray | None = None,
) -> dict[str, Any]:
    """Assimilate the LAST entry of ``observed_flat`` (already appended by the
    caller) into the system of its protocol only.  Recomputes protocol p's
    loglik over ALL of p's observations (as ``assimilate_u15`` does), updates
    its weights, and when ESS_p < ess_fraction * P resamples and rejuvenates
    system p only.  Returns a small info dict (protocol, n_p, bucket, ess,
    resampled, acceptance/beta of the rejuvenation if any).

    2026-10-31 additive ``observed_y`` (default ``None`` = old behaviour):
    per-observation y aligned with ``observed_flat`` (see
    ``FactorisedIndependentGpnBackend.protocol_observations``)."""
    G = backend.G
    p = int(observed_flat[-1]) // G
    idx, y, act, n_p = backend.protocol_observations(
        p, observed_flat, noisy_truth_flat, observed_y=observed_y
    )
    Pn = backend.n_particles
    latent = state.latents[p]
    ll_dev = backend.loglik(p, latent, idx, y, act)
    new_loglik = np.asarray(ll_dev).reshape(Pn)
    logw = state.logweights[p] + new_loglik - state.old_loglik[p]
    weights, ess = normalize_logweights(logw)
    resampled = False
    info = None
    beta_start = None
    if ess < ess_fraction * Pn:
        indices = systematic_resample(
            weights, derive_stream_seed(inference_seed, n_initial, len(observed_flat), purpose=resample_purpose(p))
        )
        latent = backend.shard_particles(backend.host_particles(latent)[indices])
        gathered_ll = new_loglik[indices]
        logw = np.full(Pn, -math.log(Pn))
        beta_start = state.pcn_states[p].current(n_p)
        latent, ll_dev, info = backend.rejuvenate(
            p, latent, jax.device_put(gathered_ll.reshape(backend.n_devices, backend.local_particles),
                                      backend.particle_sharding),
            derive_stream_seed(inference_seed, n_initial, len(observed_flat), purpose=rejuvenate_purpose(p)),
            idx, y, act, beta_start,
        )
        new_loglik = np.asarray(ll_dev).reshape(Pn)
        ess = float(Pn)
        state.resampling_count_pp[p] += 1
        state.pcn_states[p].update(info)
        state.last_acceptance = float(info.last_segment_acceptance)
        state.last_pcn_beta = float(info.beta_used_final_segment)
        resampled = True
    state.latents[p] = latent
    state.logweights[p] = logw
    state.old_loglik[p] = new_loglik
    state.ess_pp[p] = ess
    state._moments_cache[p] = None
    return {
        "protocol": p, "n_p": n_p, "bucket": int(y.shape[0]), "ess": float(ess), "resampled": resampled,
        "info": info,
    }


def state_posterior_moments(
    backend: FactorisedIndependentGpnBackend,
    state: FactorisedState,
    observed_flat: list[int] | np.ndarray,
    noisy_truth_flat: np.ndarray,
    *,
    observed_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean, variance, within)`` over all Q*G candidates; only protocols whose
    cache was invalidated by an observation are recomputed."""
    G = backend.G
    means, variances, withins = [], [], []
    for p in range(backend.n_protocols):
        if state._moments_cache[p] is None:
            idx, y, act, _ = backend.protocol_observations(
                p, observed_flat, noisy_truth_flat, observed_y=observed_y
            )
            m, s, w = backend.protocol_moments(p, state.latents[p], state.weights(p), idx, y, act)
            state._moments_cache[p] = (m, np.maximum(s - m * m, 0.0), w)
        m, v, w = state._moments_cache[p]
        means.append(m)
        variances.append(v)
        withins.append(w)
    return np.concatenate(means), np.concatenate(variances), np.concatenate(withins)
