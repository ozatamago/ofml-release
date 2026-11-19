"""Line colours shared by the result figures of the arXiv (TMLR) version (figures 3 and 4), 2026-11-18.

A surrogate keeps its colour in every figure.  The OFML surrogates are blue (OFML-GPN, the method of the theory)
and teal (OFML-Ens); the baselines of section 6.2 take green, red, violet and orange, after figures4papers
(blue for the method of interest, red for the closest alternative) and the colourful figures 1 and 2.
Acquisition functions that do not use the surrogate's predictive uncertainty (space-filling, random) choose their
points without the surrogate and are drawn in black and light grey; IVR and MaxVar take the colour of the surrogate
(solid and dashed).

Checked with check_palette_cvd.py (OKLab distance x100, Machado 2009 simulation):
  within a panel of figure 4 and within each row of the arXiv figure 3: normal >= 15, simulated >= 8 for all pairs;
  in the one-row (camera-ready) figure 3, three pairs across the GP and ensemble families are below the targets
  (OFML-GPN / DAGNN-Cat-Ens protan 2.5, Independent-GPN / Shared-Ens protan 3.8, OFML-Ens / Independent-GPN
  normal 12.8 and tritan 4.2); there the line style separates the families (GP solid, ensembles dashed).
"""
SURROGATE = {
    "OFML-GPN": "#0F4D92",
    "OFML-Ens": "#42949E",
    "Independent-GPN": "#59A14F",
    "Shared-DAG-Ens": "#B64342",
    "DAGNN-Cat-Ens": "#7B3294",
    "Shared-Ens": "#E08A1E",
}
NO_SURROGATE_ACQ = {"spacefill": ("#272727", "-"), "random": ("#B5B5B5", (0, (5, 2)))}
UNCERTAINTY_ACQ_LS = {"ivr": "-", "maxvar": (0, (5, 2))}
GREY = "#767676"
DASH = (0, (5, 2))
