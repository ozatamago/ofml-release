#!/bin/bash
# Start the pinning loop of pin_job.py for a Slurm job on this machine (xeon6530-h100nvl) unless it is already running.
# usage: pin_start.sh JOBID LOGFILE
cd "${OFML_ROOT:?set OFML_ROOT to the repository root}"/acquisition
if pgrep -u "$USER" -f "^/usr/bin/python3 prestage/pin_job.py $1 --loop" > /dev/null; then echo "already running"; exit 0; fi
nohup /usr/bin/python3 prestage/pin_job.py "$1" --loop 30 >> "$2" 2>&1 < /dev/null &
sleep 2
pgrep -u "$USER" -af "^/usr/bin/python3 prestage/pin_job.py $1 --loop"
