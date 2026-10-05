#!/bin/bash
#SBATCH --job-name=eventpiv-anim-traj
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Kein --gres: [D] liest Partition und Fit aus den Artefakten und rechnet
# nichts nach. Ein Job trotzdem: jedes Panel haelt den ganzen Eventstrom,
# bei dichten Runs mehr, als rema traegt.
#
# PANELS muss als EIN Hydra-Wert ankommen: ein nacktes fit,gt liest Hydra
# als Liste und bricht mit "Ambiguous value" ab.

set -euo pipefail
export PYTHONUNBUFFERED=1
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

SPLIT="${SPLIT:?SPLIT fehlt, etwa SPLIT=train}"
echo "Daten: $EVENTPIV_DATA"
RUN="${RUN:?RUN=<nr> setzen}"
PANELS="${PANELS:-tracklets,fit,gt}"

echo "Node $(hostname)  Job $SLURM_JOB_ID  Start: $(date)"
echo "=== $SPLIT/run$RUN, panels=$PANELS ==="
python scripts/animate_trajektorien.py \
    selection.split="$SPLIT" selection.run_nr="$RUN" \
    "anim.panels='$PANELS'" ${EXTRA:-}
echo "Ende: $(date)"
