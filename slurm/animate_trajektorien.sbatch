#!/bin/bash
#SBATCH --job-name=eventpiv-anim-traj
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# [D]: ruft scripts/animate_trajektorien.py fuer einen Run auf.
#   SPLIT   Pflicht, der Split.
#   RUN     Pflicht, eine Run-Nummer.
#   PANELS  frei, kommagetrennt; Vorgabe tracklets,fit,gt.
#   EXTRA   frei, Overrides, durch Leerzeichen getrennt.
# EVENTPIV_DATA und EVENTPIV_OUT muessen gesetzt sein.
# Allgemeines zu sbatch steht im slurm-Abschnitt von scripts/_cli.py.
# Bedarf: nur CPU; auf dichten Runs viel Arbeitsspeicher.
# Beispiel:
#   SPLIT=train RUN=8 sbatch slurm/animate_trajektorien.sbatch
#   SPLIT=train RUN=8 PANELS=trajektorien,gt \
#     sbatch slurm/animate_trajektorien.sbatch
# Fehler:
#   Fit-Datei fehlt: ohne PANELS ist das fit-Panel dabei, also erst [7].
#   Artefakt fehlt: EXTRA muss die Gruppen der Laeufe [5a] bis [7] tragen.
#
# Kein --gres: [D] liest Partition und Fit aus den Artefakten und rechnet
# nichts nach. Ein Job trotzdem: jedes Panel haelt den ganzen Eventstrom.
#
# PANELS bekommt unten innere Anfuehrungszeichen, sonst liest Hydra das
# Komma als Sweep und bricht mit "Ambiguous value" ab.

set -euo pipefail
export PYTHONUNBUFFERED=1
# An die eigene Umgebung anpassen: venv und Pfad zum Repo.
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
