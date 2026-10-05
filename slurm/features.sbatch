#!/bin/bash
#SBATCH --job-name=eventpiv-build-features
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Kein --gres: die Merkmalstabelle rechnet auf der CPU (KDTree, bincount).
# Ein Prozess je Run wie bei den Tracklets -- der Label-Strom eines Runs
# liegt vollstaendig im Speicher.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

RUNS="${RUNS:?RUNS fehlt, etwa RUNS=66}"
SPLIT="${SPLIT:?SPLIT fehlt, etwa SPLIT=val}"
echo "Daten: $EVENTPIV_DATA"
EXTRA_SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra SAETZE <<< "$EXTRA_SETS"
# read liefert bei leerer Eingabe ein LEERES Array, nicht eines mit einem
# leeren Element -- die Schleife liefe null Mal und der Job endete
# wortlos mit Erfolg.
if [ ${#SAETZE[@]} -eq 0 ]; then SAETZE=(""); fi

for e in "${SAETZE[@]}"; do
  for r in $RUNS; do
    echo "=== run${r}   EXTRA='$e' ==="
    # shellcheck disable=SC2086
    python scripts/build_features.py \
        selection.split="$SPLIT" "selection.runs=[$r]" $e \
      || echo "run${r} ($e) FEHLGESCHLAGEN -- weiter" >&2
  done
done