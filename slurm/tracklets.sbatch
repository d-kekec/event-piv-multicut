#!/bin/bash
#SBATCH --job-name=eventpiv-build-tracklets
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Ein Prozess JE RUN, nicht einer fuer alle: der Spitzenspeicher liegt bei
# 1,8 GB je Run, und ein langlebiger Prozess gibt ihn dem Betriebssystem
# nicht zuverlaessig zurueck. Kein Array-Job -- 100 Tasks waeren
# Verdraengung gegenueber anderen Nutzern, und sequentiell ist es schnell
# genug.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

RUNS="${RUNS:?RUNS fehlt, etwa RUNS=66}"
SPLIT="${SPLIT:?SPLIT fehlt, etwa SPLIT=val}"
echo "Daten: $EVENTPIV_DATA"

# Mehrere Parametersaetze in EINER Allokation. Semikolon trennt die
# Saetze: EXTRA_SETS="solver.cost_offset=-1;solver.cost_offset=-0.5"
EXTRA_SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra SAETZE <<< "$EXTRA_SETS"
# read liefert bei leerer Eingabe ein LEERES Array, nicht eines mit einem
# leeren Element -- die Schleife liefe null Mal und der Job endete
# wortlos mit Erfolg.
if [ ${#SAETZE[@]} -eq 0 ]; then SAETZE=(""); fi

echo "Node $(hostname)  Job $SLURM_JOB_ID  Start: $(date)"
# Welche Karte, und wie voll ist sie beim Start? Eine fremd belegte Karte
# zeigt sich sonst erst als OOM mitten im Lauf.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

for e in "${SAETZE[@]}"; do
  for r in $RUNS; do
    echo "=== run${r}   EXTRA='$e' ==="
    # shellcheck disable=SC2086
    python scripts/build_tracklets.py \
        selection.split="$SPLIT" "selection.runs=[$r]" $e \
      || echo "run${r} ($e) FEHLGESCHLAGEN -- weiter" >&2
  done
done

echo "Ende: $(date)"