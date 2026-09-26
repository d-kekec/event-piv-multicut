#!/bin/bash
#SBATCH --job-name=eventpiv-train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
# Der Datensatz traegt 5 Mio. Paare je Run: 50 Mio. bei zehn Runs,
# 150 Mio. bei dreissig. --time je Aufruf setzen.
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# KEIN --gres: train.py rechnet auf der CPU (Befund 28.07.2026). Ein
# 33-Parameter-MLP braucht keine Karte, und ohne GRES-Anforderung kommt der
# Job deutlich schneller dran.
# --cpus-per-task=8: torch parallelisiert die Matrixoperationen ueber Threads.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

echo "Node $(hostname)  Job $SLURM_JOB_ID  Start: $(date)"
# EXTRA_SETS: mehrere Parametersaetze, durch ';' getrennt, NACHEINANDER in
# derselben Allokation. EXTRA bleibt der Einzelfall.
SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra _SETS <<< "$SETS"
for s in "${_SETS[@]}"; do
    echo "=== EXTRA='$s' ==="
    # shellcheck disable=SC2086
    python scripts/train_edge_nn.py $s || echo "FEHLGESCHLAGEN: $s"
done
echo "Ende: $(date)"