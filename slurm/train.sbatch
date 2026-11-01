#!/bin/bash
#SBATCH --job-name=eventpiv-train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
# --time haengt an der Paarzahl, also an pairs.runs; je Aufruf setzen.
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# [3] Training: ruft scripts/train_edge_nn.py auf, je Satz ein Lauf.
#   EXTRA       frei, Overrides, durch Leerzeichen getrennt.
#   EXTRA_SETS  frei, mehrere Saetze, durch ; getrennt, je Satz ein Lauf.
# EVENTPIV_DATA und EVENTPIV_OUT muessen gesetzt sein.
# Allgemeines zu sbatch steht im slurm-Abschnitt von scripts/_cli.py.
# Bedarf: nur CPU; Groessenordnung zehn GB Arbeitsspeicher.
# Beispiel:
#   sbatch slurm/train.sbatch
#   EXTRA="datensatz=batch1" sbatch slurm/train.sbatch
# Fehler:
#   Paar-Datensatz fehlt: die Gruppen muessen die des Paarbaus [2] sein.
#
# KEIN --gres: train.py rechnet auf der CPU, das kleine MLP braucht keine
# Karte. --cpus-per-task: torch parallelisiert die Matrixoperationen ueber
# Threads.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
# An die eigene Umgebung anpassen: venv und Pfad zum Repo.
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

echo "Node $(hostname)  Job $SLURM_JOB_ID  Start: $(date)"
# Ist EXTRA_SETS gesetzt, gilt EXTRA nicht.
SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra _SETS <<< "$SETS"
# Leere Eingabe: ein leerer Satz, sonst liefe die Schleife null Mal.
if [ ${#_SETS[@]} -eq 0 ]; then _SETS=(""); fi
for s in "${_SETS[@]}"; do
    echo "=== EXTRA='$s' ==="
    # shellcheck disable=SC2086
    python scripts/train_edge_nn.py $s || echo "FEHLGESCHLAGEN: $s"
done
echo "Ende: $(date)"