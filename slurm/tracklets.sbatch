#!/bin/bash
#SBATCH --job-name=eventpiv-build-tracklets
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# [5a] Tracklets: ruft scripts/build_tracklets.py auf, je Run und Satz ein
# Lauf.
#   RUNS        Pflicht, Runs durch Leerzeichen getrennt.
#   SPLIT       Pflicht, der Split.
#   EXTRA       frei, Overrides, durch Leerzeichen getrennt.
#   EXTRA_SETS  frei, mehrere Saetze, durch ; getrennt, je Satz ein Lauf.
# EVENTPIV_DATA und EVENTPIV_OUT muessen gesetzt sein.
# Allgemeines zu sbatch steht im slurm-Abschnitt von scripts/_cli.py.
# Bedarf: GPU. --time an die Zahl der Runs anpassen.
# Beispiel:
#   RUNS="8 13" SPLIT=train sbatch --gres=gpu:rtx4090:1 slurm/tracklets.sbatch
#   RUNS="32" SPLIT=train EXTRA="datensatz=batch1" \
#     sbatch --gres=gpu:rtx4090:1 slurm/tracklets.sbatch
# Fehler:
#   CUDA-OOM: die Karte war fremd belegt; die Belegung beim Start steht im Log.
#   window.* ohne io.out_path ueberschreibt das Artefakt des ganzen Runs.
#   Netz fehlt: die Gruppen muessen die des Trainings [3] sein.
#
# Ein Prozess JE RUN, nicht einer fuer alle: ein Absturz auf der Karte,
# etwa in RAMA, nimmt nur diesen Run mit, und ein langlebiger Prozess gaebe
# den Spitzenspeicher eines Runs, Groessenordnung GB, nicht zuverlaessig
# zurueck. Kein Array-Job: er belegte je Run eine eigene Karte.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
# An die eigene Umgebung anpassen: venv und Pfad zum Repo.
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

RUNS="${RUNS:?RUNS fehlt, etwa RUNS=66}"
SPLIT="${SPLIT:?SPLIT fehlt, etwa SPLIT=val}"
echo "Daten: $EVENTPIV_DATA"

# Etwa EXTRA_SETS="solver.cost_offset=-1;solver.cost_offset=-0.5".
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