#!/bin/bash
#SBATCH --job-name=eventpiv-pairs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# --time haengt an der Zahl der Runs in pairs.runs; je Aufruf setzen.
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# [2] Paar-Datensatz: ruft scripts/build_pair_dataset.py einmal auf. Die
# Runs setzt pairs.runs (Gruppe pairsampling), keine Variable.
#   EXTRA  frei, Overrides, durch Leerzeichen getrennt.
# EVENTPIV_DATA und EVENTPIV_OUT muessen gesetzt sein.
# Allgemeines zu sbatch steht im slurm-Abschnitt von scripts/_cli.py.
# Bedarf: GPU fuer den Jaccard je Fenster, dicht als n^2 * 4 Byte; auf
# dichten Daten ueber zehn GB VRAM. Arbeitsspeicher waechst mit ja_frac.
# Beispiel:
#   sbatch slurm/pairs.sbatch
#   EXTRA="neighborhood=batch1 pairsampling=batch1" sbatch slurm/pairs.sbatch
# Fehler:
#   Labels fehlen: erst slurm/labels.sbatch fuer dieselben Runs.
#   Nach einer Aenderung an neighborhood.* alles ab [2] neu bauen.
#   max_no und ja_vorrat begrenzen die Datei, nicht den Speicher beim Sammeln.

set -euo pipefail          

export PYTHONUNBUFFERED=1
export WANDB_SILENT=true  

# An die eigene Umgebung anpassen: venv und Pfad zum Repo.
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

echo "Node: $(hostname)   Job: $SLURM_JOB_ID   Start: $(date)"
python -m eventpiv.kern.config          # Pfade zeigen -- erster Verdacht bei Problemen

# EXTRA ungequotet: die Overrides sollen in einzelne Argumente zerfallen.
EXTRA="${EXTRA:-}"
# shellcheck disable=SC2086
python scripts/build_pair_dataset.py $EXTRA

echo "Ende: $(date)"