#!/bin/bash
#SBATCH --job-name=eventpiv-anker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Regressionsanker (tests/test_anker.py) als Job: ruft pytest mit Marker auf.
#   EVENTPIV_NAME  Pflicht, der Datensatz (stella, batch1); waehlt die Anker.
#   MARKER         frei, anker (Stufe 1) oder anker2 (Stufe 2), Vorgabe beide.
# EVENTPIV_DATA und EVENTPIV_OUT muessen gesetzt sein, passend zu
# EVENTPIV_NAME.
# Allgemeines zu sbatch steht im slurm-Abschnitt von scripts/_cli.py.
# Bedarf: GPU; Stufe 2 loest einen ganzen Run wie [6c] und braucht
# entsprechend viel Arbeitsspeicher. Die Karte im Beispiel waehlt dafuer den
# Knoten mit viel Arbeitsspeicher.
# Beispiel:
#   EVENTPIV_NAME=stella MARKER=anker2 sbatch --gres=gpu:rtx5090:1 \
#     slurm/anker.sbatch
# Fehler:
#   Uebersprungen statt geprueft: Netz oder Kandidaten fehlen, Grund im Log.
#   Kein Anker fuer EVENTPIV_NAME: jeder Fall wird uebersprungen.

set -euo pipefail
export PYTHONUNBUFFERED=1
# An die eigene Umgebung anpassen: venv und Pfad zum Repo.
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

echo "Node $(hostname)  Job ${SLURM_JOB_ID}  Datensatz ${EVENTPIV_NAME:?EVENTPIV_NAME fehlt: stella oder batch1, passend zu EVENTPIV_DATA und EVENTPIV_OUT}"
pytest -m "${MARKER:-anker or anker2}" -v -rs