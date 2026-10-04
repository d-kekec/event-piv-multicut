#!/bin/bash
#SBATCH --job-name=eventpiv-anker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Die Regressionsanker als Job -- fuer Datensaetze, die rema nicht traegt.
# Welcher Datensatz, kommt aus der Umgebung beim Absenden: eventpiv_use
# setzt EVENTPIV_NAME, EVENTPIV_DATA und EVENTPIV_OUT, und sbatch reicht
# die Umgebung weiter.
#
# MARKER waehlt die Stufe, Vorgabe beide: MARKER=anker2 sbatch ...

set -euo pipefail
export PYTHONUNBUFFERED=1
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

echo "Node $(hostname)  Job ${SLURM_JOB_ID}  Datensatz ${EVENTPIV_NAME:?eventpiv_use vor dem Absenden}"
pytest -m "${MARKER:-anker or anker2}" -v -rs