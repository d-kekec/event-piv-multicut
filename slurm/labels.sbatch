#!/bin/bash
#SBATCH --job-name=eventpiv-labels
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
# Je TASK, nicht je Run -- mit PRO_TASK>1 entsprechend heraufsetzen.
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# [1] Labels: ruft scripts/build_event_labels.py auf, je Array-Task ein
# Python-Lauf ueber PRO_TASK Runs.
#   SPLIT     Pflicht, der Split.
#   RUNS      frei, Runs durch Leerzeichen getrennt; ohne alle Runs des Splits.
#   PRO_TASK  frei, Runs je Array-Task, Vorgabe 1.
#   EXTRA     frei, Overrides, durch Leerzeichen getrennt.
# EVENTPIV_DATA und EVENTPIV_OUT muessen gesetzt sein.
# Allgemeines zu sbatch steht im slurm-Abschnitt von scripts/_cli.py.
# Bedarf: nur CPU; auf dichten Daten um zwanzig GB Arbeitsspeicher je Task.
# --array=0 bis ceil(Runs/PRO_TASK)-1 ist Pflicht; --time gilt je Task.
# Beispiel:
#   SPLIT=train RUNS="2 6 8" PRO_TASK=3 sbatch --array=0 slurm/labels.sbatch
#   SPLIT=train RUNS="32" EXTRA="labeling=batch1" sbatch --array=0 \
#     slurm/labels.sbatch
# Fehler:
#   batch1 ohne EXTRA="labeling=batch1": Labels mit STELLA-Parametern, still.
#   Ohne --array: set -u bricht an SLURM_ARRAY_TASK_ID ab.
#   LabelingMismatch: Labels passen nicht; neu mit EXTRA="io.overwrite=true".

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
# An die eigene Umgebung anpassen: venv und Pfad zum Repo.
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

: "${SPLIT:?SPLIT fehlt -- direkt vor sbatch setzen, etwa SPLIT=train}"
echo "Daten: $EVENTPIV_DATA"

# Zusaetzliche Hydra-Overrides, z.B. EXTRA="io.overwrite=true" beim
# Neubau nach einer Formataenderung. BEWUSST nicht fest eingebaut:
# io.overwrite=true schaltet die LabelingMismatch-Pruefung ab, und die
# soll nur dann fallen, wenn jemand sie ausdruecklich faellen laesst.
EXTRA="${EXTRA:-}"

if [[ -n "${RUNS:-}" ]]; then
    read -ra RUN_ARR <<< "$RUNS"
else
    mapfile -t RUN_ARR < <(python -c "
from eventpiv.kern import config
print('\n'.join(str(r) for r in config.available_runs('$SPLIT')))")
fi

# Runs je Task. Ein Array mit einem Job je Run ist bei hundert Runs
# unuebersichtlich und kostet hundertmal Import und Prozessstart.
# Uebergeben wird die ganze Scheibe an EINEN Python-Lauf: build_labels
# schleift selbst darueber und sammelt fehlgeschlagene Runs ein.
PRO_TASK="${PRO_TASK:-1}"
start=$(( SLURM_ARRAY_TASK_ID * PRO_TASK ))

if (( start >= ${#RUN_ARR[@]} )); then
    echo "Task $SLURM_ARRAY_TASK_ID liegt ausserhalb der Liste" \
         "(${#RUN_ARR[@]} Runs im Split $SPLIT, $PRO_TASK je Task)" >&2
    exit 1
fi

MEINE=("${RUN_ARR[@]:start:PRO_TASK}")
LISTE=$(IFS=,; echo "${MEINE[*]}")
echo "Node $(hostname)  Array $SLURM_ARRAY_JOB_ID  Task $SLURM_ARRAY_TASK_ID"
echo "-> $SPLIT / runs [$LISTE]   EXTRA='$EXTRA'   Start: $(date)"

# shellcheck disable=SC2086 -- EXTRA soll in einzelne Argumente zerfallen
python scripts/build_event_labels.py selection.split="$SPLIT" \
    "selection.runs=[$LISTE]" $EXTRA

echo "Ende: $(date)"