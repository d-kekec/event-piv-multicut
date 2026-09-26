#!/bin/bash
#SBATCH --job-name=eventpiv-labels
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
# Je TASK, nicht je Run -- mit PRO_TASK>1 entsprechend heraufsetzen.
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err


set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

: "${SPLIT:?SPLIT nicht gesetzt -- vor dem sbatch exportieren}"

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