#!/bin/bash
#SBATCH --job-name=eventpiv-stufe2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --gres hier SCHON, anders als bei train.sbatch: das Stufe-1-Netz hat 41
# Parameter und rechnet auf der CPU, dieses hier rund 105.000 und schiebt
# je Stapel 1024 x 65 Tokens durch zwei Attention-Schichten.
#
# --time=03:00:00 ist eine OBERGRENZE, keine Schaetzung: der Probelauf mit
# zwei Epochen lief durch, die Sekunden je Epoche stehen ab dem naechsten
# Lauf im Log (zeile["sec"]). Danach herunterziehen.
#
# --cpus-per-task=8: die Paar-Indizes werden je Stapel auf der CPU
# geschnitten und uebertragen, waehrend die Karte rechnet.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

# Der Arbeitspunkt von Stufe 1, auf dessen Tracklets die Kandidaten
# gebaut wurden. Steht im DATEINAMEN beider Artefakte, nicht im Slug --
# ein falscher Wert findet die Datei nicht und bricht mit ihrem Pfad ab.
OFFSET="${OFFSET:-0}"
# Dieselbe Liste wie beim Kandidatenbau -- sie IST der Pfad.
KD_RUNS="${KD_RUNS:-}"
KD_ARG=""
if [ -n "$KD_RUNS" ]; then
  KD_ARG="kandidaten.runs=[$(echo "$KD_RUNS" | tr ' ' '\n' | sort -n | paste -sd,)]"
fi
# Runs, die NICHT in den Holdout gezogen werden duerfen. Aufsteigend
# sortiert, weil die Reihenfolge in den Transformer-Hash eingeht.
NOT_HOLDOUT="${NOT_HOLDOUT:-}"
FEST_ARG=""
if [ -n "$NOT_HOLDOUT" ]; then
  FEST_ARG="io.train_fest=[$(echo "$NOT_HOLDOUT" | tr ' ' '\n' | sort -n | paste -sd,)]"
fi

# Mehrere Parametersaetze in EINER Allokation, Semikolon getrennt:
#   EXTRA_SETS="transformer.label_modus=streng;transformer.label_modus=weich"
EXTRA_SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra SAETZE <<< "$EXTRA_SETS"
if [ ${#SAETZE[@]} -eq 0 ]; then SAETZE=(""); fi

echo "Node $(hostname)  Job $SLURM_JOB_ID  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for e in "${SAETZE[@]}"; do
  echo "=== cost_offset=$OFFSET  EXTRA='$e' ==="
  # shellcheck disable=SC2086
    python scripts/train_stufe2.py solver.cost_offset="$OFFSET" $KD_ARG $FEST_ARG $e \
    || echo "EXTRA='$e' FEHLGESCHLAGEN -- weiter" >&2
done
echo "Ende: $(date)"