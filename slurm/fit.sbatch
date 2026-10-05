#!/bin/bash
#SBATCH --job-name=eventpiv-fit
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# KEIN --gres: der Fit ist eine Python-Schleife je Event auf einem Kern,
# torch wird nicht geladen. Ohne GRES-Anforderung kommt der Job schneller
# dran.
#
# fit_trajektorien.py schreitet die Runs selbst ab. Die Partition aus [6c]
# muss mit DENSELBEN Overrides entstanden sein -- sie IST der Pfad. Fehlt
# sie, meldet das Skript WARN und nimmt den naechsten Run.
#
# QC ist die Sweep-Achse, leerzeichengetrennt. Leer = ein Lauf mit dem
# Wert aus der Konfiguration. q_c steht in fit.short() und damit im
# Dateinamen -- die Laeufe ueberschreiben sich nicht.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

# Ohne Vorgabe: ein leeres selection.runs liesse das Skript ALLE Runs des
# Splits nehmen, und QC vervielfacht das noch.
RUNS="${RUNS:?RUNS fehlt, etwa RUNS=66}"
SPLIT="${SPLIT:?SPLIT fehlt, etwa SPLIT=val}"
LISTE=$(echo "$RUNS" | tr ' ' ',')

EXTRA_SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra SAETZE <<< "$EXTRA_SETS"
if [ ${#SAETZE[@]} -eq 0 ]; then SAETZE=(""); fi

QC="${QC:-}"
if [ -z "$QC" ]; then QC_LISTE=(""); else read -ra QC_LISTE <<< "$QC"; fi

echo "Node $(hostname)  Job $SLURM_JOB_ID  Start: $(date)"
# Welche DATEN gelesen werden, sagt die Umgebung (eventpiv_use), nicht
# die Gruppe. Die Zeile haelt es im Log fest; ist nichts gesetzt, bricht
# set -u hier ab statt einmal je Wert im Skript.
echo "Daten: $EVENTPIV_DATA"
for e in "${SAETZE[@]}"; do
  for q in "${QC_LISTE[@]}"; do
    QC_ARG=""
    [ -n "$q" ] && QC_ARG="fit.q_c=$q"
    echo "=== split=$SPLIT runs=$LISTE  q_c=${q:-<config>}  EXTRA='$e' ==="
    # shellcheck disable=SC2086
    python scripts/fit_trajektorien.py \
        selection.split="$SPLIT" "selection.runs=[$LISTE]" $QC_ARG $e \
      || echo "q_c='$q' EXTRA='$e' FEHLGESCHLAGEN -- weiter" >&2
  done
done
echo "Ende: $(date)"