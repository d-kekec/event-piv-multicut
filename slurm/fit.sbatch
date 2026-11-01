#!/bin/bash
#SBATCH --job-name=eventpiv-fit
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# [7] Fit: ruft scripts/fit_trajektorien.py je Satz und q_c einmal fuer alle
# Runs auf.
#   RUNS        Pflicht, Runs durch Leerzeichen getrennt.
#   SPLIT       Pflicht, der Split.
#   QC          frei, Werte fuer fit.q_c, durch Leerzeichen getrennt.
#   EXTRA       frei, Overrides, durch Leerzeichen getrennt.
#   EXTRA_SETS  frei, mehrere Saetze, durch ; getrennt, je Satz ein Lauf.
# EVENTPIV_DATA und EVENTPIV_OUT muessen gesetzt sein.
# Allgemeines zu sbatch steht im slurm-Abschnitt von scripts/_cli.py.
# Bedarf: nur CPU; Groessenordnung zehn GB Arbeitsspeicher. --time fuer
# mehrere dichte Runs hoeher setzen.
# Beispiel:
#   RUNS="8 13" SPLIT=train sbatch slurm/fit.sbatch
#   RUNS="8 13" SPLIT=train QC="1e6 1e7" sbatch slurm/fit.sbatch
# Fehler:
#   batch1 ohne fitsatz=batch1: still die STELLA-Vorgaben, sichtbar im Namen.
#   WARN, Partition fehlt: erst [6c] mit denselben Gruppen fuer diesen Run.
#
# KEIN --gres: der Fit ist eine Python-Schleife je Event auf einem Kern,
# torch wird nicht geladen.
#
# fit_trajektorien.py schreitet die Runs selbst ab. Die Partition aus [6c]
# muss mit DENSELBEN Overrides entstanden sein -- sie IST der Pfad.
#
# QC setzt fit.q_c und ist die Sweep-Achse, leerzeichengetrennt. Leer = ein
# Lauf mit dem Wert aus der Konfiguration. q_c steht in fit.short() und
# damit im Dateinamen -- die Laeufe ueberschreiben sich nicht.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
# An die eigene Umgebung anpassen: venv und Pfad zum Repo.
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

# Ohne Vorgabe: ein leeres selection.runs liesse das Skript ALLE Runs des
# Splits nehmen, und QC vervielfacht das noch.
RUNS="${RUNS:?RUNS fehlt, etwa RUNS=66}"
SPLIT="${SPLIT:?SPLIT fehlt, etwa SPLIT=val}"
LISTE=$(echo "$RUNS" | tr ' ' ',')

EXTRA_SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra SAETZE <<< "$EXTRA_SETS"
# Leere Eingabe: ein leerer Satz, sonst liefe die Schleife null Mal.
if [ ${#SAETZE[@]} -eq 0 ]; then SAETZE=(""); fi

QC="${QC:-}"
if [ -z "$QC" ]; then QC_LISTE=(""); else read -ra QC_LISTE <<< "$QC"; fi

echo "Node $(hostname)  Job $SLURM_JOB_ID  Start: $(date)"
# Welche DATEN gelesen werden, sagt EVENTPIV_DATA, nicht die Gruppe. Die
# Zeile haelt es im Log fest; ist nichts gesetzt, bricht set -u hier ab
# statt einmal je Wert im Skript.
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