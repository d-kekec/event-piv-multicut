#!/bin/bash
#SBATCH --job-name=eventpiv-stufe2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# [6b] Stufe-2-Training: ruft scripts/train_stufe2.py auf, je Satz ein Lauf.
#   OFFSET       frei, solver.cost_offset der Quelle aus Stufe 1, Vorgabe 0.
#   KD_RUNS      frei, ersetzt kandidaten.runs und damit den Slug.
#   NOT_HOLDOUT  frei, Runs, die nicht in den Holdout duerfen (io.train_fest).
#   EXTRA        frei, Overrides, durch Leerzeichen getrennt.
#   EXTRA_SETS   frei, mehrere Saetze, durch ; getrennt, je Satz ein Lauf.
# EVENTPIV_DATA und EVENTPIV_OUT muessen gesetzt sein.
# Allgemeines zu sbatch steht im slurm-Abschnitt von scripts/_cli.py.
# Bedarf: GPU; der Knotentensor liegt im VRAM, auf dichten Daten um 20 GB --
# dafuer die Karte im Beispiel.
# Beispiel:
#   sbatch --gres=gpu:rtx5090:1 slurm/stufe2.sbatch
#   EXTRA="datensatz=batch1 kandidatensatz=batch1 transformersatz=batch1" \
#     sbatch --gres=gpu:rtx5090:1 slurm/stufe2.sbatch
# Fehler:
#   batch1 ohne transformersatz=batch1: still ein Netz mit den Vorgaben.
#   Kandidaten fehlen: alle kandidaten.runs mit demselben OFFSET bauen.
#   Netz existiert: Abbruch statt Ueberschreiben, io.overwrite=true erzwingt.
#
# --gres hier SCHON, anders als bei train.sbatch: das MLP von Stufe 1
# rechnet auf der CPU, der Transformer hat um Groessenordnungen mehr
# Parameter und schiebt je Stapel transformer.batch x (2*n_events+1) Tokens
# durch die Attention-Schichten.
#
# --time ist eine Obergrenze; die Sekunden je Epoche stehen im Log
# (zeile["sec"]), danach anpassen.
#
# --cpus-per-task=8: die Paar-Indizes werden je Stapel auf der CPU
# geschnitten und uebertragen, waehrend die Karte rechnet.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
# An die eigene Umgebung anpassen: venv und Pfad zum Repo.
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

# OFFSET setzt solver.cost_offset, den Betriebspunkt von Stufe 1 und damit
# die QUELLE: die Tracklets [5a], auf denen die Kandidaten gebaut wurden.
# Er steht im DATEINAMEN von Tracklets, Kandidaten und Netz, nicht im Slug
# -- ein falscher Wert findet die Datei nicht und bricht mit ihrem Pfad ab.
OFFSET="${OFFSET:-0}"
# KD_RUNS wie beim Kandidatenbau; sortiert, weil die Reihenfolge im Hash steht.
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

# Etwa EXTRA_SETS="transformer.d_model=32;transformer.d_model=64".
EXTRA_SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra SAETZE <<< "$EXTRA_SETS"
# Leere Eingabe: ein leerer Satz, sonst liefe die Schleife null Mal.
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