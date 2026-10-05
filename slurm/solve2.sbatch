#!/bin/bash
#SBATCH --job-name=eventpiv-solve2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# solve_stufe2.py schreitet die Runs selbst ab und oeffnet je Run einen
# W&B-Lauf. --gres, weil die Kantenkosten auf der Karte entstehen und RAMA
# die Karte ohnehin braucht.
#
# EXTRA_SETS waehlt das NETZ, nicht den Solver: mehrere Transformer
# nacheinander gegen dieselben Runs, semikolongetrennt.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

RUNS="${RUNS:?RUNS fehlt, etwa RUNS=66}"
SPLIT="${SPLIT:?SPLIT fehlt, etwa SPLIT=val}"
echo "Daten: $EVENTPIV_DATA"
LISTE=$(echo "$RUNS" | tr ' ' ',')

# Dieselbe Liste wie beim Kandidatenbau -- sie IST der Pfad zum Netz.
KD_RUNS="${KD_RUNS:-}"
KD_ARG=""
if [ -n "$KD_RUNS" ]; then
  KD_ARG="kandidaten.runs=[$(echo "$KD_RUNS" | tr ' ' '\n' | sort -n | paste -sd,)]"
fi

EXTRA_SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra SAETZE <<< "$EXTRA_SETS"
if [ ${#SAETZE[@]} -eq 0 ]; then SAETZE=(""); fi

# cost_offset des Stufe-2-Solvers, leerzeichengetrennt. Leer = ein Lauf
# mit dem Wert aus der Konfiguration. Der Offset steht in solver2.short()
# und damit im Dateinamen -- die Laeufe ueberschreiben sich nicht.
OFFSET="${OFFSET:-}"
if [ -z "$OFFSET" ]; then OFF_LISTE=(""); else read -ra OFF_LISTE <<< "$OFFSET"; fi

echo "Node $(hostname)  Job $SLURM_JOB_ID  Start: $(date)"
for e in "${SAETZE[@]}"; do
  for o in "${OFF_LISTE[@]}"; do
    OFF_ARG=""
    [ -n "$o" ] && OFF_ARG="solver2.cost_offset=$o"
    echo "=== split=$SPLIT runs=$LISTE  cost_offset=${o:-<config>}  EXTRA='$e' ==="
    # shellcheck disable=SC2086
    python scripts/solve_stufe2.py \
        selection.split="$SPLIT" "selection.runs=[$LISTE]" $KD_ARG $OFF_ARG $e \
      || echo "cost_offset='$o' EXTRA='$e' FEHLGESCHLAGEN -- weiter" >&2
  done
done
echo "Ende: $(date)"