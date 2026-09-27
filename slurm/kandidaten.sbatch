#!/bin/bash
#SBATCH --job-name=eventpiv-kandidaten
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# KEIN --gres: build_kandidaten.py importiert torch nicht. Gerechnet wird
# der KDTree (scipy) und connected_components. Ohne GRES-Anforderung kommt
# der Job schneller dran.
#
# Ein Prozess fuer ALLE Runs, anders als bei tracklets.sbatch: der
# Speicher wird nach jedem Run wieder frei. Die Spitze je Run waechst mit
# seiner Eventzahl und mit kandidaten.alpha, auf dichten Daten auf mehrere
# zehn GB -- dann per -w die Node waehlen.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

RUNS="${RUNS:-2 11 19 30 32 35 44 59 61 76}"
SPLIT="${SPLIT:-train}"
# Kommaliste fuer Hydra: selection.runs=[2,11,...]
LISTE=$(echo "$RUNS" | tr ' ' ',')
# Die IDENTITAET des Kandidatensatzes, getrennt von dem, WAS gebaut wird.
# selection.runs sagt, welche Runs entstehen; kandidaten.runs bestimmt den
# Slug und damit das Verzeichnis. Die Liste gehoert der Gruppe
# (kandidatensatz=...), auf batch1 den Vorgaben. KD_RUNS ueberschreibt sie
# nur, wenn es ausdruecklich gesetzt ist -- etwa fuer eine Probe unter
# eigenem Slug. Aufsteigend sortiert: die REIHENFOLGE geht in den Hash ein.
KD_ARG=()
if [ -n "${KD_RUNS:-}" ]; then
  KD_ARG=("kandidaten.runs=[$(echo "$KD_RUNS" | tr ' ' '\n' | sort -n | paste -sd,)]")
fi

# Mehrere Parametersaetze in EINER Allokation, Semikolon getrennt:
#   EXTRA_SETS="solver.cost_offset=-0.5;kandidaten.alpha=60000"
EXTRA_SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra SAETZE <<< "$EXTRA_SETS"
# Wie in tracklets.sbatch: leere Eingabe ergibt ein LEERES Array, die
# Schleife liefe null Mal und der Job endete wortlos mit Erfolg.
if [ ${#SAETZE[@]} -eq 0 ]; then SAETZE=(""); fi

echo "Node $(hostname)  Job $SLURM_JOB_ID  Start: $(date)"
for e in "${SAETZE[@]}"; do
  echo "=== EXTRA='$e' ==="
  # shellcheck disable=SC2086
    python scripts/build_kandidaten.py \
      selection.split="$SPLIT" "selection.runs=[$LISTE]" \
      ${KD_ARG[@]+"${KD_ARG[@]}"} $e \
    || echo "EXTRA='$e' FEHLGESCHLAGEN -- weiter" >&2
done
echo "Ende: $(date)"