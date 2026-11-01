#!/bin/bash
#SBATCH --job-name=eventpiv-solve2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# [6c] Stufe-2-Solve: ruft scripts/solve_stufe2.py je Satz und Versatz
# einmal fuer alle Runs auf.
#   RUNS        Pflicht, Runs durch Leerzeichen getrennt.
#   SPLIT       Pflicht, der Split.
#   OFFSET      frei, Werte fuer solver2.cost_offset, durch Leerzeichen.
#   KD_RUNS     frei, ersetzt kandidaten.runs und damit den Pfad zum Netz.
#   EXTRA       frei, Overrides, durch Leerzeichen getrennt.
#   EXTRA_SETS  frei, mehrere Saetze, durch ; getrennt, je Satz ein Lauf.
# EVENTPIV_DATA und EVENTPIV_OUT muessen gesetzt sein.
# Allgemeines zu sbatch steht im slurm-Abschnitt von scripts/_cli.py.
# Bedarf: GPU; den Spitzenspeicher setzt die Bewertung (IDF1), auf dichten
# Daten mehrere zehn GB Arbeitsspeicher. Die Karte im Beispiel waehlt dafuer
# den Knoten mit viel Arbeitsspeicher.
# Beispiel:
#   RUNS="8 13" SPLIT=train sbatch --gres=gpu:rtx5090:1 slurm/solve2.sbatch
#   EXTRA="datensatz=batch1 kandidatensatz=batch1 transformersatz=batch1" \
#     RUNS="32" SPLIT=train sbatch --gres=gpu:rtx5090:1 slurm/solve2.sbatch
# Fehler:
#   batch1 ohne transformersatz=batch1: still das Netz mit den Vorgaben,
#   wenn eines gebaut ist.
#   Kandidaten fehlen: erst slurm/kandidaten.sbatch mit demselben SPLIT.
#   WARN zu kandidaten.split: einen anderen Split waehlt SPLIT, nicht dieses.
#
# solve_stufe2.py schreitet die Runs selbst ab und oeffnet je Run einen
# W&B-Lauf. --gres, weil die Kantenkosten auf der Karte entstehen und RAMA
# die Karte ohnehin braucht.
#
# EXTRA_SETS waehlt hier das NETZ, nicht den Solver: mehrere Transformer
# nacheinander gegen dieselben Runs. Den Versatz des Solvers waehlt OFFSET.

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
# An die eigene Umgebung anpassen: venv und Pfad zum Repo.
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

RUNS="${RUNS:?RUNS fehlt, etwa RUNS=66}"
SPLIT="${SPLIT:?SPLIT fehlt, etwa SPLIT=val}"
echo "Daten: $EVENTPIV_DATA"
LISTE=$(echo "$RUNS" | tr ' ' ',')

# KD_RUNS wie beim Kandidatenbau, sortiert; die Liste IST der Pfad zum Netz.
KD_RUNS="${KD_RUNS:-}"
KD_ARG=""
if [ -n "$KD_RUNS" ]; then
  KD_ARG="kandidaten.runs=[$(echo "$KD_RUNS" | tr ' ' '\n' | sort -n | paste -sd,)]"
fi

EXTRA_SETS="${EXTRA_SETS:-${EXTRA:-}}"
IFS=';' read -ra SAETZE <<< "$EXTRA_SETS"
# Leere Eingabe: ein leerer Satz, sonst liefe die Schleife null Mal.
if [ ${#SAETZE[@]} -eq 0 ]; then SAETZE=(""); fi

# OFFSET setzt solver2.cost_offset, den Versatz des Stufe-2-Solvers,
# leerzeichengetrennt. Leer = ein Lauf mit dem Wert aus der Konfiguration.
# Der Versatz steht in solver2.short() und damit im Dateinamen -- die
# Laeufe ueberschreiben sich nicht.
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