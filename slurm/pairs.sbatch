#!/bin/bash
#SBATCH --job-name=eventpiv-pairs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Gemessen 20.08. mit mehrdeutigen Endpunkten: 195 s (run32) bis 404 s
# (run61) je Run. --time je Aufruf setzen, es haengt an der Run-Zahl.
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail          

export PYTHONUNBUFFERED=1
export WANDB_SILENT=true  

# sbatch vererbt die Umgebung des aufrufenden Shells (--export=ALL ist Default),
# EVENTPIV_DATA/EVENTPIV_OUT sind also schon da. Das explizite source macht den
# Job unabhaengig davon, aus welcher Shell er abgesetzt wurde.
source "$HOME/eventpiv-env/bin/activate"
cd "$HOME/event-piv-multicut"

echo "Node: $(hostname)   Job: $SLURM_JOB_ID   Start: $(date)"
python -m eventpiv.kern.config          # Pfade zeigen -- erster Verdacht bei Problemen

# Overrides ueber EXTRA, wie in tracklets.sbatch -- EINE Konvention fuer
# alle Jobs. UNGEQUOTET, damit die Zeichenkette in Woerter zerfaellt;
# Hydra-Overrides enthalten keine Leerzeichen.
EXTRA="${EXTRA:-}"
# shellcheck disable=SC2086
python scripts/build_pair_dataset.py $EXTRA

echo "Ende: $(date)"