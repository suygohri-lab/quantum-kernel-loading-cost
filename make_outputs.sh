#!/usr/bin/env bash
# Rebuilds tables, figures and Online Resources from the stored results (minutes).
set -e
NJ="${1:-6}"
export QK_TIE=middle
export QK_S_GRID=0.002,0.005,0.01,0.02,0.05,0.1,0.25,0.5,1,2,4,8
QSD="moons circles breast_cancer iris wine digits38 digits17 digits49 ionosphere sonar banknote pima blood haberman ilpd spambase"
cd "$(dirname "$0")/code"
OUT=../results
mkdir -p "$OUT"
echo "[surrogate_analyse]"
python surrogate_study.py --out "$OUT" --analyse-only > "$OUT/log_surrogate_analyse.txt" 2>&1
echo "[lever_table]"
python lever_table.py --results "$OUT" > "$OUT/log_lever_table.txt" 2>&1
echo "[v2_analysis]"
python v2_analysis.py --results "$OUT" > "$OUT/log_v2_analysis.txt" 2>&1
echo "[online_resources]"
python make_online_resources.py --results "$OUT" --out ../online_resources > "$OUT/log_online_resources.txt" 2>&1
echo "[figures]"
python make_figures.py --results "$OUT" --out "$OUT"/figs > "$OUT/log_figures.txt" 2>&1
echo "[tables]"
python make_tables.py --results "$OUT" --out "$OUT"/tables > "$OUT/log_tables.txt" 2>&1
echo DONE
