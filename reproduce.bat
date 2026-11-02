@echo off
rem Re-runs every experiment (several hours), then rebuilds all outputs.  Usage: reproduce.bat [n_jobs]
setlocal
set NJ=%1
if "%NJ%"=="" set NJ=6
set QK_TIE=middle
set QK_S_GRID=0.002,0.005,0.01,0.02,0.05,0.1,0.25,0.5,1,2,4,8
set QSD=moons circles breast_cancer iris wine digits38 digits17 digits49 ionosphere sonar banknote pima blood haberman ilpd spambase
cd /d "%~dp0code"
set OUT=..\results
if not exist %OUT% mkdir %OUT%
echo [selftest]
python qkcommon.py --selftest > %OUT%\log_selftest.txt 2>&1 || goto :fail
echo [env]
python environment_report.py --out %OUT% > %OUT%\log_env.txt 2>&1 || goto :fail
echo [gate_counts]
python gate_counts.py --out %OUT% > %OUT%\log_gate_counts.txt 2>&1 || goto :fail
echo [loading]
python loading_counts.py --out %OUT% > %OUT%\log_loading.txt 2>&1 || goto :fail
echo [multi_dataset]
python multi_dataset_study.py --out %OUT% --n-jobs %NJ% --extra-baselines --diagnostics > %OUT%\log_multi_dataset.txt 2>&1 || goto :fail
echo [fixed_bandwidth]
python fixed_bandwidth_study.py --out %OUT% --n-jobs %NJ% > %OUT%\log_fixed_bandwidth.txt 2>&1 || goto :fail
echo [qubit_scaling]
python qubit_scaling_study.py --out %OUT% --n-jobs %NJ% --repeats 2 --extra-baselines --datasets %QSD% > %OUT%\log_qubit_scaling.txt 2>&1 || goto :fail
echo [reduction]
python reduction_study.py --out %OUT% --n-jobs %NJ% --latent > %OUT%\log_reduction.txt 2>&1 || goto :fail
echo [shots_device]
python shots_device_study.py --out %OUT% --part both --R 10 --R-device 10 --n-jobs %NJ% > %OUT%\log_shots_device.txt 2>&1 || goto :fail
echo [device_reanalysis]
python device_reanalysis.py --out %OUT% > %OUT%\log_device_reanalysis.txt 2>&1 || goto :fail
echo [frontier]
python cost_frontier.py --results %OUT% --acc-source mds --shots-file mds_shots.csv > %OUT%\log_frontier.txt 2>&1 || goto :fail
echo [surrogate]
python surrogate_study.py --out %OUT% --n-jobs %NJ% > %OUT%\log_surrogate.txt 2>&1 || goto :fail
echo [relabel]
python relabel_study.py --out %OUT% --n-jobs %NJ% > %OUT%\log_relabel.txt 2>&1 || goto :fail
echo [rel_edge]
python rel_edge_check.py --out %OUT% --n-jobs %NJ% > %OUT%\log_rel_edge.txt 2>&1 || goto :fail
echo [rel_surrogate]
python rel_surrogate.py --out %OUT% --n-jobs %NJ% > %OUT%\log_rel_surrogate.txt 2>&1 || goto :fail
echo [dose]
python dose_response.py --out %OUT% --n-jobs %NJ% > %OUT%\log_dose.txt 2>&1 || goto :fail
echo [rank]
python rank_check.py --out %OUT% --results %OUT% > %OUT%\log_rank.txt 2>&1 || goto :fail
echo [stationarity]
python stationarity_study.py --results %OUT% --out %OUT% > %OUT%\log_stationarity.txt 2>&1 || goto :fail
echo [bwl]
python bandwidth_labels.py --out %OUT% --n-jobs %NJ% > %OUT%\log_bwl.txt 2>&1 || goto :fail
call "%~dp0make_outputs.bat" || goto :fail
echo DONE
goto :eof
:fail
echo FAILED - see the last log file in results
exit /b 1
