@echo off
rem Rebuilds tables, figures and Online Resources from the stored results (minutes).
setlocal
set NJ=%1
if "%NJ%"=="" set NJ=6
set QK_TIE=middle
set QK_S_GRID=0.002,0.005,0.01,0.02,0.05,0.1,0.25,0.5,1,2,4,8
set QSD=moons circles breast_cancer iris wine digits38 digits17 digits49 ionosphere sonar banknote pima blood haberman ilpd spambase
cd /d "%~dp0code"
set OUT=..\results
if not exist %OUT% mkdir %OUT%
echo [surrogate_analyse]
python surrogate_study.py --out %OUT% --analyse-only > %OUT%\log_surrogate_analyse.txt 2>&1 || goto :fail
echo [lever_table]
python lever_table.py --results %OUT% > %OUT%\log_lever_table.txt 2>&1 || goto :fail
echo [v2_analysis]
python v2_analysis.py --results %OUT% > %OUT%\log_v2_analysis.txt 2>&1 || goto :fail
echo [online_resources]
python make_online_resources.py --results %OUT% --out ..\online_resources > %OUT%\log_online_resources.txt 2>&1 || goto :fail
echo [figures]
python make_figures.py --results %OUT% --out %OUT%\figs > %OUT%\log_figures.txt 2>&1 || goto :fail
echo [tables]
python make_tables.py --results %OUT% --out %OUT%\tables > %OUT%\log_tables.txt 2>&1 || goto :fail
echo DONE
goto :eof
:fail
echo FAILED - see the last log file in results
exit /b 1
