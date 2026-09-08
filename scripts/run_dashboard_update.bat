@echo off
rem Wrapper invoked by the "NFL Fantasy Dashboard Update" Windows Scheduled
rem Task (Tuesdays 7:00 AM). Runs update_dashboard.py and logs output, since
rem a scheduled run has no console to watch for errors (e.g. Google Drive
rem not mounted, expired OAuth token).

cd /d "C:\Users\user\repos\nfl_research_2026"
if not exist "logs" mkdir "logs"

echo. >> "logs\update_dashboard.log"
echo ===== %date% %time% ===== >> "logs\update_dashboard.log"

"C:\Users\user\repos\nfl_research_2026\nfl_research_2026\Scripts\python.exe" "scripts\update_dashboard.py" >> "logs\update_dashboard.log" 2>&1

echo Exit code: %ERRORLEVEL% >> "logs\update_dashboard.log"
