@echo off
REM Double-click launcher for Segmentation Annotation Studio on Windows.
REM Runs start_all.ps1 with the execution policy bypassed for this process only
REM (no global policy change). Any args are forwarded, e.g. set PROD=1 first.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_all.ps1" %*
