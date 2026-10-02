@echo off
setlocal

rem
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"

set "VOLT_HOME=%ROOT%\.volteval"
"%ROOT%\.venv\Scripts\evaluator.exe" %*
exit /b %ERRORLEVEL%
