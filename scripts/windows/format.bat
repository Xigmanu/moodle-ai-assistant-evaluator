@echo off
setlocal

rem
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "BIN=%ROOT%\.venv\Scripts"
set "TARGET=%ROOT%\src\evaluator"

"%BIN%\autoflake.exe" --in-place --recursive --remove-all-unused-imports "%TARGET%" || exit /b 1
"%BIN%\isort.exe" --profile black "%TARGET%" || exit /b 1
"%BIN%\black.exe" "%TARGET%" || exit /b 1
