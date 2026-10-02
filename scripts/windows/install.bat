@echo off
setlocal

rem
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
if not defined PYTHON set "PYTHON=python"

set "PROFILE=prod"
if "%~1"=="--dev" (
    set "PROFILE=dev"
) else if not "%~1"=="" (
    echo unknown argument %~1 1>&2
    exit /b 2
)

if "%PROFILE%"=="dev" (
    set "WORKSPACE=%ROOT%\.volteval"
) else if defined LOCALAPPDATA (
    set "WORKSPACE=%LOCALAPPDATA%\.volteval"
) else (
    set "WORKSPACE=%USERPROFILE%\AppData\Local\.volteval"
)

if not exist "%WORKSPACE%\experiments" mkdir "%WORKSPACE%\experiments" || exit /b 1

for %%F in ("%ROOT%\config\experiments\*.yaml" "%ROOT%\config\experiments\*.yml") do (
    call :seed_file "%%~fF" "%WORKSPACE%\experiments\%%~nxF" || exit /b 1
)

call :seed_file "%ROOT%\config\config.example.toml" "%WORKSPACE%\config.toml" || exit /b 1
call :seed_file "%ROOT%\.env.example" "%WORKSPACE%\.env" || exit /b 1

set "VENV=%ROOT%\.venv"
if "%PROFILE%"=="dev" goto :install_dev

%PYTHON% -m pip install --user --upgrade "%ROOT%" || exit /b 1
exit /b 0

:install_dev
if not exist "%VENV%\Scripts\python.exe" (
    %PYTHON% -m venv "%VENV%" || exit /b 1
)
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
"%VENV%\Scripts\python.exe" -m pip install -e "%ROOT%[dev]" || exit /b 1
exit /b 0

rem
:seed_file
if "%PROFILE%"=="dev" goto :seed_copy
if exist "%~2" exit /b 0
:seed_copy
copy /y "%~1" "%~2" >nul
exit /b
