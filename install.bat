@echo off
chcp 65001 >nul 2>&1
title VRC Auto Fish - Install
cd /d "%~dp0"

echo ============================================
echo   VRC Auto Fish - Portable Installer
echo ============================================
echo.

set PYTHON_DIR=%~dp0python
set PYTHON_EXE=%PYTHON_DIR%\python.exe
set PYTHON_VER=3.10.11
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VER%/python-%PYTHON_VER%-amd64.exe
set INSTALLER_EXE=%~dp0python-installer.exe

:: ============================================
::  Step 1: Get Python
:: ============================================
if exist "%PYTHON_EXE%" (
    echo [OK] Local Python found: %PYTHON_DIR%
    goto :install_deps
)

echo [1/3] Downloading Python %PYTHON_VER%...
echo       (Full installer ~25 MB, includes tkinter and all standard modules)
echo.

curl --version >nul 2>&1
if errorlevel 1 (
    echo   Using PowerShell to download...
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%INSTALLER_EXE%' -UseBasicParsing }"
) else (
    curl -L --progress-bar -o "%INSTALLER_EXE%" "%PYTHON_URL%"
)

if not exist "%INSTALLER_EXE%" (
    echo [ERROR] Download failed. Check internet connection and try again.
    pause
    exit /b 1
)

echo.
echo   Installing Python to %PYTHON_DIR% ...

:: Pre-create target dir — installer requires it to exist on some systems
mkdir "%PYTHON_DIR%" >nul 2>&1

:: TargetDir must be quoted as "TargetDir=path" (not TargetDir="path")
:: All args on one line — ^ continuation can corrupt arguments
:: /passive = progress bar, no user input, installer handles UAC itself
"%INSTALLER_EXE%" /passive /log "%~dp0python-install.log" "TargetDir=%PYTHON_DIR%" InstallAllUsers=0 PrependPath=0 Shortcuts=0 Include_launcher=0 Include_test=0 Include_doc=0 AssociateFiles=0

del "%INSTALLER_EXE%" >nul 2>&1

:: --------------------------------------------------------
:: Fallback: if same Python version is already installed,
:: the installer enters Modify mode and ignores TargetDir.
:: In that case, copy from the existing installation.
:: --------------------------------------------------------
if not exist "%PYTHON_EXE%" (
    echo   Installer did not create local copy, looking for existing Python 3.10...
    set "FOUND_PYTHON="
    if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "FOUND_PYTHON=%LOCALAPPDATA%\Programs\Python\Python310"
    if exist "%ProgramFiles%\Python310\python.exe" set "FOUND_PYTHON=%ProgramFiles%\Python310"
)
if not exist "%PYTHON_EXE%" if defined FOUND_PYTHON (
    echo   Found at %FOUND_PYTHON% - copying to project...
    xcopy "%FOUND_PYTHON%" "%PYTHON_DIR%\" /E /I /Q /Y >nul
)

if not exist "%PYTHON_EXE%" (
    echo.
    echo [ERROR] Python installation failed!
    echo.
    echo   Install log saved to: %~dp0python-install.log
    echo   Open it to see the exact error from the installer.
    echo.
    echo   Common fixes:
    echo     - Temporarily disable antivirus and run install.bat again
    echo     - Make sure you have at least 500 MB free on this drive
    echo     - Move the folder closer to drive root, e.g.:
    echo         D:\vrc-fish\  instead of  C:\Users\...\Desktop\...
    echo.
    pause
    exit /b 1
)

del "%~dp0python-install.log" >nul 2>&1

echo [OK] Python %PYTHON_VER% installed to python\
echo.

:: ============================================
::  Step 2: Install pip + dependencies
:: ============================================
:install_deps
echo [2/3] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip --quiet --no-warn-script-location
echo.

:: PyTorch: auto-detect GPU
echo [3/3] Detecting GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   No NVIDIA GPU detected - installing CPU PyTorch
    "%PYTHON_EXE%" -m pip install torch torchvision ^
        --index-url https://download.pytorch.org/whl/cpu ^
        --no-warn-script-location
) else (
    echo   NVIDIA GPU detected - installing CUDA 12.8 PyTorch
    "%PYTHON_EXE%" -m pip install torch torchvision ^
        --index-url https://download.pytorch.org/whl/cu128 ^
        --no-warn-script-location
)
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed!
    pause
    exit /b 1
)

echo.
echo   Installing remaining packages...
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt" --no-warn-script-location
if errorlevel 1 (
    echo [ERROR] Dependency installation failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Done! Run start.bat to launch
echo ============================================
echo.
pause
