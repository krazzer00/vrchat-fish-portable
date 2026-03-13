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
set INSTALLER_EXE=python-installer.exe

:: ============================================
::  Step 1: Get Python
:: ============================================
if exist "%PYTHON_EXE%" (
    echo [OK] Local Python found: %PYTHON_DIR%
    goto :fix_pth
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
"%INSTALLER_EXE%" /quiet ^
    TargetDir="%PYTHON_DIR%" ^
    InstallAllUsers=0 ^
    PrependPath=0 ^
    Shortcuts=0 ^
    Include_launcher=0 ^
    Include_test=0 ^
    Include_doc=0 ^
    AssociateFiles=0

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python installation failed!
    del "%INSTALLER_EXE%" >nul 2>&1
    pause
    exit /b 1
)

del "%INSTALLER_EXE%" >nul 2>&1
echo [OK] Python %PYTHON_VER% installed to python\
echo.

:: ============================================
::  Step 2: Fix ._pth if embedded Python
::  (embedded Python ships with python310._pth
::   that blocks site-packages by default)
:: ============================================
:fix_pth
if exist "%PYTHON_DIR%\python310.zip" (
    echo [2/3] Embedded Python detected - enabling site-packages...
    (
        echo python310.zip
        echo .
        echo import site
    ) > "%PYTHON_DIR%\python310._pth"
    echo [OK] site-packages enabled
) else (
    echo [2/3] Full Python detected - no path fix needed
)
echo.

:: ============================================
::  Step 3: Install pip + dependencies
:: ============================================
echo [3/3] Installing dependencies...
echo.

:: Upgrade pip
echo   Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip --quiet --no-warn-script-location
echo.

:: PyTorch: auto-detect GPU
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
echo   Installing remaining packages (ultralytics + dependencies)...
"%PYTHON_EXE%" -m pip install -r requirements.txt --no-warn-script-location
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
