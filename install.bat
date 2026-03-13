@echo off
chcp 65001 >nul 2>&1
title VRC Auto Fish - Install
cd /d "%~dp0"

:: ============================================
::  Self-elevate to Administrator if needed
::  (Python installer requires admin rights
::   to install to a custom TargetDir)
:: ============================================
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Requesting administrator privileges...
    echo        (required for Python installer)
    echo.
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs -WorkingDirectory '%~dp0'"
    exit /b
)

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
"%INSTALLER_EXE%" /quiet ^
    TargetDir="%PYTHON_DIR%" ^
    InstallAllUsers=0 ^
    PrependPath=0 ^
    Shortcuts=0 ^
    Include_launcher=0 ^
    Include_test=0 ^
    Include_doc=0 ^
    AssociateFiles=0

del "%INSTALLER_EXE%" >nul 2>&1

if not exist "%PYTHON_EXE%" (
    echo.
    echo [ERROR] Python installation failed!
    echo.
    echo   Possible reasons:
    echo     - Antivirus blocked the installer
    echo     - Insufficient disk space
    echo     - Path too long (try moving the folder closer to drive root)
    echo.
    echo   Try running install.bat again, or move the project to a shorter path
    echo   e.g. D:\vrc-fish\ instead of D:\vrchat-fish-portable-main\
    echo.
    pause
    exit /b 1
)

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
