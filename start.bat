@echo off
chcp 65001 >nul 2>&1
title VRC Auto Fish
cd /d "%~dp0"

set PYTHON_EXE=%~dp0python\python.exe

:: ============================================
::  Check local Python
:: ============================================
if not exist "%PYTHON_EXE%" (
    echo ============================================
    echo   Local Python not found!
    echo   Running install.bat first...
    echo ============================================
    echo.
    call install.bat
    if errorlevel 1 (
        echo [ERROR] Installation failed. Fix errors above and try again.
        pause
        exit /b 1
    )
    echo.
)

:: ============================================
::  Launch
:: ============================================
echo Starting VRC Auto Fish...
echo.
"%PYTHON_EXE%" main.py
if errorlevel 1 pause
