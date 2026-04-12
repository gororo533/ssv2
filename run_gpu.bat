@echo off
call conda activate gpu
cd /d %~dp0
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python -u %*
