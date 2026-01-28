@echo off
cd /d %~dp0
call venv\Scripts\activate
python api_server.py
