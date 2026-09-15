@echo off
call "%~dp0..\start_train.bat" gnn --rules gomoku --sim 800 --workers 1 --gpw 16 --lr 1e-3 --no-arena --infinite
