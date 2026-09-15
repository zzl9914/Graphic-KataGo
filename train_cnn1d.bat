@echo off
call "%~dp0..\start_train.bat" cnn1d --sim 256 --workers 1 --gpw 32 --steps 16 --buffer-drop-from-round 5 --lr 1e-4 --value-weight 30 --own-weight 5 --q-lambda 0.5 --temperature 0.1 --infinite
