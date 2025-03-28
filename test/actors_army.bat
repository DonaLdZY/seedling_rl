@echo off
for /l %%i in (1,1,16) do (
    start "Actor%%i" cmd /K "cd /D "E:\Files\Documents\seedling_rl" && python run_actor.py"
)