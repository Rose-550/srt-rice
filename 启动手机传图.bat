@echo off
chcp 65001 >nul
title 稻视界 · 手机传图服务
cd /d "%~dp0"
echo ============================================================
echo  稻视界 · 手机传图（局域网直传原图）
echo ------------------------------------------------------------
echo  启动过程中如果 Windows 弹出「是否允许访问网络」，
echo  请务必点【允许访问】—— 不点的话手机就连不上。
echo ============================================================
echo.
echo  如果端口 8766 被占用（程序里已经开过），先关掉那个再运行本脚本。
echo.
pause
D:\anoconda3\envs\rice-gpu\python.exe checks\phone_upload_server.py
echo.
echo 服务已退出。
pause