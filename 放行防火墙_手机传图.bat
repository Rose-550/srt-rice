@echo off
chcp 65001 >nul
title 放行防火墙 - 手机传图
:: 自动请求管理员权限（会弹 UAC，点「是」）
net session >nul 2>&1
if %errorLevel% neq 0 (
  echo 正在请求管理员权限，请在弹窗里点「是」…
  powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
echo ============================================================
echo  放行 Windows 防火墙的 8766 端口（手机传图用）
echo ============================================================
netsh advfirewall firewall delete rule name="RicePhoneUpload" >nul 2>&1
netsh advfirewall firewall add rule name="RicePhoneUpload" dir=in action=allow protocol=TCP localport=8766
netsh advfirewall firewall add rule name="RicePhoneUpload-UDP" dir=in action=allow protocol=UDP localport=8766 >nul 2>&1
echo.
echo 已添加规则。当前 8766 相关规则：
netsh advfirewall firewall show rule name="RicePhoneUpload" | findstr /C:"规则名称" /C:"Rule Name" /C:"已启用" /C:"Enabled" /C:"LocalPort" /C:"本地端口"
echo.
echo 现在让手机浏览器打开服务窗口里显示的那个地址（形如 http://192.168.x.x:8766/）。
echo 若仍打不开，说明是网络层面（校园网设备隔离），请改用手机热点方案。
echo.
pause