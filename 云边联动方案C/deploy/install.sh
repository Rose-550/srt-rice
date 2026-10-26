#!/usr/bin/env bash
# 稻视界 · 中转服务一键部署（阿里云 ECS / Ubuntu 20.04+ 或 Alibaba Cloud Linux）
#
# 用法（在服务器上以 root 执行）：
#   scp -r 云边联动方案C relay_server.py root@你的服务器:/tmp/rice-deploy/
#   bash /tmp/rice-deploy/deploy/install.sh
#
# 脚本只做四件事：建目录和用户、放代码、装 systemd 单元、启动。
# Nginx 与证书见 deploy/README.md（需要域名解析好之后再做）。
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR=/opt/rice-relay
SERVICE=rice-relay

echo "==> 1/4 准备目录与用户"
id -u rice >/dev/null 2>&1 || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin rice
mkdir -p "$APP_DIR/data"
chown -R rice:rice "$APP_DIR"

echo "==> 2/4 放代码"
install -m 644 "$SRC_DIR/relay_server.py" "$APP_DIR/relay_server.py"
install -m 644 "$SRC_DIR/deploy/rice-relay.service" "/etc/systemd/system/$SERVICE.service"
if [ ! -f "$APP_DIR/relay.env" ]; then
  install -m 600 "$SRC_DIR/deploy/relay.env.example" "$APP_DIR/relay.env"
  echo "    已生成 $APP_DIR/relay.env —— 请先改成你自己的口令与域名！"
fi
chown -R rice:rice "$APP_DIR"

echo "==> 3/4 说明安全组"
echo "    阿里云控制台 → 安全组 → 入方向放行 80 与 443（9000 不必对外，Nginx 走本机回环）"

echo "==> 4/4 启动服务"
systemctl daemon-reload
systemctl enable --now "$SERVICE"
sleep 1
systemctl --no-pager --lines=8 status "$SERVICE" || true

echo
echo "本机自测：curl -s http://127.0.0.1:9000/api/ping"
echo "配好 Nginx 与证书后，用电脑端跑部署自检："
echo "    python3 check_relay.py https://your-domain.com 你的口令"
