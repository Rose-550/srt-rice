#!/usr/bin/env bash
# 稻视界 · 中转服务一键部署（Ubuntu/Debian 与 Alibaba Cloud Linux/CentOS 都适用）
# 用法：解压后 sudo bash setup.sh
set -e
APP=/opt/rice-relay
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> 1/6 安装依赖（python3 / nginx / certbot）"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq python3 nginx certbot python3-certbot-nginx
else
  (command -v dnf >/dev/null && dnf install -y python3 nginx certbot python3-certbot-nginx) || \
  (command -v yum >/dev/null && yum install -y python3 nginx certbot python3-certbot-nginx)
fi

echo "==> 2/6 读取配置"
read -rp "  服务器公网域名（已备案，如 rice.example.com）: " DOMAIN
read -rp "  电脑端拉取口令（自己定一串，别用弱口令）: " TOKEN

echo "==> 3/6 放文件到 $APP"
id -u rice >/dev/null 2>&1 || useradd --system --home $APP --shell /sbin/nologin rice
mkdir -p $APP/data
install -m 644 "$HERE/relay_server.py" $APP/relay_server.py
install -m 644 "$HERE/check_relay.py" $APP/check_relay.py
cat > $APP/relay.env <<EOF
RELAY_TOKEN=$TOKEN
RELAY_PUBLIC_BASE=https://$DOMAIN
EOF
chmod 600 $APP/relay.env
chown -R rice:rice $APP

echo "==> 4/6 装 systemd 服务"
install -m 644 "$HERE/deploy/rice-relay.service" /etc/systemd/system/rice-relay.service
systemctl daemon-reload
systemctl enable --now rice-relay
sleep 1
curl -s http://127.0.0.1:9000/api/ping && echo " ← 本机自测通过"

echo "==> 5/6 配 Nginx 反向代理"
cat > /etc/nginx/conf.d/rice-relay.conf <<EOF
server {
    listen 80;
    server_name $DOMAIN;
    client_max_body_size 30m;
    location /api/ {
        proxy_pass http://127.0.0.1:9000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 120s;
        proxy_request_buffering off;
    }
}
EOF
nginx -t && systemctl reload nginx

echo "==> 6/6 申请 HTTPS 证书（小程序正式版必须 https）"
echo "    如果这一步失败（域名没解析/IP 不对），先修好解析再跑："
echo "    certbot --nginx -d $DOMAIN"
certbot --nginx -d "$DOMAIN" -m "admin@$DOMAIN" --agree-tos --redirect -n || true

echo
echo "==============================================="
echo " 部署完成，跑一次自检："
echo "   python3 $APP/check_relay.py https://$DOMAIN $TOKEN"
echo " 电脑端：「自动巡检 → 云端同步」填 https://$DOMAIN 和同一口令"
echo "==============================================="
