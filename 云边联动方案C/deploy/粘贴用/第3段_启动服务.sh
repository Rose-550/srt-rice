#!/usr/bin/env bash
# 【第 3 段】写配置与服务单元 → 启动 → 本机自检
set -e
APP=/opt/rice-relay
DOMAIN=rice-srt.cn
TOKEN=rice-srt-2026-1014
cat > $APP/relay.env <<EOF
RELAY_TOKEN=$TOKEN
RELAY_PUBLIC_BASE=http://$DOMAIN
EOF
chmod 600 $APP/relay.env
cat > /etc/systemd/system/rice-relay.service <<'__SVC__'
# 稻视界 · 中转服务 systemd 单元
# 安装：见 deploy/README.md（install.sh 会自动放好并启用）
[Unit]
Description=Rice Relay (mini-program -> PC bridge)
After=network.target

[Service]
Type=simple
User=rice
Group=rice
WorkingDirectory=/opt/rice-relay
# 口令等敏感配置放这里，权限 600，不进代码库
EnvironmentFile=/opt/rice-relay/relay.env
ExecStart=/usr/bin/python3 /opt/rice-relay/relay_server.py 9000
Restart=always
RestartSec=3
# 让服务只能写自己的数据目录
ReadWritePaths=/opt/rice-relay/data
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target

__SVC__
chown -R rice:rice $APP
systemctl daemon-reload
systemctl enable --now rice-relay
sleep 2
echo "=== 本机自检（应返回 ok:true）==="
curl -s http://127.0.0.1:9000/api/ping || echo "（没起来：journalctl -u rice-relay -n 30）"
echo
echo "=== 第 3 段完成。接下来装 Nginx + 证书： ==="
echo "apt-get update && apt-get install -y nginx certbot python3-certbot-nginx"
echo "printf '%s\n' 'server {' '    listen 80;' '    server_name rice-srt.cn;' '    client_max_body_size 30m;' '    location /api/ {' '        proxy_pass http://127.0.0.1:9000;' '        proxy_set_header Host \$host;' '        proxy_read_timeout 120s;' '        proxy_request_buffering off;' '    }' '}' > /etc/nginx/conf.d/rice-relay.conf"
echo "nginx -t && systemctl reload nginx"
echo "certbot --nginx -d rice-srt.cn -m admin@rice-srt.cn --agree-tos --redirect"
