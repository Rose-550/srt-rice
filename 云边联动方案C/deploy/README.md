# 中转服务部署手册（阿里云 ECS）

> 目标：让**不同网络**下的小程序把「图 + 识别结果」送到家里/办公室的电脑上。
> 架构：小程序 ──HTTPS──> 这台服务器 ──> 电脑端**主动拉取**（电脑不需要公网 IP）。

## 0. 前置条件

| 项 | 说明 |
|---|---|
| 阿里云 ECS | 最低配就够（1 核 1G），系统 Ubuntu 20.04+ / Alibaba Cloud Linux |
| 域名 + **ICP 备案** | 小程序只认**已备案**的 https 域名，没备案会被微信拒绝 |
| 安全组 | 入方向放行 **80、443**（9000 不要对外） |
| 证书 | 阿里云「数字证书管理服务」下载免费证书，或 certbot 签 Let's Encrypt |

## 1. 上传代码并安装（3 条命令）

```bash
# 在本地把整个方案目录传到服务器
scp -r 云边联动方案C root@你的服务器IP:/tmp/rice-deploy

# 登录服务器后执行一键安装
ssh root@你的服务器IP
bash /tmp/rice-deploy/deploy/install.sh
```

脚本会建用户 `rice`、把代码放到 `/opt/rice-relay`、装好 systemd 单元并启动。
**然后立刻改配置**（口令与域名）：

```bash
vi /opt/rice-relay/relay.env     # 改 RELAY_TOKEN 与 RELAY_PUBLIC_BASE
systemctl restart rice-relay
curl -s http://127.0.0.1:9000/api/ping    # 应返回 {"ok": true, ...}
```

## 2. 配 Nginx + HTTPS

```bash
# 域名先解析到这台 ECS 的公网 IP，然后：
yum install -y nginx || apt-get install -y nginx
cp /tmp/rice-deploy/deploy/nginx-rice-relay.conf /etc/nginx/conf.d/rice-relay.conf
vi /etc/nginx/conf.d/rice-relay.conf      # 把 your-domain.com 换成你的域名
# 证书放好（阿里云控制台下载 Nginx 版，或 certbot）
mkdir -p /etc/nginx/ssl && cp 你的证书.pem /etc/nginx/ssl/your-domain.com.pem
cp 你的私钥.key /etc/nginx/ssl/your-domain.com.key
nginx -t && systemctl reload nginx
```

## 3. 部署自检（**这一步别跳过**）

```bash
python3 /tmp/rice-deploy/check_relay.py https://your-domain.com 你的口令
```

它会真的走一遍「建记录 → 传图 → 补传识别结果 → 取图」，并检查：
- 健康检查通不通（域名/证书/反代/安全组）
- 三步接口是否都能用
- **取回的图片与上传字节完全一致**（证明不转码、EXIF/GPS 不丢）
- **识别结果 label/score 是否真存进去了**（这步最容易漏）

## 4. 小程序端

见上一层目录的 [README](../README.md#接入你已有的小程序三步)：拷 `miniprogram/utils/relay.js`，
在小程序后台把域名加进 **request 与 uploadFile 合法域名**，识别完成后调 `relay.sendToPc(...)`。

## 5. 电脑端

程序 →「自动巡检 → **云端同步**」→ 填 `https://your-domain.com` 与口令 → 点「立即同步」。
拉下来的图进收件箱，自动巡检照常检测、复核、提醒、落库，带 GPS 的自动落到地块地图。

## 6. 运维

```bash
systemctl status rice-relay        # 看状态
journalctl -u rice-relay -f        # 看日志
ls /opt/rice-relay/data/images     # 手机上传的原图
cat /opt/rice-relay/data/records.jsonl | wc -l   # 累计记录数
```

- **备份**：拷走 `/opt/rice-relay/data` 即可（含原图与索引）
- **磁盘**：图片按张累积，长期跑建议加个定时任务清理三个月前的图
- **安全**：`RELAY_TOKEN` 只给电脑端用；小程序端不要放任何 Key
