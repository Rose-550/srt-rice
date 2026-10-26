# 稻视界 · 云边联动（方案 C）

地里的小程序拍照上传 → 中转服务 → 农户家里的电脑拉取、识别、存档、落点到地图。

**电脑不需要公网 IP，手机和电脑不需要在同一个网络。** 两端都是「往外连」，这是方案 C 的核心。

```
地里：手机小程序 ──► 中转服务（公网） ◄── 农户家：电脑
     拍照 + GPS                         定时拉取 → YOLO 识别 → 写点位表 → 出地图
```

## 一、已验证的事实

本地全链路已跑通（真实模型、真实图片、真实坐标）：

| 环节 | 实测结果 |
|---|---|
| 中转服务启动 | 正常，`/api/ping` 返回 `{ok:true}` |
| 模拟小程序上传 3 条 | 3/3 成功，含经纬度与地址 |
| 电脑端拉取 | 拉到 3 条新记录，重复拉取为 0 条（去重生效） |
| 本地推理 | ultralytics 加载 `best.pt`，识别出 稻水象甲 86%、白叶枯病 69%（检出 2 个） |
| 写点位表 | 坐标、类别、置信度、时间、备注全部写入 |
| **地图出图** | `gen_points.py` 输出 **`完成：共 3 个点位 → map.html`** |

也就是说：**你原有的地图链路一行代码没改，直接吃到了小程序上传的数据。**

## 二、目录说明

| 文件 | 作用 |
|---|---|
| `relay_server.py` | 中转服务端，**零依赖**（只用 Python 标准库） |
| `sync_pc.py` | 电脑端拉取 + 识别 + 写点位表；可命令行跑，也可嵌入 PyQt5 |
| `mock_upload.py` | 模拟小程序上传，用于没有真机时验证全链路 |
| `config.json` | 电脑端配置（中转地址、模型路径、点位表路径等） |
| `miniprogram/pages/upload/` | 小程序端页面：拍照 + 定位 + 上传 + 隐私授权 |

## 三、本地跑通（5 分钟）

打开三个终端，都在 `云边联动方案C` 目录下：

```bash
# 终端 1：启动中转服务
python relay_server.py 9000

# 终端 2：模拟小程序上传 3 条（没有真机也能验证）
python mock_upload.py --count 3

# 终端 3：电脑端拉取 + 识别 + 写点位表
python sync_pc.py --loop
```

看到 `[sync] 拉到 N 条` 和识别结果就成功了。然后生成地图：

```bash
cd ../AI+教育大赛/web地图demo
python gen_points.py      # 输出 map.html，双击打开
```

> 地图需要高德 Key：把申请到的 Key 放进 `web地图demo/key.txt`（你原来就有这个流程）。

## 四、接入你的小程序

1. 把 `miniprogram/pages/upload/` 整个目录拷进你的小程序项目。
2. 在 `app.json` 的 `pages` 数组里加一行：`"pages/upload/upload"`。
3. 改 `upload.js` 顶部的 `DEFAULT_BASE` 为你的中转服务地址（也可以在小程序界面里直接填）。
4. **上线前必做两件事**（见下方第六节）：配置合法域名、声明隐私指引。

## 五、部署到云服务器

你只需要一台最便宜的轻量服务器（学生机几十块一年）：

```bash
# 服务器上
mkdir -p /opt/rice-relay && cd /opt/rice-relay
# 把 relay_server.py 传上去
export RELAY_TOKEN='自己设一个口令'        # 电脑端拉取要带这个口令
nohup python3 relay_server.py 9000 > relay.log 2>&1 &
```

然后挂 Nginx + HTTPS（小程序正式版要求 https）：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    ssl_certificate     /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    client_max_body_size 25m;          # 要大于单张图片上限
    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

电脑端把 `config.json` 里的 `relay_base` 改成 `https://your-domain.com`，`token` 填成同一个口令。

## 六、上线前必做的两件事（不做会直接失败）

### 1. 配置合法域名

小程序后台 →「开发管理 → 开发设置 → 服务器域名」：

- `request 合法域名`：加 `https://your-domain.com`
- `uploadFile 合法域名`：加同一个

必须是 **https**、必须是**已备案**的域名。**这一步不做，正式版请求会直接被拦。**

### 2. 声明隐私指引（加定位功能必须做）

你的小程序**已经上线**，所以加定位不能直接发版：

1. 小程序后台 →「设置 → 服务内容声明 → 用户隐私保护指引」；
2. 勾选并声明**「地理位置」**用途（例如：用于记录病虫害发生位置并在田间地图上落点）；
3. 提交后等审核通过；
4. 再提交新版本代码。

**不做这一步，`wx.getLocation` 会直接报 `getLocation:fail appid隐私受限`。**

## 七、接口约定（要改需求时看这里）

| 方法 | 路径 | 调用方 | 说明 |
|---|---|---|---|
| GET | `/api/ping` | 运维 | 健康检查 |
| POST | `/api/records` | 小程序 | 创建记录，返回 `id`、`upload_url`、`finish_url` |
| POST | `/api/upload/<id>` | 小程序 | 上传图片二进制（请求体即文件） |
| POST | `/api/finish/<id>` | 小程序 | 补传备注等元数据 |
| GET | `/api/records?since=&limit=` | 电脑端 | 增量拉取，`since` 为上次最大时间戳 |
| GET | `/api/image/<id>` | 电脑端 | 下载图片 |

电脑端拉取接口支持 `X-Auth-Token` 头校验（服务端设了 `RELAY_TOKEN` 就生效）。

## 八、嵌入你现有的 PyQt5 应用

`sync_pc.py` 末尾有一段**可直接使用的嵌入示例**（`CloudSyncWorker`）：

- 跑在 `QThread` 里，不阻塞界面；
- 每处理一条就 `emit` 一次 `new_record` 信号，主窗口收到后刷新记录列表；
- 复用你现有的 `QThread` + `pyqtSignal` 模式，和 `InferenceThread` 一个风格。

## 九、设计上的几个取舍（答辩时可以用）

1. **拉取而非推送**：电脑定时来取，不需要公网 IP、不需要端口映射；家里断网数据也不丢，恢复后自动补齐（按时间戳增量 + 记录 ID 去重）。
2. **识别放在电脑端**：模型可以随时替换而不用重新发版小程序；也顺带避免了把 API Key 放进客户端。
3. **中转服务零依赖**：只用 Python 标准库，服务器上不用装任何包，部署最省事。
4. **复用既有地图链路**：`点位表.csv` 的表头结构没变，`gen_points.py` 和 `map.html` 完全不用改。
5. **为后续演进留了口子**：以后要加端侧推理（小程序 AI 推理能力）、多农户账号、专家远程指导，都在这套接口上加，不用推翻重来。

## 十、已知限制

- 中转服务用 `records.jsonl` 追加索引，**记录量到万级**建议换 SQLite/云数据库；
- 图片走中转会占服务器带宽和磁盘，长期建议图片存对象存储（OSS/COS），只传 URL；
- 没有账号体系，所有电脑端拉的是同一个池子，**多农户场景需要按农户隔离**；
- `sync_pc.py` 的 `--watch` 模式是最朴素的轮询，要求实时性更高可以换成 WebSocket 长连接。

---

## 接入你已有的小程序（三步）

1. **拷一个文件**：把 `miniprogram/utils/relay.js` 复制到你自己的小程序项目里。
2. **配域名**：小程序后台 →「开发管理 → 服务器域名」→ `request` 与 `uploadFile` 合法域名
   都加上你的中转域名（**必须 https**，不能填 IP；开发阶段可在开发者工具里勾「不校验合法域名」临时调试）。
   同时「设置 → 服务内容声明 → 用户隐私保护指引」里声明**地理位置**用途，否则 `getLocation` 会直接失败。
3. **识别完成后调一行**：

```js
const relay = require('../../utils/relay.js');
relay.setBase('https://你的中转域名');

// 你的识别结果（label/score 就是你小程序里已经算出来的值）
relay.locate().catch(() => null).then(loc => relay.sendToPc({
  filePath: tempFilePath,          // 拍照/选图的临时路径
  label: '稻瘟病', score: 0.88, boxes: 2, model: 'yolov8n-onnx',
  lat: loc && loc.lat, lng: loc && loc.lng,
  note: '东片1号田',
})).then(r => console.log('已送到电脑端', r.id))
   .catch(e => wx.showToast({ title: '发送失败：' + e.message, icon: 'none' }));
```

电脑端那边不用装任何东西：程序的「自动巡检 → 云端同步」填上同一个域名，点一下就把图和结论拉进收件箱，
自动巡检会照常检测、复核、提醒、落库，带 GPS 的自动落到地块地图上；
手机端的结论会作为**端云一致性**的对照写进台账。
