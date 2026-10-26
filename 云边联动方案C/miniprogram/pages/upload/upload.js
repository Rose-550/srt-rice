// pages/upload/upload.js —— 稻视界 · 田间采样（上传给电脑端）
//
// 流程：隐私授权 → 定位 → 拍照 → 创建记录 → 上传图片 → 补传元数据
// 三步接口与中转服务端 /api/records、/api/upload/<id>、/api/finish/<id> 对应。
//
// 上线前必做：
//   1) 小程序后台「开发管理 → 服务器域名 → request/uploadFile 合法域名」里加上你的中转域名（必须 https）
//   2) 隐私保护指引里声明「地理位置」用途，否则 getLocation 会直接失败
//   3) 不要在这里放任何 API Key —— 识别在电脑端做

const DEFAULT_BASE = 'https://your-domain.com';   // TODO 换成你的中转服务域名

Page({
  data: {
    baseUrl: DEFAULT_BASE,
    loc: null,
    locText: '未定位',
    locating: false,
    photoPath: '',
    note: '',
    submitting: false,
    status: '',
    statusType: '',
    lastId: '',
  },

  onLoad() {
    const saved = wx.getStorageSync('relay_base');
    if (saved) this.setData({ baseUrl: saved });
  },

  onBaseInput(e) {
    this.setData({ baseUrl: e.detail.value.trim() });
  },

  onNoteInput(e) {
    this.setData({ note: e.detail.value });
  },

  setStatus(msg, isErr) {
    this.setData({ status: msg, statusType: isErr ? 'err' : '' });
  },

  // ---------- 隐私授权 + 定位 ----------
  getLocation() {
    this.setData({ locating: true, locText: '定位中…' });
    const doGet = () => wx.getLocation({
      type: 'gcj02',              // 高德/腾讯地图都用 gcj02，别用 wgs84
      isHighAccuracy: true,
      success: (res) => {
        this.setData({
          loc: { lat: res.latitude, lng: res.longitude, accuracy: res.accuracy },
          locText: res.latitude.toFixed(6) + ', ' + res.longitude.toFixed(6),
          locating: false,
        });
      },
      fail: (err) => {
        this.setData({ locating: false, locText: '定位失败' });
        const msg = (err && err.errMsg) || '';
        if (msg.indexOf('privacy') > -1 || msg.indexOf('authorize') > -1) {
          this.setStatus('定位权限未开启：请在小程序后台「用户隐私保护指引」中声明地理位置，并在手机上允许授权。', true);
        } else {
          this.setStatus('定位失败：' + msg, true);
        }
      },
    });

    // 基础库 2.32.3+ 需要先弹隐私授权；低版本没有这个接口，直接定位
    if (wx.getPrivacySetting) {
      wx.getPrivacySetting({
        success: (res) => {
          if (res.needAuthorization) {
            wx.requirePrivacyAuthorize({ success: doGet, fail: () => {
              this.setData({ locating: false, locText: '未授权' });
              this.setStatus('需要同意隐私授权才能获取位置', true);
            }});
          } else {
            doGet();
          }
        },
        fail: doGet,
      });
    } else {
      doGet();
    }
  },

  // ---------- 选照片 ----------
  choosePhoto() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['camera', 'album'],
      sizeType: ['compressed'],        // 压缩版足够识别，也省流量
      camera: 'back',
      success: (res) => {
        this.setData({ photoPath: res.tempFiles[0].tempFilePath, status: '', lastId: '' });
      },
      fail: () => {},
    });
  },

  // ---------- 三步上传 ----------
  onSubmit() {
    if (!this.data.photoPath) return;
    const base = (this.data.baseUrl || '').replace(/\/+$/, '');
    if (!/^https?:\/\//.test(base)) {
      this.setStatus('请先填写正确的中转服务地址（要带 http:// 或 https://）', true);
      return;
    }
    wx.setStorageSync('relay_base', base);

    if (!this.data.loc) {
      this.setStatus('提示：还没定位，将只上传照片不带坐标。', false);
    }
    this.setData({ submitting: true, status: '', lastId: '' });

    const meta = {
      device: (wx.getDeviceInfo ? wx.getDeviceInfo().model : '') || '',
      lat: this.data.loc ? this.data.loc.lat : null,
      lng: this.data.loc ? this.data.loc.lng : null,
      accuracy: this.data.loc ? this.data.loc.accuracy : null,
      note: this.data.note || '',
    };

    // 步骤 1：创建记录
    wx.request({
      url: base + '/api/records',
      method: 'POST',
      header: { 'content-type': 'application/json' },
      data: meta,
      success: (res) => {
        if (!res.data || !res.data.ok) {
          this.finish(false, '创建记录失败：' + JSON.stringify(res.data || res.statusCode));
          return;
        }
        const id = res.data.id;

        // 步骤 2：上传图片（wx.uploadFile 会把文件作为请求体，与 mock_upload.py 行为一致）
        wx.uploadFile({
          url: (res.data.upload_url || base + '/api/upload/' + id),
          filePath: this.data.photoPath,
          name: 'file',                                  // 服务端读的是原始请求体，此字段不影响
          header: { 'X-Filename': 'photo.jpg' },
          success: (up) => {
            let okBody = {};
            try { okBody = JSON.parse(up.data); } catch (e) {}
            if (up.statusCode !== 200 || !okBody.ok) {
              this.finish(false, '图片上传失败：HTTP ' + up.statusCode);
              return;
            }
            // 步骤 3：补传元数据（可选，失败不影响主流程）
            wx.request({
              url: res.data.finish_url || base + '/api/finish/' + id,
              method: 'POST',
              header: { 'content-type': 'application/json' },
              data: { note: meta.note, device: meta.device },
              complete: () => this.finish(true, '', id),
            });
          },
          fail: (err) => this.finish(false, '图片上传失败：' + ((err && err.errMsg) || '')),
        });
      },
      fail: (err) => this.finish(false, '连接中转服务失败：' + ((err && err.errMsg) || '') +
        '（检查域名是否已在后台配置、是否 https）'),
    });
  },

  finish(ok, msg, id) {
    this.setData({ submitting: false });
    if (ok) {
      this.setData({
        lastId: id, photoPath: '', note: '', statusType: '',
        status: '上传成功！家里的电脑会在 30 秒内取走这张照片，完成识别并记录到地图。',
      });
    } else {
      this.setData({ status: msg, statusType: 'err' });
    }
  },
});
