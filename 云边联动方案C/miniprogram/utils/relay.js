// utils/relay.js —— 稻视界 · 把手机端结果送到电脑端
//
// 用法（在你小程序识别完成的地方加三行）：
//     const relay = require('../../utils/relay.js');
//     relay.setBase('https://你的中转域名');        // 或让用户在设置里填，存本地
//     relay.sendToPc({
//       filePath: tempFilePath,                     // 拍照/选图的临时路径
//       label: '稻瘟病', score: 0.88, boxes: 2,     // ← 你的识别结果（关键）
//       model: 'yolov8n-onnx',
//       lat: 32.0106, lng: 118.63,                  // 有就带，电脑端会自动落到地图上
//       note: '东片1号田',
//     }).then(r => console.log('已送到电脑端', r.id))
//       .catch(e => wx.showToast({ title: '发送失败：' + e.message, icon: 'none' }));
//
// 上线前必做两件事（否则 wx.request / wx.getLocation 会直接失败）：
//   1) 小程序后台 →「开发管理 → 服务器域名」→ request 与 uploadFile 合法域名里
//      加上你的中转域名（必须 https，不能是 IP，不能带端口以外的路径）
//   2) 「设置 → 服务内容声明 → 用户隐私保护指引」里声明「地理位置」用途
//
// 说明：识别在手机端做（或你的后端做），电脑端只是**接收与归档**；
//       三步接口 create → upload → finish，补传的 finish 才带识别结果。

const STORAGE_KEY = 'relay_base';
const DEFAULT_BASE = 'https://your-domain.com';   // TODO 换成你的中转域名

function getBase() {
  let b = '';
  try { b = wx.getStorageSync(STORAGE_KEY) || ''; } catch (e) { b = ''; }
  return (b || DEFAULT_BASE).replace(/\/+$/, '');
}

function setBase(url) {
  const b = String(url || '').trim().replace(/\/+$/, '');
  if (!/^https:\/\//.test(b)) {
    // 正式版必须是 https；开发阶段可在开发者工具里勾「不校验合法域名」临时用 http
    console.warn('[relay] 建议使用 https 域名：', b);
  }
  try { wx.setStorageSync(STORAGE_KEY, b); } catch (e) {}
  return b;
}

function request(url, method, data) {
  return new Promise(function (resolve, reject) {
    wx.request({
      url: url, method: method, data: data || {},
      header: { 'content-type': 'application/json' },
      success: function (res) {
        const body = res.data || {};
        if (res.statusCode === 200 && body.ok) { resolve(body); return; }
        reject(new Error((body && (body.error || body.msg)) || ('HTTP ' + res.statusCode)));
      },
      fail: function (err) {
        const m = (err && err.errMsg) || '';
        if (m.indexOf('domain') > -1 || m.indexOf('合法域名') > -1) {
          reject(new Error('域名未在小程序后台配置：请在「服务器域名」里加上 ' + url.split('/api/')[0]));
        } else {
          reject(new Error(m || '网络失败'));
        }
      },
    });
  });
}

// 1) 建记录（带上位置/设备/备注）
function createRecord(meta) {
  return request(getBase() + '/api/records', 'POST', meta || {});
}

// 2) 上传图片（原文件字节；服务端原样落盘，不转码）
function uploadImage(id, filePath, filename) {
  return new Promise(function (resolve, reject) {
    wx.uploadFile({
      url: getBase() + '/api/upload/' + id,
      filePath: filePath,
      name: 'file',
      header: { 'X-Filename': filename || 'photo.jpg' },
      success: function (up) {
        let body = {};
        try { body = JSON.parse(up.data); } catch (e) {}
        if (up.statusCode === 200 && body.ok) { resolve(body); return; }
        reject(new Error((body && body.error) || ('HTTP ' + up.statusCode)));
      },
      fail: function (err) { reject(new Error((err && err.errMsg) || '上传失败')); },
    });
  });
}

// 3) 补传元数据 + **识别结果**（这一步电脑端才拿得到你手机上的判断）
function finishRecord(id, patch) {
  return request(getBase() + '/api/finish/' + id, 'POST', patch || {});
}

/**
 * 一次性把「图 + 位置 + 识别结果」送到电脑端。
 * @returns {Promise<{id:string}>}
 */
function sendToPc(opt) {
  const o = opt || {};
  if (!o.filePath) return Promise.reject(new Error('缺少图片路径'));
  const meta = {
    device: o.device || ((wx.getDeviceInfo && wx.getDeviceInfo().model) || ''),
    lat: (o.lat === undefined ? null : o.lat),
    lng: (o.lng === undefined ? null : o.lng),
    accuracy: (o.accuracy === undefined ? null : o.accuracy),
    address: o.address || '',
    note: o.note || '',
  };
  let rid = '';
  return createRecord(meta)
    .then(function (r) { rid = r.id; return uploadImage(rid, o.filePath, o.filename); })
    .then(function () {
      const patch = { note: meta.note, device: meta.device };
      if (o.label) patch.label = o.label;                    // ← 识别结果
      if (o.score !== undefined) patch.score = o.score;
      if (o.boxes !== undefined) patch.boxes = o.boxes;
      if (o.model) patch.model = o.model;
      if (o.raw) patch.raw = o.raw;
      return finishRecord(rid, patch);
    })
    .then(function () { return { id: rid }; });
}

// 可选：拿到定位（gcj02 与高德/腾讯一致；别用 wgs84，否则地图上会偏）
function locate() {
  return new Promise(function (resolve, reject) {
    wx.getLocation({
      type: 'gcj02', isHighAccuracy: true,
      success: function (res) {
        resolve({ lat: res.latitude, lng: res.longitude, accuracy: res.accuracy });
      },
      fail: function (err) {
        const m = (err && err.errMsg) || '';
        reject(new Error(m.indexOf('privacy') > -1 || m.indexOf('authorize') > -1
          ? '未授权定位：请在隐私保护指引里声明「地理位置」，并引导用户授权'
          : (m || '定位失败')));
      },
    });
  });
}

module.exports = {
  getBase: getBase, setBase: setBase,
  createRecord: createRecord, uploadImage: uploadImage, finishRecord: finishRecord,
  sendToPc: sendToPc, locate: locate,
};
