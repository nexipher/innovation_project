// 浏览器替身：只实现页面用到的那些接口，够跑 store 与决定逻辑。
const _store = new Map();
globalThis.localStorage = {
  getItem: (k) => (_store.has(k) ? _store.get(k) : null),
  setItem: (k, v) => _store.set(k, String(v)),
  removeItem: (k) => _store.delete(k),
};
// 每个选择器都返回一个可写的替身元素；#flash 的内容由断言检查。
function _element(id) {
  return { id: id, hidden: false, textContent: "", innerHTML: "", value: "",
           onclick: null, onchange: null, dataset: {},
           classList: { toggle: () => {}, add: () => {}, remove: () => {} },
           addEventListener: () => {}, click: () => {}, querySelector: () => null,
           files: [] };
}
// 同一选择器返回**同一个**替身：真实 DOM 是无状态查询，而这里要能读回写进去的东西
const _elements = new Map();
function _memo(sel) {
  if (!_elements.has(sel)) { _elements.set(sel, _element(sel)); }
  return _elements.get(sel);
}
globalThis.document = {
  addEventListener: () => {},
  querySelector: (sel) => _memo(sel),
  querySelectorAll: () => [],
  createElement: () => _element("a"),
};
globalThis.Blob = class { constructor(parts) { this.parts = parts; } };
globalThis.URL = { createObjectURL: () => "blob:x" };
globalThis.setTimeout = (fn) => 0;
globalThis.clearTimeout = () => {};
// 同步版 FileReader：本测试要的是校验规则，不是异步
globalThis.FileReader = class {
  readAsText(file) { this.result = file._text; this.onload(); }
};
let _failures = 0;
function assert(condition, message) {
  if (!condition) { _failures++; console.error("FAIL: " + message); }
  else { console.log("ok: " + message); }
}
function flashText() { return document.querySelector("#flash").textContent || ""; }
function finish() { process.exit(_failures ? 1 : 0); }
