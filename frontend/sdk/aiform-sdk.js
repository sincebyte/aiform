/**
 * Aiform — 原生 JS SDK：绑定 HTML 表单并调用后端预测接口。
 * 全局对象：window.Aiform
 */
(function (global) {
  "use strict";

  var cfg = {
    apiBase: "",
    database: "",
    table: "",
    userId: "",
    userIdColumn: "user_id",
    orderByColumn: "id",
    customPrompt: "",
    limit: 10,
    form: null,
    excludeFields: [],
    onProgress: null,
  };

  function resolveForm(el) {
    if (!el) return null;
    if (typeof el === "string") return document.querySelector(el);
    return el;
  }

  /** 控件自身或关联 label[for=id] 上的 data-prompt（控件优先）。 */
  function fieldPromptFromNode(node) {
    var p = node.getAttribute("data-prompt");
    if (p == null && node.dataset && node.dataset.prompt !== undefined) {
      p = node.dataset.prompt;
    }
    if (p != null && String(p).trim()) {
      return String(p).trim();
    }
    if (node.id && node.form) {
      var lab = node.form.querySelector('label[for="' + node.id + '"]');
      if (lab) {
        p = lab.getAttribute("data-prompt");
        if (p == null && lab.dataset && lab.dataset.prompt !== undefined) {
          p = lab.dataset.prompt;
        }
        if (p != null && String(p).trim()) {
          return String(p).trim();
        }
      }
    }
    return null;
  }

  function collectFields(form) {
    var meta = Object.create(null);
    var order = [];
    var exclude = {};
    if (cfg.userIdColumn) exclude[cfg.userIdColumn] = true;
    for (var e = 0; e < cfg.excludeFields.length; e++) {
      exclude[cfg.excludeFields[e]] = true;
    }
    var nodes = form.querySelectorAll("input, textarea, select");
    for (var i = 0; i < nodes.length; i++) {
      var node = nodes[i];
      var name = node.name;
      if (!name) continue;
      if (exclude[name]) continue;
      var type = (node.type || "").toLowerCase();
      if (type === "hidden" || type === "submit" || type === "button" || type === "file")
        continue;

      var prompt = fieldPromptFromNode(node);

      if (!meta[name]) {
        meta[name] = {
          name: name,
          tag: node.tagName.toLowerCase(),
          type: node.type || null,
          prompt: prompt || null,
        };
        order.push(name);
      } else if (prompt && (!meta[name].prompt || !String(meta[name].prompt).trim())) {
        meta[name].prompt = prompt;
      }
    }

    var fields = [];
    for (var j = 0; j < order.length; j++) {
      var m = meta[order[j]];
      var field = { name: m.name, tag: m.tag, type: m.type };
      if (m.prompt && String(m.prompt).trim()) field.prompt = String(m.prompt).trim();
      console.log("[Aiform] field:", field.name, "prompt:", field.prompt || "(none)");
      fields.push(field);
    }
    return fields;
  }

  function currentValues(form) {
    var out = {};
    var fields = collectFields(form);
    for (var i = 0; i < fields.length; i++) {
      var name = fields[i].name;
      var ctrl = form.elements.namedItem(name);
      if (!ctrl) continue;
      if (ctrl instanceof RadioNodeList) {
        var v = null;
        for (var j = 0; j < ctrl.length; j++) {
          if (ctrl[j].checked) {
            v = ctrl[j].value;
            break;
          }
        }
        out[name] = v;
      } else if (ctrl instanceof HTMLSelectElement && ctrl.multiple) {
        var sel = [];
        for (var k = 0; k < ctrl.options.length; k++) {
          if (ctrl.options[k].selected) sel.push(ctrl.options[k].value);
        }
        out[name] = sel.join(",");
      } else if ("value" in ctrl) {
        out[name] = ctrl.value;
      }
    }
    return out;
  }

  function applyFields(form, fieldsMap) {
    if (!fieldsMap || typeof fieldsMap !== "object") return;
    var names = Object.keys(fieldsMap);
    for (var i = 0; i < names.length; i++) {
      var name = names[i];
      var val = fieldsMap[name];
      if (val === null || val === undefined) continue;
      var ctrl = form.elements.namedItem(name);
      if (!ctrl) continue;
      var strVal = String(val);
      if (ctrl instanceof RadioNodeList) {
        for (var j = 0; j < ctrl.length; j++) {
          if (ctrl[j].value === strVal) {
            ctrl[j].checked = true;
            break;
          }
        }
      } else if (ctrl instanceof HTMLInputElement && ctrl.type === "checkbox") {
        ctrl.checked =
          strVal === "true" || strVal === "1" || strVal.toLowerCase() === "on";
      } else if ("value" in ctrl) {
        ctrl.value = strVal;
      }
    }
  }

  var Aiform = {
    /**
     * @param {object} opts
     * @param {string} [opts.apiBase] 后端根 URL，默认当前站点
     * @param {string} opts.database 库名
     * @param {string} opts.table 表名
     * @param {string|number} opts.userId 当前登录用户 id
     * @param {string} [opts.userIdColumn]
     * @param {string} [opts.orderByColumn]
     * @param {string} [opts.customPrompt]
     * @param {number} [opts.limit] 查询最近历史记录条数，默认 10，范围 1–100
     * @param {string[]|string} [opts.excludeFields] 不参与预测的字段名列表（逗号分隔字符串或数组）；userIdColumn 自动排除
     * @param {HTMLFormElement|string} opts.form 表单元素或选择器
     */
    init: function (opts) {
      opts = opts || {};
      cfg.apiBase = (opts.apiBase || "").replace(/\/$/, "");
      cfg.database = opts.database || "";
      cfg.table = opts.table || "";
      cfg.userId = opts.userId != null ? String(opts.userId) : "";
      cfg.userIdColumn = opts.userIdColumn || "user_id";
      cfg.orderByColumn = opts.orderByColumn || "id";
      cfg.customPrompt = opts.customPrompt || "";
      cfg.limit = opts.limit != null ? Number(opts.limit) : 10;
      cfg.form = resolveForm(opts.form);
      cfg.onProgress = opts.onProgress || null;
      var ex = opts.excludeFields || [];
      if (!Array.isArray(ex)) ex = String(ex).split(",").map(function (s) { return s.trim(); });
      cfg.excludeFields = ex;
      return this;
    },

    /** 与 init 相同语义的快捷方法 */
    configure: function (opts) {
      return this.init(opts);
    },

    /**
     * 收集表单并调用 /api/v1/predict，将返回的 JSON.fields 写入表单。
     * @returns {Promise<object>} 完整响应体 { fields, meta }
     */
    fill: async function () {
      if (!cfg.form) throw new Error("Aiform: 请先 init({ form: ... }) 绑定表单");
      var fields = collectFields(cfg.form);
      var body = {
        database: cfg.database,
        table: cfg.table,
        user_id: cfg.userId,
        user_id_column: cfg.userIdColumn,
        order_by_column: cfg.orderByColumn,
        limit: cfg.limit,
        custom_prompt: cfg.customPrompt,
        fields: fields,
        current_values: currentValues(cfg.form),
      };
      var url = cfg.apiBase + "/api/v1/predict";
      console.log("[Aiform] sending fields:", JSON.stringify(fields));
      console.log("[Aiform] full body:", JSON.stringify(body));
      var res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        var errText = await res.text();
        throw new Error("Aiform: HTTP " + res.status + " " + errText.slice(0, 200));
      }

      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      var result = null;

      while (true) {
        var r = await reader.read();
        if (r.done) break;
        buffer += decoder.decode(r.value, { stream: true });
        var lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (line.indexOf("data: ") !== 0) continue;
          var jsonStr = line.slice(6);
          var event;
          try {
            event = JSON.parse(jsonStr);
          } catch (e) {
            console.warn("[Aiform] SSE 解析失败:", jsonStr);
            continue;
          }
          if (event.type === "stage" && typeof cfg.onProgress === "function") {
            cfg.onProgress(event.stage, event.message);
          } else if (event.type === "result") {
            result = event.data;
          } else if (event.type === "error") {
            throw new Error(event.message || "未知错误");
          }
        }
      }

      if (!result) throw new Error("Aiform: 未收到预测结果");
      if (result.fields) applyFields(cfg.form, result.fields);
      return result;
    },

    /** fill 的别名 */
    autoFill: function () {
      return this.fill();
    },

    /** 仅根据接口返回对象更新表单（不发起请求） */
    applyResponse: function (data) {
      if (!cfg.form) throw new Error("Aiform: 未绑定表单");
      if (data && data.fields) applyFields(cfg.form, data.fields);
    },

    collectFields: function () {
      return cfg.form ? collectFields(cfg.form) : [];
    },

    getCurrentValues: function () {
      return cfg.form ? currentValues(cfg.form) : {};
    },
  };

  global.Aiform = Aiform;
})(typeof window !== "undefined" ? window : globalThis);
