(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const elements = {
    conversation: $("#conversation"), emptyState: $("#empty-state"), emptyEyebrow: $("#empty-eyebrow"),
    messageList: $("#message-list"), template: $("#message-template"), tabsList: $("#tabs-list"),
    addTab: $("#add-tab-button"), statusChip: $("#status-chip"), statusText: $("#status-text"),
    connectButton: $("#connect-button"), connectLabel: $("#connect-button .connect-label"), connectArrow: $("#connect-button .connect-arrow"),
    clearButton: $("#clear-button"), settingsButton: $("#settings-button"), settingsModal: $("#settings-modal"),
    closeSettings: $("#close-settings"), settingsForm: $("#settings-form"), backendList: $("#backend-radio-list"),
    auth: $("#auth-select"), apiKeyField: $("#api-key-field"), apiKey: $("#api-key-input"), refreshModels: $("#refresh-models-button"),
    voiceField: $("#voice-field"), voice: $("#voice-select"), modelList: $("#model-radio-list"), catalogStatus: $("#catalog-status"),
    thinking: $("#thinking-select"), screenshot: $("#screenshot-toggle"), imageModel: $("#image-model-select"),
    imageResolution: $("#image-resolution-select"), systemPrompt: $("#system-prompt-input"), catalogWarnings: $("#catalog-warnings"),
    accountUsage: $("#account-usage"), usageLimits: $("#usage-limits"), usageTokens: $("#usage-tokens"),
    modelModal: $("#model-modal"), closeModelPicker: $("#close-model-picker"), modelPickerForm: $("#model-picker-form"),
    newBackendList: $("#new-backend-radio-list"), newModelList: $("#new-model-radio-list"), newVoiceField: $("#new-voice-field"), newVoice: $("#new-voice-select"),
    composerForm: $("#composer-form"), composerInput: $("#composer-input"), sendButton: $("#send-button"),
    pending: $("#pending-attachments"), fileInput: $("#file-input"), uploadButton: $("#upload-button"), captureButton: $("#capture-button"),
    micButton: $("#mic-button"), micLevel: $("#mic-level"), modelPill: $("#model-pill"),
    errorBanner: $("#error-banner"), errorText: $("#error-text"), dismissError: $("#dismiss-error"),
    workspace: $("#workspace"), bottomWorkspace: $("#bottom-workspace"), dockResize: $("#dock-resize-handle"),
    noteSidebar: $("#note-sidebar"), sidebarResize: $("#sidebar-resize-handle"), chatWorkspace: $("#chat-workspace"),
    chatWorkspaceButton: $("#chat-workspace-button"), noteEditorWorkspace: $("#note-editor-workspace"),
    noteList: $("#note-list"), newNoteButton: $("#new-note-button"), openNoteButton: $("#open-note-button"),
    noteFileInput: $("#note-file-input"), noteEditor: $("#note-editor"), activeNoteName: $("#active-note-name"), noteSaveStatus: $("#note-save-status"),
  };

  const backendOptions = [
    { id: "codex_gpt_live", label: "GPT Live API", description: "Codex-managed gpt-live-* WebRTC voice session" },
    { id: "open_ai_realtime", label: "Realtime API", description: "OpenAI Realtime WebSocket voice session" },
    { id: "codex_text", label: "Text model", description: "Codex app-server text and tool session" },
  ];

  let appState = null;
  let socket = null;
  let reconnectTimer = null;
  let renderedMessageTabId = null;
  let lastTabsSignature = "";
  let lastPendingSignature = "";
  let lastSettingsSignature = "";
  let dismissedError = null;
  let localError = null;
  let settingsBackend = "codex_gpt_live";
  let newTabBackend = "codex_gpt_live";
  let micStream = null;
  let micContext = null;
  let micSource = null;
  let micProcessor = null;
  let micActive = false;
  let micStarting = false;
  let micOptOutTabId = null;
  let micFailedTabId = null;
  let autoConnectAttemptedTabId = null;
  let manualDisconnectTabId = null;
  let outputContext = null;
  let outputCursor = 0;
  let noteFiles = [];
  let activeNote = null;
  let noteSaveTimer = null;
  let noteDirty = false;
  let workspaceMode = "chat";

  function activeTab() {
    return appState?.tabs?.find((tab) => tab.id === appState.active_tab_id) || appState?.tabs?.[0] || null;
  }

  async function request(path, payload = undefined) {
    const options = payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
    const response = await fetch(path, options);
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) throw new Error(body?.error || `Request failed (${response.status})`);
    return body;
  }

  function connectSocket() {
    clearTimeout(reconnectTimer);
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${location.host}/ws`);
    socket.binaryType = "arraybuffer";
    socket.addEventListener("message", (event) => {
      if (typeof event.data === "string") {
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === "state") applyState(payload.state);
        } catch (error) { console.warn("Malformed state update", error); }
      } else if (event.data instanceof ArrayBuffer) playPcm16(event.data);
    });
    socket.addEventListener("close", () => { stopMicrophone(); reconnectTimer = setTimeout(connectSocket, 900); });
    socket.addEventListener("error", () => socket.close());
  }

  function applyState(state) {
    if (!state?.tabs) return;
    const previousTab = activeTab();
    const previousTabId = previousTab?.id ?? null;
    const previousConnection = previousTab?.connection ?? null;
    appState = state;
    localError = null;
    const tab = activeTab();
    if (previousTabId !== tab?.id) {
      renderedMessageTabId = null;
      lastPendingSignature = "";
      micOptOutTabId = null;
      micFailedTabId = null;
      autoConnectAttemptedTabId = null;
      manualDisconnectTabId = null;
    }
    renderTabs();
    renderHeader();
    renderMessages();
    renderPending();
    renderError();
    if (!elements.settingsModal.hidden) {
      const signature = settingsSignature(tab);
      if (signature !== lastSettingsSignature && !elements.settingsModal.contains(document.activeElement)) populateSettings(false);
    }
    if (!elements.modelModal.hidden) renderNewTabPicker();
    if (tab?.connection === "live" && (previousConnection !== "live" || previousTabId !== tab.id || !micActive)) ensureDefaultMicrophone();
    else if (tab?.connection === "offline" && !tab.error && manualDisconnectTabId !== tab.id) ensureAutoConnection();
  }

  function settingsSignature(tab) {
    if (!tab) return "";
    return JSON.stringify([tab.id, tab.settings, appState?.catalog?.refreshed_at, appState?.catalog?.loading]);
  }

  function renderTabs() {
    const tabs = appState.tabs || [];
    const signature = JSON.stringify(tabs.map((tab) => [tab.id, tab.title, tab.connection, tab.id === appState.active_tab_id]));
    if (signature === lastTabsSignature) return;
    lastTabsSignature = signature;
    elements.tabsList.replaceChildren(...tabs.map((tab) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "session-tab";
      button.dataset.active = String(tab.id === appState.active_tab_id);
      button.dataset.state = tab.connection;
      const dot = document.createElement("span"); dot.className = "tab-dot";
      const label = document.createElement("span"); label.className = "tab-label"; label.textContent = tab.title;
      button.append(dot, label);
      if (tabs.length > 1) {
        const close = document.createElement("span"); close.className = "tab-close"; close.textContent = "×"; close.title = "Close tab";
        close.addEventListener("click", (event) => { event.stopPropagation(); request("/api/tabs/close", { tab_id: tab.id }).catch(showLocalError); });
        button.append(close);
      }
      button.addEventListener("click", () => {
        if (tab.id === appState.active_tab_id) return;
        stopMicrophone();
        micOptOutTabId = null;
        micFailedTabId = null;
        request("/api/tabs/switch", { tab_id: tab.id }).catch(showLocalError);
      });
      return button;
    }));
  }

  function renderHeader() {
    const tab = activeTab();
    if (!tab) return;
    elements.statusChip.dataset.state = tab.connection;
    elements.statusText.textContent = tab.status || "Ready";
    const isLive = tab.connection === "live";
    const isBusy = ["connecting", "reconnecting"].includes(tab.connection);
    elements.connectButton.dataset.live = String(isLive);
    elements.connectButton.disabled = isBusy;
    elements.connectLabel.textContent = isLive ? "Disconnect" : isBusy ? "Connecting" : "Connect";
    elements.connectArrow.textContent = isLive ? "×" : isBusy ? "···" : "↗";
    elements.modelPill.textContent = tab.title;
    elements.emptyEyebrow.textContent = tab.title;
    elements.micButton.disabled = !isLive || tab.settings.backend === "codex_text";
    elements.captureButton.disabled = false;
    if (tab.connection === "offline" && micActive) stopMicrophone();
  }

  function renderMessages() {
    const tab = activeTab();
    if (!tab) return;
    const messages = (tab.messages || []).filter((message) => message.role !== "system");
    elements.emptyState.hidden = messages.length > 0;
    elements.messageList.hidden = messages.length === 0;
    const distance = elements.conversation.scrollHeight - elements.conversation.scrollTop - elements.conversation.clientHeight;
    const stick = distance < 100 || messages.some((message) => message.streaming);
    if (renderedMessageTabId !== tab.id) {
      elements.messageList.replaceChildren();
      renderedMessageTabId = tab.id;
    }
    const existing = new Map([...elements.messageList.children].map((node) => [Number(node.dataset.messageId), node]));
    const retained = new Set();
    messages.forEach((message, index) => {
      let node = existing.get(message.id);
      if (!node) {
        node = elements.template.content.firstElementChild.cloneNode(true);
        node.dataset.messageId = String(message.id);
      }
      retained.add(message.id);
      updateMessageNode(node, message);
      const nodeAtIndex = elements.messageList.children[index] || null;
      if (nodeAtIndex !== node) elements.messageList.insertBefore(node, nodeAtIndex);
    });
    for (const [id, node] of existing) if (!retained.has(id)) node.remove();
    if (stick) requestAnimationFrame(() => { elements.conversation.scrollTop = elements.conversation.scrollHeight; });
  }

  function updateMessageNode(node, message) {
    node.dataset.role = message.role;
    node.dataset.streaming = String(Boolean(message.streaming));
    const role = message.role === "user" ? "You" : "Assistant";
    const roleNode = node.querySelector(".message-role");
    if (roleNode.textContent !== role) roleNode.textContent = role;
    const metrics = formatMetrics(message);
    const timing = node.querySelector(".message-timing");
    if (timing.textContent !== metrics) timing.textContent = metrics;
    const text = message.text || (message.streaming ? "" : "No text response");
    const textNode = node.querySelector(".message-text");
    if (textNode.textContent !== text) textNode.textContent = text;

    const attachments = node.querySelector(".message-attachments");
    const attachmentSignature = JSON.stringify((message.attachments || []).map((item) => [item.id, item.status, item.byte_size, item.seconds, item.data_url?.length]));
    if (attachments.dataset.signature !== attachmentSignature) {
      attachments.dataset.signature = attachmentSignature;
      attachments.replaceChildren(...(message.attachments || []).map(renderAttachment));
    }
    attachments.hidden = !message.attachments?.length;

    const tools = node.querySelector(".message-tools");
    const toolSignature = JSON.stringify((message.tool_calls || []).map((tool) => [tool.call_id, tool.status, tool.output, tool.arguments]));
    if (tools.dataset.signature !== toolSignature) {
      tools.dataset.signature = toolSignature;
      tools.replaceChildren(...(message.tool_calls || []).map(renderTool));
    }
    tools.hidden = !message.tool_calls?.length;

    const image = node.querySelector(".message-image");
    if (message.generated_image_url) {
      if (image.src !== message.generated_image_url) image.src = message.generated_image_url;
      image.hidden = false;
    } else {
      image.removeAttribute("src");
      image.hidden = true;
    }
  }

  function formatMetrics(message) {
    const start = formatClock(message.started_at);
    const end = message.finished_at ? formatClock(message.finished_at) : "…";
    const parts = [`${start}–${end}`];
    if (Number.isFinite(message.elapsed_ms)) parts.push(formatDuration(message.elapsed_ms));
    if (Number.isFinite(message.token_count)) parts.push(`${message.token_count_is_estimate ? "≈" : ""}${message.token_count.toLocaleString()} tok`);
    if (Number.isFinite(message.tokens_per_second)) parts.push(`${message.tokens_per_second.toFixed(1)} tok/s`);
    return parts.join(" · ");
  }

  function formatClock(milliseconds) {
    if (!Number.isFinite(milliseconds)) return "";
    return new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(milliseconds));
  }

  function formatDuration(milliseconds) {
    if (milliseconds < 1000) return `${milliseconds}ms`;
    if (milliseconds < 60000) return `${(milliseconds / 1000).toFixed(milliseconds < 10000 ? 1 : 0)}s`;
    const minutes = Math.floor(milliseconds / 60000);
    const seconds = Math.floor((milliseconds % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  }

  function renderAttachment(attachment) {
    const card = document.createElement("figure");
    card.className = "attachment-card";
    card.dataset.status = attachment.status;
    if (attachment.kind === "image" && attachment.data_url) {
      const image = document.createElement("img");
      image.src = attachment.data_url;
      image.alt = attachment.name;
      image.loading = "lazy";
      image.title = "Open exact image payload";
      image.addEventListener("click", () => window.open(attachment.data_url, "_blank", "noopener"));
      card.appendChild(image);
    } else if (attachment.kind === "audio" && attachment.data_url) {
      const audioWrap = document.createElement("div");
      audioWrap.className = "attachment-audio";
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = attachment.data_url;
      const download = document.createElement("a");
      download.href = attachment.data_url;
      download.download = attachment.name || "audio.wav";
      download.textContent = "Save WAV";
      audioWrap.append(audio, download);
      card.appendChild(audioWrap);
    }
    const caption = document.createElement("figcaption");
    const size = attachment.byte_size ? ` · ${formatBytes(attachment.byte_size)}` : "";
    const dimensions = attachment.width ? ` · ${attachment.width}×${attachment.height}` : "";
    caption.textContent = `${attachment.included_screen ? "Screen · " : ""}${attachment.name}${dimensions}${size} · ${attachment.status}`;
    card.appendChild(caption);
    return card;
  }

  function renderTool(tool) {
    const details = document.createElement("details"); details.className = "tool-call"; details.dataset.status = tool.status;
    const summary = document.createElement("summary"); summary.textContent = `${tool.name} · ${tool.status === "done" ? "Complete" : "Running"}`;
    const pre = document.createElement("pre"); pre.textContent = [tool.arguments, tool.output ? `Result\n${tool.output}` : ""].filter(Boolean).join("\n\n");
    details.append(summary, pre); return details;
  }

  function renderPending() {
    const tab = activeTab();
    const pending = tab?.pending_attachments || [];
    const signature = JSON.stringify([tab?.id, ...pending.map((attachment) => [attachment.id, attachment.status, attachment.data_url?.length])]);
    elements.pending.hidden = pending.length === 0;
    if (signature === lastPendingSignature) return;
    lastPendingSignature = signature;
    elements.pending.replaceChildren(...pending.map((attachment) => {
      const item = document.createElement("div"); item.className = "pending-item";
      if (attachment.kind === "image" && attachment.data_url) {
        const image = document.createElement("img"); image.src = attachment.data_url; image.alt = ""; item.appendChild(image);
      } else if (attachment.kind === "audio") {
        const icon = document.createElement("i"); icon.className = "pending-audio-icon"; icon.textContent = "♫"; item.appendChild(icon);
      }
      const label = document.createElement("span"); label.textContent = attachment.included_screen ? "Current screen" : attachment.name;
      const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "×"; remove.ariaLabel = "Remove attachment";
      remove.addEventListener("click", () => request("/api/attachments/remove", { tab_id: tab.id, attachment_id: attachment.id }).catch(showLocalError));
      item.append(label, remove); return item;
    }));
  }

  function renderError() {
    const error = localError || activeTab()?.error || null;
    const show = Boolean(error && error !== dismissedError);
    elements.errorBanner.hidden = !show;
    elements.errorText.textContent = show ? error : "";
  }

  function renderBackendRadios(root, name, selected, onChange) {
    root.replaceChildren(...backendOptions.map((option) => {
      const label = document.createElement("label"); label.className = "backend-radio";
      const input = document.createElement("input"); input.type = "radio"; input.name = name; input.value = option.id; input.checked = option.id === selected;
      input.addEventListener("change", () => onChange(option.id));
      const copy = document.createElement("span"); copy.innerHTML = `<b>${escapeHtml(option.label)}</b><small>${escapeHtml(option.description)}</small>`;
      label.append(input, copy); return label;
    }));
  }

  function modelsForBackend(backend) {
    const catalog = appState?.catalog || {};
    if (backend === "codex_gpt_live") return catalog.gpt_live_models || [];
    if (backend === "open_ai_realtime") return catalog.realtime_models || [];
    return catalog.text_models || [];
  }

  function voicesForBackend(backend) {
    const catalog = appState?.catalog || {};
    if (backend === "codex_gpt_live") return catalog.gpt_live_voices || [];
    if (backend === "open_ai_realtime") return catalog.realtime_voices || [];
    return [];
  }

  function renderModelRadios(root, name, backend, selected) {
    const models = modelsForBackend(backend);
    root.replaceChildren(...models.map((model, index) => {
      const label = document.createElement("label"); label.className = "model-radio";
      const input = document.createElement("input"); input.type = "radio"; input.name = name; input.value = model.id; input.checked = model.id === selected || (!selected && index === 0);
      const copy = document.createElement("span");
      const badge = model.source === "openai_api" ? "API" : model.source === "codex_account" ? "Account" : "Official";
      copy.innerHTML = `<b>${escapeHtml(model.label)}</b><small>${escapeHtml(model.description || model.id)}</small><em>${badge}</em>`;
      label.append(input, copy); return label;
    }));
  }

  function populateSelect(select, values, selected) {
    select.replaceChildren(...values.map((value) => {
      const option = document.createElement("option");
      const id = typeof value === "string" ? value : value.id;
      const label = typeof value === "string" ? value : value.label;
      option.value = id; option.textContent = label; option.selected = id === selected; return option;
    }));
    if (selected && !values.some((value) => (typeof value === "string" ? value : value.id) === selected)) {
      const option = document.createElement("option"); option.value = selected; option.textContent = selected; option.selected = true; select.prepend(option);
    }
  }

  function populateSettings(resetBackend = true) {
    const tab = activeTab(); if (!tab) return;
    const settings = tab.settings;
    const selectedModel = resetBackend ? settings.model : (checkedValue("settings-model") || settings.model);
    const selectedVoice = resetBackend ? settings.voice : (elements.voice.value || settings.voice);
    if (resetBackend) settingsBackend = settings.backend;
    renderBackendRadios(elements.backendList, "settings-backend", settingsBackend, (backend) => {
      settingsBackend = backend;
      elements.auth.value = "codex";
      renderSettingsChoices(null, null);
    });
    if (resetBackend) {
      elements.auth.value = settings.auth_mode;
      elements.thinking.value = settings.thinking_level;
      elements.screenshot.checked = Boolean(settings.send_screenshot);
      elements.imageResolution.value = settings.image_resolution;
      elements.systemPrompt.value = settings.system_prompt;
    }
    elements.apiKeyField.dataset.visible = String(elements.auth.value === "api_key");
    renderSettingsChoices(selectedModel, selectedVoice);
    populateSelect(elements.imageModel, appState.catalog.image_models || [], resetBackend ? settings.image_model : (elements.imageModel.value || settings.image_model));
    elements.catalogStatus.textContent = appState.catalog.loading ? "Loading…" : appState.catalog.refreshed_at ? `Updated ${formatClock(appState.catalog.refreshed_at)}` : "";
    const warnings = appState.catalog.warnings || [];
    elements.catalogWarnings.hidden = warnings.length === 0;
    elements.catalogWarnings.textContent = warnings.join("\n");
    renderAccountUsage();
    lastSettingsSignature = settingsSignature(tab);
  }

  function renderAccountUsage() {
    const usage = appState?.catalog?.account_usage;
    elements.accountUsage.hidden = !usage;
    if (!usage) {
      elements.usageLimits.replaceChildren();
      elements.usageTokens.replaceChildren();
      return;
    }
    elements.usageLimits.replaceChildren(...(usage.rate_limits || []).map((limit) => {
      const card = document.createElement("article"); card.className = "usage-limit-card";
      const title = document.createElement("div");
      const label = document.createElement("strong"); label.textContent = limit.name;
      const plan = document.createElement("span"); plan.textContent = limit.plan || "";
      title.append(label, plan); card.appendChild(title);
      for (const [windowName, window] of [["Primary", limit.primary], ["Secondary", limit.secondary]]) {
        if (!window) continue;
        const row = document.createElement("div"); row.className = "usage-window";
        const copy = document.createElement("span");
        copy.textContent = `${windowName} · ${Math.max(0, 100 - window.used_percent)}% left${window.window_duration_minutes ? ` · ${window.window_duration_minutes}m` : ""}`;
        const meter = document.createElement("i"); meter.style.setProperty("--used", `${Math.min(100, Math.max(0, window.used_percent))}%`);
        row.append(copy, meter);
        if (window.resets_at) { const reset = document.createElement("small"); reset.textContent = `Resets ${new Date(window.resets_at * 1000).toLocaleString()}`; row.append(reset); }
        card.appendChild(row);
      }
      const credits = document.createElement("p");
      credits.textContent = limit.unlimited_credits ? "Credits · unlimited" : limit.credit_balance ? `Credits · ${limit.credit_balance}` : limit.reached_reason ? limit.reached_reason : "";
      if (credits.textContent) card.appendChild(credits);
      return card;
    }));
    const tokenUsage = usage.token_usage || {};
    const stats = [
      ["Lifetime", tokenUsage.lifetime_tokens],
      [tokenUsage.latest_day || "Latest day", tokenUsage.latest_day_tokens],
      ["Recent reported", tokenUsage.recent_reported_tokens],
      ["Peak day", tokenUsage.peak_daily_tokens],
      ["Reset credits", usage.reset_credits],
    ].filter(([, value]) => Number.isFinite(value));
    elements.usageTokens.replaceChildren(...stats.map(([label, value]) => {
      const item = document.createElement("div");
      const strong = document.createElement("strong"); strong.textContent = Number(value).toLocaleString();
      const span = document.createElement("span"); span.textContent = label;
      item.append(strong, span); return item;
    }));
  }

  function renderSettingsChoices(selectedModel, selectedVoice) {
    renderModelRadios(elements.modelList, "settings-model", settingsBackend, selectedModel);
    const voices = voicesForBackend(settingsBackend);
    elements.voiceField.hidden = settingsBackend === "codex_text";
    populateSelect(elements.voice, voices, selectedVoice);
    elements.apiKeyField.dataset.visible = String(elements.auth.value === "api_key");
  }

  function settingsPayload() {
    const current = activeTab().settings;
    return {
      backend: settingsBackend,
      auth_mode: elements.auth.value,
      model: checkedValue("settings-model") || current.model,
      voice: settingsBackend === "codex_text" ? "" : elements.voice.value,
      thinking_level: elements.thinking.value,
      system_prompt: elements.systemPrompt.value,
      send_screenshot: elements.screenshot.checked,
      image_model: elements.imageModel.value,
      image_resolution: elements.imageResolution.value,
    };
  }

  function openSettings() { lastSettingsSignature = ""; populateSettings(); elements.settingsModal.hidden = false; document.body.style.overflow = "hidden"; }
  function closeSettings() { elements.settingsModal.hidden = true; document.body.style.overflow = ""; }

  function renderNewTabPicker(resetModel = false) {
    const selectedModel = resetModel ? null : checkedValue("new-model");
    const selectedVoice = resetModel ? null : elements.newVoice.value;
    renderBackendRadios(elements.newBackendList, "new-backend", newTabBackend, (backend) => {
      newTabBackend = backend;
      renderNewTabPicker(true);
    });
    renderModelRadios(elements.newModelList, "new-model", newTabBackend, selectedModel);
    const voices = voicesForBackend(newTabBackend);
    elements.newVoiceField.hidden = newTabBackend === "codex_text";
    populateSelect(elements.newVoice, voices, selectedVoice || voices[0]);
  }

  function openModelPicker() { renderNewTabPicker(); elements.modelModal.hidden = false; document.body.style.overflow = "hidden"; }
  function closeModelPicker() { elements.modelModal.hidden = true; document.body.style.overflow = ""; }

  function checkedValue(name) { return document.querySelector(`input[name="${name}"]:checked`)?.value || null; }

  async function toggleConnection() {
    const tab = activeTab(); if (!tab) return;
    dismissedError = null;
    if (tab.connection === "live") {
      manualDisconnectTabId = tab.id;
      micOptOutTabId = tab.id;
      stopMicrophone();
      await request("/api/disconnect", { tab_id: tab.id });
      return;
    }
    if (tab.settings.auth_mode === "api_key" && !elements.apiKey.value.trim() && !appState.has_credentials) {
      openSettings(); elements.apiKey.focus(); return;
    }
    manualDisconnectTabId = null;
    micOptOutTabId = null;
    micFailedTabId = null;
    autoConnectAttemptedTabId = tab.id;
    await request("/api/connect", { tab_id: tab.id, api_key: elements.apiKey.value.trim() || null });
    elements.apiKey.value = "";
  }

  async function sendComposer() {
    const text = elements.composerInput.value.trim(); if (!text) return;
    const tab = activeTab(); if (!tab) return;
    elements.composerInput.value = ""; resizeComposer(); updateSendButton();
    try {
      await request("/api/message", { tab_id: tab.id, text });
    } catch (error) {
      elements.composerInput.value = text;
      resizeComposer(); updateSendButton();
      throw error;
    }
  }

  async function uploadFiles(files) {
    const tab = activeTab(); if (!tab) return;
    for (const file of files) {
      if (!file.type.startsWith("image/") && !file.type.startsWith("audio/")) continue;
      const data_url = await readDataUrl(file);
      await request("/api/upload", { tab_id: tab.id, name: file.name, data_url });
    }
    elements.fileInput.value = "";
  }

  function readDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(reader.error); reader.readAsDataURL(file);
    });
  }

  function resizeComposer() { const input = elements.composerInput; input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 170)}px`; }
  function updateSendButton() { elements.sendButton.disabled = elements.composerInput.value.trim().length === 0; }

  async function ensureDefaultMicrophone() {
    const tab = activeTab();
    if (!tab || tab.connection !== "live" || tab.settings.backend === "codex_text") return;
    if (micOptOutTabId === tab.id || micFailedTabId === tab.id || micActive || micStarting) return;
    try { await startMicrophone(); } catch (error) {
      micFailedTabId = tab.id;
      showLocalError(error);
    }
  }

  async function startMicrophone() {
    const tab = activeTab();
    if (micActive || micStarting || tab?.connection !== "live" || tab.settings.backend === "codex_text") return;
    micStarting = true;
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      micContext = new AudioContext({ latencyHint: "interactive" });
      micSource = micContext.createMediaStreamSource(micStream);
      micProcessor = micContext.createScriptProcessor(2048, 1, 1);
      micProcessor.onaudioprocess = (event) => {
        if (!micActive || socket?.readyState !== WebSocket.OPEN) return;
        const samples = event.inputBuffer.getChannelData(0);
        socket.send(floatToPcm16(downsample(samples, micContext.sampleRate, 24000)).buffer);
        updateMicMeter(samples);
      };
      micSource.connect(micProcessor); micProcessor.connect(micContext.destination); micActive = true;
      micFailedTabId = null;
      elements.micButton.dataset.active = "true";
      elements.micButton.title = "Stop microphone";
      elements.micButton.setAttribute("aria-label", "Stop microphone");
    } finally {
      micStarting = false;
    }
  }

  function stopMicrophone() {
    micActive = false;
    micStarting = false;
    elements.micButton.dataset.active = "false";
    elements.micButton.title = "Start microphone";
    elements.micButton.setAttribute("aria-label", "Start microphone");
    elements.micLevel.style.transform = "scaleX(0)";
    if (micProcessor) { micProcessor.disconnect(); micProcessor.onaudioprocess = null; micProcessor = null; }
    if (micSource) { micSource.disconnect(); micSource = null; }
    if (micStream) { micStream.getTracks().forEach((track) => track.stop()); micStream = null; }
    if (micContext) { micContext.close().catch(() => {}); micContext = null; }
  }

  function downsample(input, sourceRate, targetRate) {
    if (targetRate >= sourceRate) return input;
    const ratio = sourceRate / targetRate; const output = new Float32Array(Math.max(1, Math.round(input.length / ratio))); let offset = 0;
    for (let index = 0; index < output.length; index += 1) { const next = Math.min(input.length, Math.round((index + 1) * ratio)); let sum = 0; let count = 0; for (; offset < next; offset += 1) { sum += input[offset]; count += 1; } output[index] = count ? sum / count : 0; }
    return output;
  }

  function floatToPcm16(input) { const output = new Int16Array(input.length); for (let i = 0; i < input.length; i += 1) { const sample = Math.max(-1, Math.min(1, input[i])); output[i] = sample < 0 ? sample * 32768 : sample * 32767; } return output; }
  function updateMicMeter(samples) { let power = 0; for (const sample of samples) power += sample * sample; elements.micLevel.style.transform = `scaleX(${Math.min(1, Math.sqrt(power / samples.length) * 9)})`; }
  async function ensureOutputContext() { if (!outputContext) outputContext = new AudioContext({ latencyHint: "interactive" }); if (outputContext.state === "suspended") await outputContext.resume(); outputCursor = Math.max(outputCursor, outputContext.currentTime); }
  async function playPcm16(buffer) { try { await ensureOutputContext(); const input = new Int16Array(buffer); if (!input.length) return; const audio = outputContext.createBuffer(1, input.length, 24000); const channel = audio.getChannelData(0); for (let i = 0; i < input.length; i += 1) channel[i] = input[i] / 32768; const source = outputContext.createBufferSource(); source.buffer = audio; source.connect(outputContext.destination); const start = Math.max(outputCursor, outputContext.currentTime + .015); source.start(start); outputCursor = start + audio.duration; } catch (error) { console.warn(error); } }

  function setWorkspaceMode(mode) {
    workspaceMode = mode === "note" ? "note" : "chat";
    document.body.dataset.workspaceMode = workspaceMode;
    elements.chatWorkspace.hidden = workspaceMode !== "chat";
    elements.noteEditorWorkspace.hidden = workspaceMode !== "note";
    elements.chatWorkspaceButton.dataset.active = String(workspaceMode === "chat");
    renderNoteList();
    if (workspaceMode === "chat") elements.composerInput.focus();
    else if (activeNote) elements.noteEditor.focus();
  }

  async function loadNotes() {
    const response = await request("/api/notes");
    noteFiles = response.files || [];
    renderNoteList();
  }

  function renderNoteList() {
    if (!noteFiles.length) {
      const empty = document.createElement("div");
      empty.className = "note-list-empty";
      empty.textContent = "Create or open a Markdown/text file.";
      elements.noteList.replaceChildren(empty);
      return;
    }
    elements.noteList.replaceChildren(...noteFiles.map((name) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = name;
      button.title = name;
      button.dataset.active = String(activeNote?.name === name && workspaceMode === "note");
      button.addEventListener("click", () => openNote(name).catch(showLocalError));
      return button;
    }));
  }

  async function openNote(name) {
    await flushNoteSave();
    const document = await request("/api/notes/open", { name });
    activeNote = document;
    noteDirty = false;
    elements.activeNoteName.textContent = document.name;
    elements.noteEditor.value = document.content;
    elements.noteEditor.disabled = false;
    setNoteStatus("Saved", "saved");
    setWorkspaceMode("note");
  }

  async function createNote() {
    const document = await request("/api/notes/create", { name: "Untitled.md" });
    await loadNotes();
    activeNote = document;
    noteDirty = false;
    elements.activeNoteName.textContent = document.name;
    elements.noteEditor.value = document.content;
    elements.noteEditor.disabled = false;
    setNoteStatus("Saved", "saved");
    setWorkspaceMode("note");
  }

  async function importNoteFile(file) {
    const content = await file.text();
    const document = await request("/api/notes/import", { name: file.name, content });
    await loadNotes();
    activeNote = document;
    noteDirty = false;
    elements.activeNoteName.textContent = document.name;
    elements.noteEditor.value = document.content;
    elements.noteEditor.disabled = false;
    setNoteStatus("Imported", "saved");
    setWorkspaceMode("note");
  }

  function scheduleNoteSave() {
    if (!activeNote) return;
    noteDirty = true;
    setNoteStatus("Unsaved changes", "saving");
    clearTimeout(noteSaveTimer);
    noteSaveTimer = setTimeout(() => saveActiveNote().catch(showLocalError), 450);
  }

  async function saveActiveNote() {
    if (!activeNote || !noteDirty) return;
    clearTimeout(noteSaveTimer);
    noteSaveTimer = null;
    const name = activeNote.name;
    const content = elements.noteEditor.value;
    setNoteStatus("Saving…", "saving");
    try {
      const document = await request("/api/notes/save", { name, content });
      if (activeNote?.name === name && elements.noteEditor.value === content) {
        activeNote = document;
        noteDirty = false;
        setNoteStatus("Saved", "saved");
      }
    } catch (error) {
      setNoteStatus("Save failed", "error");
      throw error;
    }
  }

  async function flushNoteSave() {
    if (noteSaveTimer) clearTimeout(noteSaveTimer);
    noteSaveTimer = null;
    await saveActiveNote();
  }

  function setNoteStatus(text, state = "") {
    elements.noteSaveStatus.textContent = text;
    elements.noteSaveStatus.dataset.state = state;
  }

  function restoreWorkspaceSizes() {
    const dock = Number(localStorage.getItem("live-assistant-dock-height"));
    const sidebar = Number(localStorage.getItem("live-assistant-notes-width"));
    if (Number.isFinite(dock) && dock >= 142) elements.workspace.style.setProperty("--dock-height", `${dock}px`);
    if (Number.isFinite(sidebar) && sidebar >= 90) elements.workspace.style.setProperty("--notes-width", `${sidebar}px`);
  }

  function installResizeHandle(handle, kind) {
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      elements.workspace.dataset.resizing = kind;
      const bounds = elements.workspace.getBoundingClientRect();
      const startY = event.clientY;
      const startX = event.clientX;
      const startDock = elements.bottomWorkspace.getBoundingClientRect().height;
      const startSidebar = elements.noteSidebar.getBoundingClientRect().width;
      let latestDock = startDock;
      let latestSidebar = startSidebar;
      let finished = false;
      const move = (moveEvent) => {
        if (kind === "dock") {
          latestDock = Math.max(142, Math.min(bounds.height * .72, startDock + startY - moveEvent.clientY));
          elements.workspace.style.setProperty("--dock-height", `${latestDock}px`);
        } else {
          latestSidebar = Math.max(90, Math.min(340, startSidebar + moveEvent.clientX - startX));
          elements.workspace.style.setProperty("--notes-width", `${latestSidebar}px`);
        }
      };
      const finish = () => {
        if (finished) return;
        finished = true;
        window.removeEventListener("pointermove", move, true);
        window.removeEventListener("pointerup", finish, true);
        window.removeEventListener("pointercancel", finish, true);
        delete elements.workspace.dataset.resizing;
        localStorage.setItem("live-assistant-dock-height", String(Math.round(latestDock)));
        localStorage.setItem("live-assistant-notes-width", String(Math.round(latestSidebar)));
      };
      window.addEventListener("pointermove", move, true);
      window.addEventListener("pointerup", finish, true);
      window.addEventListener("pointercancel", finish, true);
      try { handle.setPointerCapture(event.pointerId); } catch (_) {}
    });
  }

  async function ensureAutoConnection() {
    const tab = activeTab();
    if (!tab || tab.connection !== "offline" || autoConnectAttemptedTabId === tab.id) return;
    autoConnectAttemptedTabId = tab.id;
    try {
      await request("/api/connect", { tab_id: tab.id, api_key: null });
    } catch (error) {
      showLocalError(error);
    }
  }

  function showLocalError(error) { localError = error?.message || String(error); renderError(); }
  function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char])); }
  function formatBytes(bytes) { if (bytes < 1024) return `${bytes} B`; if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`; return `${(bytes / 1048576).toFixed(1)} MB`; }

  elements.connectButton.addEventListener("click", () => toggleConnection().catch(showLocalError));
  elements.clearButton.addEventListener("click", () => request("/api/clear", { tab_id: activeTab()?.id }).catch(showLocalError));
  elements.settingsButton.addEventListener("click", openSettings);
  elements.modelPill.addEventListener("click", openSettings);
  elements.addTab.addEventListener("click", openModelPicker);
  elements.closeSettings.addEventListener("click", closeSettings);
  elements.closeModelPicker.addEventListener("click", closeModelPicker);
  for (const modal of [elements.settingsModal, elements.modelModal]) modal.addEventListener("click", (event) => { if (event.target === modal) modal === elements.settingsModal ? closeSettings() : closeModelPicker(); });
  elements.auth.addEventListener("change", () => { elements.apiKeyField.dataset.visible = String(elements.auth.value === "api_key"); });
  elements.refreshModels.addEventListener("click", () => request("/api/catalog/refresh", { api_key: elements.apiKey.value.trim() || null }).catch(showLocalError));
  elements.settingsForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const tabId = activeTab().id;
      await request("/api/settings", { tab_id: tabId, settings: settingsPayload() });
      manualDisconnectTabId = null;
      autoConnectAttemptedTabId = null;
      micOptOutTabId = null;
      closeSettings();
    } catch (error) { showLocalError(error); }
  });
  elements.modelPickerForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await request("/api/tabs/add", { backend: newTabBackend, model: checkedValue("new-model") || "", voice: newTabBackend === "codex_text" ? null : elements.newVoice.value });
      manualDisconnectTabId = null;
      autoConnectAttemptedTabId = null;
      closeModelPicker();
    } catch (error) { showLocalError(error); }
  });
  elements.composerForm.addEventListener("submit", (event) => { event.preventDefault(); sendComposer().catch(showLocalError); });
  elements.composerInput.addEventListener("input", () => { resizeComposer(); updateSendButton(); });
  elements.composerInput.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); sendComposer().catch(showLocalError); } });
  elements.uploadButton.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", () => uploadFiles(elements.fileInput.files).catch(showLocalError));
  document.addEventListener("paste", (event) => {
    const images = [...(event.clipboardData?.files || [])].filter((file) => file.type.startsWith("image/"));
    if (!images.length) return;
    event.preventDefault();
    uploadFiles(images).catch(showLocalError);
  });
  for (const dropTarget of [elements.composerForm, elements.conversation]) {
    dropTarget.addEventListener("dragover", (event) => {
      if ([...(event.dataTransfer?.items || [])].some((item) => item.kind === "file")) {
        event.preventDefault();
        elements.composerForm.dataset.dragging = "true";
      }
    });
    dropTarget.addEventListener("dragleave", (event) => {
      if (!dropTarget.contains(event.relatedTarget)) elements.composerForm.dataset.dragging = "false";
    });
    dropTarget.addEventListener("drop", (event) => {
      const attachments = [...(event.dataTransfer?.files || [])].filter((file) => file.type.startsWith("image/") || file.type.startsWith("audio/"));
      elements.composerForm.dataset.dragging = "false";
      if (!attachments.length) return;
      event.preventDefault();
      uploadFiles(attachments).catch(showLocalError);
    });
  }
  elements.captureButton.addEventListener("click", () => request("/api/capture-screen", { tab_id: activeTab()?.id }).catch(showLocalError));
  elements.micButton.addEventListener("click", () => {
    const tab = activeTab();
    if (micActive) {
      micOptOutTabId = tab?.id ?? null;
      stopMicrophone();
      return;
    }
    micOptOutTabId = null;
    micFailedTabId = null;
    Promise.resolve(startMicrophone()).catch(showLocalError);
  });
  elements.chatWorkspaceButton.addEventListener("click", () => setWorkspaceMode("chat"));
  elements.newNoteButton.addEventListener("click", () => createNote().catch(showLocalError));
  elements.openNoteButton.addEventListener("click", () => elements.noteFileInput.click());
  elements.noteFileInput.addEventListener("change", () => {
    const file = elements.noteFileInput.files?.[0];
    elements.noteFileInput.value = "";
    if (file) importNoteFile(file).catch(showLocalError);
  });
  elements.noteEditor.addEventListener("input", scheduleNoteSave);
  elements.noteEditor.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveActiveNote().catch(showLocalError);
    }
  });
  installResizeHandle(elements.dockResize, "dock");
  installResizeHandle(elements.sidebarResize, "sidebar");
  elements.dismissError.addEventListener("click", () => { dismissedError = localError || activeTab()?.error || null; localError = null; renderError(); });
  for (const suggestion of document.querySelectorAll("[data-prompt]")) suggestion.addEventListener("click", () => { elements.composerInput.value = suggestion.dataset.prompt; resizeComposer(); updateSendButton(); elements.composerInput.focus(); });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (!elements.modelModal.hidden) closeModelPicker();
      else if (!elements.settingsModal.hidden) closeSettings();
      else if (workspaceMode === "note") setWorkspaceMode("chat");
    }
    if ((event.metaKey || event.ctrlKey) && event.key === ",") { event.preventDefault(); openSettings(); }
  });
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") saveActiveNote().catch(() => {}); });
  window.addEventListener("beforeunload", () => { stopMicrophone(); if (noteDirty) saveActiveNote().catch(() => {}); });

  async function boot() {
    restoreWorkspaceSizes();
    setWorkspaceMode("chat");
    connectSocket();
    try {
      const [state] = await Promise.all([request("/api/state"), loadNotes()]);
      applyState(state);
      await ensureAutoConnection();
    } catch (error) { showLocalError(error); }
    resizeComposer();
    updateSendButton();
  }
  boot();
})();
