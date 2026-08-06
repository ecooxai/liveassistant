(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const elements = {
    conversation: $("#conversation"),
    emptyState: $("#empty-state"),
    messageList: $("#message-list"),
    template: $("#message-template"),
    statusChip: $("#status-chip"),
    statusText: $("#status-text"),
    connectButton: $("#connect-button"),
    connectLabel: $("#connect-button .connect-label"),
    connectArrow: $("#connect-button .connect-arrow"),
    clearButton: $("#clear-button"),
    settingsButton: $("#settings-button"),
    settingsModal: $("#settings-modal"),
    closeSettings: $("#close-settings"),
    settingsForm: $("#settings-form"),
    backend: $("#backend-select"),
    auth: $("#auth-select"),
    apiKeyField: $("#api-key-field"),
    apiKey: $("#api-key-input"),
    model: $("#model-input"),
    voice: $("#voice-input"),
    thinking: $("#thinking-select"),
    screenshot: $("#screenshot-toggle"),
    imageModel: $("#image-model-input"),
    imageResolution: $("#image-resolution-select"),
    systemPrompt: $("#system-prompt-input"),
    composerForm: $("#composer-form"),
    composerInput: $("#composer-input"),
    sendButton: $("#send-button"),
    micButton: $("#mic-button"),
    micLevel: $("#mic-level"),
    modelPill: $("#model-pill"),
    errorBanner: $("#error-banner"),
    errorText: $("#error-text"),
    dismissError: $("#dismiss-error"),
  };

  let appState = null;
  let socket = null;
  let reconnectTimer = null;
  let lastRevision = -1;
  let dismissedError = null;
  let shouldStickToBottom = true;

  let micStream = null;
  let micContext = null;
  let micSource = null;
  let micProcessor = null;
  let micActive = false;

  let outputContext = null;
  let outputCursor = 0;

  const backendLabels = {
    codex_gpt_live: "GPT Live",
    open_ai_realtime: "Realtime",
    codex_text: "Codex Text",
  };

  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {
        ...(options.body instanceof ArrayBuffer ? {} : { "Content-Type": "application/json" }),
        ...(options.headers || {}),
      },
    });
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) {
      const message = typeof body === "object" && body?.error ? body.error : `Request failed (${response.status})`;
      throw new Error(message);
    }
    return body;
  }

  function connectSocket() {
    clearTimeout(reconnectTimer);
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${location.host}/ws`);
    socket.binaryType = "arraybuffer";

    socket.addEventListener("message", (event) => {
      if (typeof event.data === "string") {
        if (event.data === "pong") return;
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === "state" && payload.state) applyState(payload.state);
        } catch (error) {
          console.warn("Ignoring malformed state update", error);
        }
      } else if (event.data instanceof ArrayBuffer) {
        playPcm16(event.data);
      }
    });

    socket.addEventListener("close", () => {
      stopMicrophone();
      reconnectTimer = setTimeout(connectSocket, 900);
    });

    socket.addEventListener("error", () => socket.close());
  }

  function applyState(nextState) {
    if (!nextState || typeof nextState.revision !== "number") return;
    appState = nextState;
    renderHeader();
    renderMessages();
    renderError();
    renderSettingsAvailability();
  }

  function renderHeader() {
    if (!appState) return;
    const state = appState.connection || "offline";
    elements.statusChip.dataset.state = state;
    elements.statusText.textContent = appState.status || "Ready";

    const isLive = state === "live";
    const isBusy = state === "connecting" || state === "reconnecting";
    elements.connectButton.dataset.live = String(isLive);
    elements.connectButton.disabled = isBusy;
    elements.connectLabel.textContent = isLive ? "Disconnect" : isBusy ? "Connecting" : "Connect";
    elements.connectArrow.textContent = isLive ? "×" : isBusy ? "···" : "↗";

    const label = backendLabels[appState.settings.backend] || appState.settings.model || "Model";
    elements.modelPill.textContent = label;
    elements.micButton.disabled = !isLive || appState.settings.backend === "codex_text";
    if (!isLive && micActive) stopMicrophone();
  }

  function visibleMessages() {
    return (appState?.messages || []).filter((message) => message.role !== "system");
  }

  function renderMessages() {
    if (!appState || appState.revision === lastRevision) return;
    lastRevision = appState.revision;

    const messages = visibleMessages();
    elements.emptyState.hidden = messages.length > 0;
    elements.messageList.hidden = messages.length === 0;

    const oldDistance = elements.conversation.scrollHeight - elements.conversation.scrollTop - elements.conversation.clientHeight;
    shouldStickToBottom = oldDistance < 90 || messages.some((message) => message.streaming);

    const nodes = new Map(
      [...elements.messageList.children].map((node) => [node.dataset.id, node]),
    );
    const fragment = document.createDocumentFragment();

    for (const message of messages) {
      const id = String(message.id);
      let node = nodes.get(id);
      if (!node) {
        node = elements.template.content.firstElementChild.cloneNode(true);
        node.dataset.id = id;
      }
      updateMessageNode(node, message);
      fragment.appendChild(node);
      nodes.delete(id);
    }

    for (const staleNode of nodes.values()) staleNode.remove();
    elements.messageList.replaceChildren(fragment);

    if (shouldStickToBottom) {
      requestAnimationFrame(() => {
        elements.conversation.scrollTop = elements.conversation.scrollHeight;
      });
    }
  }

  function updateMessageNode(node, message) {
    node.dataset.role = message.role;
    node.dataset.streaming = String(Boolean(message.streaming));
    node.querySelector(".message-role").textContent = message.role === "user" ? "You" : "Assistant";
    node.querySelector("time").textContent = formatTime(message.created_at);
    node.querySelector(".message-text").textContent = message.text || (message.streaming ? "" : "No text response");

    const toolsRoot = node.querySelector(".message-tools");
    toolsRoot.replaceChildren(...(message.tool_calls || []).map(renderTool));
    toolsRoot.hidden = !message.tool_calls?.length;

    const image = node.querySelector(".message-image");
    if (message.image_url) {
      if (image.src !== message.image_url) image.src = message.image_url;
      image.hidden = false;
    } else {
      image.removeAttribute("src");
      image.hidden = true;
    }
  }

  function renderTool(tool) {
    const details = document.createElement("details");
    details.className = "tool-call";
    details.dataset.status = tool.status || "running";

    const summary = document.createElement("summary");
    summary.textContent = `${tool.name} · ${tool.status === "done" ? "Complete" : "Running"}`;
    details.appendChild(summary);

    const pre = document.createElement("pre");
    const sections = [];
    if (tool.arguments) sections.push(tool.arguments);
    if (tool.output) sections.push(`Result\n${tool.output}`);
    pre.textContent = sections.join("\n\n");
    details.appendChild(pre);
    return details;
  }

  function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "";
    return new Intl.DateTimeFormat([], { hour: "numeric", minute: "2-digit" }).format(new Date(seconds * 1000));
  }

  function renderError() {
    const error = appState?.error || null;
    const show = Boolean(error && error !== dismissedError);
    elements.errorBanner.hidden = !show;
    elements.errorText.textContent = show ? error : "";
  }

  function populateSettings() {
    if (!appState) return;
    const settings = appState.settings;
    elements.backend.value = settings.backend;
    elements.auth.value = settings.auth_mode;
    elements.model.value = settings.model;
    elements.voice.value = settings.voice;
    elements.thinking.value = settings.thinking_level;
    elements.screenshot.checked = Boolean(settings.send_screenshot);
    elements.imageModel.value = settings.image_model;
    elements.imageResolution.value = settings.image_resolution;
    elements.systemPrompt.value = settings.system_prompt;
    renderSettingsAvailability();
  }

  function renderSettingsAvailability() {
    elements.apiKeyField.dataset.visible = String(elements.auth.value === "api_key");
  }

  function settingsPayload() {
    return {
      backend: elements.backend.value,
      auth_mode: elements.auth.value,
      model: elements.model.value.trim(),
      voice: elements.voice.value.trim(),
      thinking_level: elements.thinking.value,
      system_prompt: elements.systemPrompt.value,
      send_screenshot: elements.screenshot.checked,
      image_model: elements.imageModel.value.trim(),
      image_resolution: elements.imageResolution.value,
    };
  }

  function openSettings() {
    populateSettings();
    elements.settingsModal.hidden = false;
    document.body.style.overflow = "hidden";
    requestAnimationFrame(() => elements.backend.focus());
  }

  function closeSettings() {
    elements.settingsModal.hidden = true;
    document.body.style.overflow = "";
  }

  async function toggleConnection() {
    if (!appState) return;
    dismissedError = null;
    if (appState.connection === "live") {
      stopMicrophone();
      await request("/api/disconnect", { method: "POST", body: "{}" });
      return;
    }

    if (appState.settings.auth_mode === "api_key" && !elements.apiKey.value.trim() && !(appState.has_credentials ?? appState.has_api_key)) {
      openSettings();
      elements.apiKey.focus();
      return;
    }

    await ensureOutputContext();
    await request("/api/connect", {
      method: "POST",
      body: JSON.stringify({ api_key: elements.apiKey.value.trim() || null }),
    });
    elements.apiKey.value = "";
  }

  async function sendComposer() {
    const text = elements.composerInput.value.trim();
    if (!text) return;
    if (appState?.connection !== "live") {
      await toggleConnection();
      return;
    }

    elements.composerInput.value = "";
    resizeComposer();
    updateSendButton();
    await request("/api/message", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
  }

  function resizeComposer() {
    const input = elements.composerInput;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 170)}px`;
  }

  function updateSendButton() {
    elements.sendButton.disabled = elements.composerInput.value.trim().length === 0;
  }

  async function startMicrophone() {
    if (micActive || appState?.connection !== "live") return;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone capture is unavailable in this browser");

    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    micContext = new AudioContext({ latencyHint: "interactive" });
    micSource = micContext.createMediaStreamSource(micStream);
    micProcessor = micContext.createScriptProcessor(2048, 1, 1);
    micProcessor.onaudioprocess = (event) => {
      if (!micActive || socket?.readyState !== WebSocket.OPEN) return;
      const samples = event.inputBuffer.getChannelData(0);
      const downsampled = downsample(samples, micContext.sampleRate, 24000);
      const pcm = floatToPcm16(downsampled);
      socket.send(pcm.buffer);
      updateMicMeter(samples);
    };
    micSource.connect(micProcessor);
    micProcessor.connect(micContext.destination);
    micActive = true;
    elements.micButton.dataset.active = "true";
    elements.micButton.setAttribute("aria-label", "Stop microphone");
    elements.micButton.title = "Stop microphone";
  }

  function stopMicrophone() {
    micActive = false;
    elements.micButton.dataset.active = "false";
    elements.micButton.setAttribute("aria-label", "Start microphone");
    elements.micButton.title = "Start microphone";
    elements.micLevel.style.transform = "scaleX(0)";

    if (micProcessor) {
      micProcessor.disconnect();
      micProcessor.onaudioprocess = null;
      micProcessor = null;
    }
    if (micSource) {
      micSource.disconnect();
      micSource = null;
    }
    if (micStream) {
      micStream.getTracks().forEach((track) => track.stop());
      micStream = null;
    }
    if (micContext) {
      micContext.close().catch(() => {});
      micContext = null;
    }
  }

  function downsample(input, sourceRate, targetRate) {
    if (targetRate >= sourceRate) return input;
    const ratio = sourceRate / targetRate;
    const length = Math.max(1, Math.round(input.length / ratio));
    const output = new Float32Array(length);
    let sourceOffset = 0;
    for (let index = 0; index < length; index += 1) {
      const nextOffset = Math.min(input.length, Math.round((index + 1) * ratio));
      let sum = 0;
      let count = 0;
      for (; sourceOffset < nextOffset; sourceOffset += 1) {
        sum += input[sourceOffset];
        count += 1;
      }
      output[index] = count ? sum / count : 0;
    }
    return output;
  }

  function floatToPcm16(input) {
    const output = new Int16Array(input.length);
    for (let index = 0; index < input.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, input[index]));
      output[index] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    return output;
  }

  function updateMicMeter(samples) {
    let power = 0;
    for (let index = 0; index < samples.length; index += 1) power += samples[index] * samples[index];
    const rms = Math.sqrt(power / samples.length);
    const level = Math.min(1, rms * 9);
    elements.micLevel.style.transform = `scaleX(${level})`;
  }

  async function ensureOutputContext() {
    if (!outputContext) outputContext = new AudioContext({ latencyHint: "interactive" });
    if (outputContext.state === "suspended") await outputContext.resume();
    outputCursor = Math.max(outputCursor, outputContext.currentTime);
  }

  async function playPcm16(buffer) {
    try {
      await ensureOutputContext();
      const input = new Int16Array(buffer);
      if (!input.length) return;
      const audioBuffer = outputContext.createBuffer(1, input.length, 24000);
      const channel = audioBuffer.getChannelData(0);
      for (let index = 0; index < input.length; index += 1) channel[index] = input[index] / 32768;
      const source = outputContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(outputContext.destination);
      const startAt = Math.max(outputCursor, outputContext.currentTime + 0.015);
      source.start(startAt);
      outputCursor = startAt + audioBuffer.duration;
    } catch (error) {
      console.warn("Assistant audio playback failed", error);
    }
  }

  elements.connectButton.addEventListener("click", () => toggleConnection().catch(showLocalError));
  elements.clearButton.addEventListener("click", () => {
    request("/api/clear", { method: "POST", body: "{}" }).catch(showLocalError);
  });
  elements.settingsButton.addEventListener("click", openSettings);
  elements.closeSettings.addEventListener("click", closeSettings);
  elements.settingsModal.addEventListener("click", (event) => {
    if (event.target === elements.settingsModal) closeSettings();
  });
  elements.auth.addEventListener("change", renderSettingsAvailability);
  elements.settingsForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await request("/api/settings", { method: "POST", body: JSON.stringify(settingsPayload()) });
      closeSettings();
    } catch (error) {
      showLocalError(error);
    }
  });

  elements.composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendComposer().catch(showLocalError);
  });
  elements.composerInput.addEventListener("input", () => {
    resizeComposer();
    updateSendButton();
  });
  elements.composerInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendComposer().catch(showLocalError);
    }
  });
  elements.micButton.addEventListener("click", () => {
    const action = micActive ? Promise.resolve(stopMicrophone()) : startMicrophone();
    Promise.resolve(action).catch(showLocalError);
  });
  elements.dismissError.addEventListener("click", () => {
    dismissedError = appState?.error || null;
    renderError();
  });

  for (const suggestion of document.querySelectorAll("[data-prompt]")) {
    suggestion.addEventListener("click", () => {
      elements.composerInput.value = suggestion.dataset.prompt;
      resizeComposer();
      updateSendButton();
      elements.composerInput.focus();
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !elements.settingsModal.hidden) closeSettings();
    if ((event.metaKey || event.ctrlKey) && event.key === ",") {
      event.preventDefault();
      openSettings();
    }
  });

  function showLocalError(error) {
    console.error(error);
    elements.errorText.textContent = error?.message || String(error);
    elements.errorBanner.hidden = false;
  }

  async function boot() {
    connectSocket();
    try {
      applyState(await request("/api/state"));
    } catch (error) {
      showLocalError(error);
    }
    resizeComposer();
    updateSendButton();
  }

  window.addEventListener("beforeunload", stopMicrophone);
  boot();
})();
