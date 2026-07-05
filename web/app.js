import { marked } from "https://cdn.jsdelivr.net/npm/marked@15/+esm";
import DOMPurify from "https://cdn.jsdelivr.net/npm/dompurify@3/+esm";

marked.setOptions({ breaks: true, gfm: true });

function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text || ""));
}

function formatClock(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  return d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function eventCategory(type) {
  if (type === "run.failed" || type === "tool_call.failed") return "fail";
  if (type.startsWith("run.")) return "run";
  if (type.startsWith("model.")) return "model";
  if (type.startsWith("tool_call.")) return "tool";
  if (type.startsWith("approval.")) return "approval";
  if (type.startsWith("message.")) return "message";
  if (type.startsWith("context.")) return "context";
  return "other";
}

const form = document.querySelector("#form");
const input = document.querySelector("#input");
const session = document.querySelector("#session");
const sessionSelect = document.querySelector("#sessionSelect");
const reloadHistory = document.querySelector("#reloadHistory");
const newSessionBtn = document.querySelector("#newSession");
const renameSessionBtn = document.querySelector("#renameSession");
const deleteSessionBtn = document.querySelector("#deleteSession");
const feed = document.querySelector("#feed");
const conversation = document.querySelector("#conversation");
const statusEl = document.querySelector("#status");
const themeToggle = document.querySelector("#themeToggle");
const runTag = document.querySelector("#runTag");
const sessionTag = document.querySelector("#sessionTag");
const send = document.querySelector("#send");

const THEME_KEY = "minibot.theme";
const RUNTIME_EVENT_TYPES = [
  "run.started",
  "context.usage",
  "context.compacted",
  "model.request.started",
  "model.request.completed",
  "tool_call.started",
  "approval.required",
  "approval.resolved",
  "tool_call.completed",
  "tool_call.failed",
  "message.delta",
  "message.completed",
  "run.completed",
  "run.cancelled",
  "run.failed",
];

let activeSessionId = "current";
let currentRunId = null;
let currentEventSource = null;
let streamingBubble = null;

function setSendState(state) {
  const next = state === "stop" ? "stop" : "send";
  send.dataset.state = next;
  send.setAttribute(
    "aria-label",
    next === "stop" ? "stop run" : "send message",
  );
  send.title = next === "stop" ? "Stop running" : "Send (⌘/Ctrl + Enter)";
  send.disabled = false;
}

async function cancelRun() {
  if (!currentRunId) return;
  send.disabled = true;
  try {
    await fetch(`/runs/${encodeURIComponent(currentRunId)}/cancel`, {
      method: "POST",
    });
  } catch (error) {
    console.error("cancel failed", error);
  } finally {
    send.disabled = false;
  }
}

function applyTheme(theme) {
  const resolved = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = resolved;
  themeToggle.textContent = resolved;
  themeToggle.setAttribute("aria-label", `switch to ${resolved === "light" ? "dark" : "light"} theme`);
  localStorage.setItem(THEME_KEY, resolved);
}

function setStatus(text, state = "idle") {
  statusEl.textContent = text;
  statusEl.dataset.state = state;
}

function clearEvents() {
  feed.innerHTML = '<div class="empty">waiting</div>';
  runTag.textContent = "no run";
}

function closeEventSource() {
  if (!currentEventSource) return;
  currentEventSource.close();
  currentEventSource = null;
}

function scrollConversation() {
  conversation.scrollTop = conversation.scrollHeight;
}

function setActiveSession(sessionId) {
  activeSessionId = sessionId || "current";
  session.value = activeSessionId;
  sessionTag.textContent = activeSessionId;
  updateSessionButtons();
}

function updateSessionButtons() {
  const protectedAlias = !activeSessionId || activeSessionId === "current";
  renameSessionBtn.disabled = protectedAlias;
  deleteSessionBtn.disabled = protectedAlias;
}

function findSessionTitle(sessionId) {
  for (const option of sessionSelect.options) {
    if (option.value !== sessionId) continue;
    const parts = option.textContent.split("·");
    if (parts.length < 2) return "";
    return parts.slice(1).join("·").trim();
  }
  return "";
}

function renderEmptyConversation(text = "empty") {
  conversation.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "empty";
  empty.textContent = text;
  conversation.append(empty);
}

function messageLabel(message) {
  return message.role;
}

function messageContent(message) {
  if (typeof message.content === "string") return message.content;
  if (message.content == null) return "";
  return JSON.stringify(message.content);
}

function isConversationMessage(message) {
  if (!["user", "assistant"].includes(message.role)) return false;
  return Boolean(messageContent(message).trim());
}

function appendMessage(message) {
  if (!isConversationMessage(message)) return;
  if (conversation.querySelector(".empty")) conversation.innerHTML = "";
  const row = document.createElement("div");
  row.className = `message ${message.role || ""}`;

  const meta = document.createElement("div");
  meta.className = "message-meta";

  const role = document.createElement("span");
  role.className = "message-role";
  role.textContent = messageLabel(message);
  meta.append(role);

  const time = formatClock(message.created_at);
  if (time) {
    const timeEl = document.createElement("span");
    timeEl.className = "message-time";
    timeEl.textContent = time;
    meta.append(timeEl);
  }

  const content = document.createElement("div");
  content.className = "message-content";
  const text = messageContent(message);
  if (message.role === "assistant") {
    content.classList.add("markdown");
    content.innerHTML = renderMarkdown(text);

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "message-copy";
    copyBtn.textContent = "copy";
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = "copied";
        setTimeout(() => {
          copyBtn.textContent = "copy";
        }, 1200);
      } catch {
        copyBtn.textContent = "failed";
      }
    });
    meta.append(copyBtn);
  } else {
    content.textContent = text;
  }

  row.append(meta, content);
  conversation.append(row);
  scrollConversation();
}

function renderConversation(messages) {
  conversation.innerHTML = "";
  const visible = messages.filter(isConversationMessage);
  if (!visible.length) {
    renderEmptyConversation("empty");
    return;
  }
  visible.forEach(appendMessage);
}

async function loadSessions() {
  const previousActive = activeSessionId;
  const response = await fetch("/sessions");
  if (!response.ok) throw new Error(`sessions HTTP ${response.status}`);
  const payload = await response.json();
  sessionSelect.innerHTML = "";
  const currentOption = document.createElement("option");
  currentOption.value = "current";
  currentOption.textContent = "current";
  sessionSelect.append(currentOption);
  for (const item of payload.sessions || []) {
    const option = document.createElement("option");
    option.value = item.session_id;
    option.textContent = `${item.session_id} · ${item.title}`;
    sessionSelect.append(option);
  }
  const optionValues = new Set(
    Array.from(sessionSelect.options, (option) => option.value),
  );
  const fallbackSessionId = payload.current_session_id || "current";
  const nextActive = optionValues.has(previousActive)
    ? previousActive
    : fallbackSessionId;
  setActiveSession(nextActive);
  sessionSelect.value = nextActive;
}

async function loadHistory(sessionId = activeSessionId) {
  const requested = sessionId || "current";
  const response = await fetch(`/sessions/${encodeURIComponent(requested)}/messages`);
  if (response.status === 404) {
    setActiveSession(requested);
    renderEmptyConversation("session not found");
    return;
  }
  if (!response.ok) throw new Error(`messages HTTP ${response.status}`);
  const payload = await response.json();
  setActiveSession(payload.session.session_id);
  renderConversation(payload.messages || []);
  await loadSessions();
  sessionSelect.value = activeSessionId;
}

function summarize(event) {
  const p = event.payload || {};
  if (event.type === "run.started") return p.input_preview || "";
  if (event.type === "context.usage") return `${p.current_tokens}/${p.budget} tokens`;
  if (event.type === "model.request.started") return `iteration ${p.iteration}`;
  if (event.type === "model.request.completed") {
    if (p.empty_reply) return "empty reply";
    return `${p.tool_call_count || 0} tool calls`;
  }
  if (event.type === "tool_call.started") return `${p.tool || ""} ${JSON.stringify(p.args || {})}`;
  if (event.type === "tool_call.completed" || event.type === "tool_call.failed") return p.summary || p.code || "";
  if (event.type === "approval.required") return `approval required for ${p.tool || ""}`;
  if (event.type === "approval.resolved") return p.approved ? "approved" : "denied";
  if (event.type === "message.completed") return "assistant message";
  if (event.type === "run.completed") return "done";
  if (event.type === "run.cancelled") return "cancelled by user";
  if (event.type === "run.failed") return `${p.error_type || "error"}: ${p.message || ""}`;
  return JSON.stringify(p);
}

function finishRun(status, state) {
  clearStreamingBubble();
  setStatus(status, state);
  setSendState("send");
  currentRunId = null;
  closeEventSource();
  loadSessions().catch((error) => console.error("load sessions failed", error));
}

function appendStreamDelta(event) {
  const p = event.payload || {};
  if (p.channel !== "text" || !p.text) return;
  if (conversation.querySelector(".empty")) conversation.innerHTML = "";
  if (!streamingBubble) {
    const row = document.createElement("div");
    row.className = "message assistant streaming";
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const role = document.createElement("span");
    role.className = "message-role";
    role.textContent = "assistant";
    meta.append(role);
    const content = document.createElement("div");
    content.className = "message-content";
    row.append(meta, content);
    conversation.append(row);
    streamingBubble = { row, content, buffer: "" };
  }
  streamingBubble.buffer += p.text;
  streamingBubble.content.textContent = streamingBubble.buffer;
  conversation.scrollTop = conversation.scrollHeight;
}

function clearStreamingBubble() {
  if (!streamingBubble) return;
  streamingBubble.row.remove();
  streamingBubble = null;
}

function appendEvent(event) {
  if (event.type === "message.delta") {
    appendStreamDelta(event);
    return;
  }
  if (feed.querySelector(".empty")) feed.innerHTML = "";
  runTag.textContent = event.run_id || "run";
  if (event.type === "run.started" && event.run_id) {
    currentRunId = event.run_id;
  }
  if (event.session_id) setActiveSession(event.session_id);

  const row = document.createElement("div");
  row.className = "event";
  row.dataset.category = eventCategory(event.type);
  if (event.type === "approval.required") row.classList.add("is-approval");

  const time = document.createElement("span");
  time.className = "event-time";
  time.textContent = formatClock(event.created_at);

  const pill = document.createElement("span");
  pill.className = "event-pill";
  pill.textContent = event.type;

  const body = document.createElement("div");
  body.className = "event-body";
  const summary = document.createElement("div");
  summary.className = "event-summary";
  summary.textContent = summarize(event);
  body.append(summary);

  if (event.type === "approval.required") {
    const actions = document.createElement("div");
    actions.className = "approval-actions";
    const approve = document.createElement("button");
    approve.className = "small approve";
    approve.type = "button";
    approve.textContent = "approve";
    const deny = document.createElement("button");
    deny.className = "small deny";
    deny.type = "button";
    deny.textContent = "deny";
    const resolve = async (approved) => {
      approve.disabled = true;
      deny.disabled = true;
      row.classList.add("is-resolved");
      await fetch(`/runs/${event.run_id}/approvals/${event.payload.approval_id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved }),
      });
    };
    approve.addEventListener("click", () => resolve(true));
    deny.addEventListener("click", () => resolve(false));
    actions.append(approve, deny);
    body.append(actions);
  }

  row.append(time, pill, body);
  feed.append(row);
  feed.scrollTop = feed.scrollHeight;

  if (
    event.type === "model.request.completed" &&
    streamingBubble &&
    (event.payload?.tool_call_count || 0) > 0
  ) {
    // Intermediate narration before tool calls: freeze the bubble so the
    // next iteration streams into a fresh one.
    streamingBubble.row.classList.remove("streaming");
    streamingBubble = null;
  }
  if (event.type === "message.completed" && event.payload && event.payload.content) {
    // The completed event carries the authoritative text; the streamed
    // bubble is advisory and gets replaced wholesale.
    clearStreamingBubble();
    appendMessage({
      role: "assistant",
      content: event.payload.content,
      created_at: event.created_at,
    });
  }
  if (event.type === "run.completed") finishRun("done", "done");
  if (event.type === "run.cancelled") finishRun("cancelled", "error");
  if (event.type === "run.failed") finishRun("failed", "error");
}

function subscribeRun(runId) {
  closeEventSource();
  const source = new EventSource(`/runs/${encodeURIComponent(runId)}/events`);
  currentEventSource = source;

  for (const type of RUNTIME_EVENT_TYPES) {
    source.addEventListener(type, (event) => {
      try {
        appendEvent(JSON.parse(event.data));
      } catch (error) {
        console.error("bad SSE event:", event.data, error);
      }
    });
  }

  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED && currentRunId === runId) {
      finishRun("disconnected", "error");
    }
  };
}

send.addEventListener("click", (event) => {
  if (send.dataset.state === "stop") {
    event.preventDefault();
    cancelRun();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (send.dataset.state === "stop") return;
  const text = input.value.trim();
  if (!text) return;
  clearEvents();
  appendMessage({ role: "user", content: text });
  input.value = "";
  currentRunId = null;
  setSendState("stop");
  setStatus("running", "running");
  const requestedSession = session.value.trim() || "current";
  try {
    const payload = await jsonRequest("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input: text,
        session_id: requestedSession,
      }),
    });
    currentRunId = payload.run_id;
    runTag.textContent = currentRunId || "run";
    subscribeRun(currentRunId);
  } catch (error) {
    setStatus("failed", "error");
    setSendState("send");
    currentRunId = null;
    appendEvent({
      id: "client-error",
      run_id: "client",
      session_id: activeSessionId,
      seq: 0,
      type: "run.failed",
      created_at: new Date().toISOString(),
      payload: { error_type: "ClientError", message: String(error) },
    });
  }
});

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.dataset.theme || "dark";
  applyTheme(current === "light" ? "dark" : "light");
});

sessionSelect.addEventListener("change", async () => {
  setActiveSession(sessionSelect.value || "current");
  await loadHistory(activeSessionId);
});

session.addEventListener("change", async () => {
  setActiveSession(session.value.trim() || "current");
  await loadHistory(activeSessionId);
});

reloadHistory.addEventListener("click", async () => {
  await loadHistory(session.value.trim() || "current");
});

async function jsonRequest(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

async function createNewSession() {
  newSessionBtn.disabled = true;
  try {
    const payload = await jsonRequest("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const id = payload.session.session_id;
    clearEvents();
    setStatus("idle", "idle");
    setActiveSession(id);
    await loadHistory(id);
  } catch (error) {
    alert(`新建失败: ${error.message}`);
  } finally {
    newSessionBtn.disabled = false;
  }
}

async function renameCurrentSession() {
  const id = activeSessionId;
  if (!id || id === "current") return;
  const current = findSessionTitle(id);
  const proposed = prompt(`Rename session ${id}`, current);
  if (proposed == null) return;
  const trimmed = proposed.trim();
  if (!trimmed || trimmed === current) return;
  try {
    await jsonRequest(`/sessions/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: trimmed }),
    });
    await loadSessions();
    sessionSelect.value = id;
  } catch (error) {
    alert(`重命名失败: ${error.message}`);
  }
}

async function deleteCurrentSession() {
  const id = activeSessionId;
  if (!id || id === "current") return;
  if (!confirm(`确认删除会话 ${id}? 此操作不可恢复。`)) return;
  try {
    await jsonRequest(`/sessions/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    clearEvents();
    setStatus("idle", "idle");
    setActiveSession("current");
    await loadHistory("current");
  } catch (error) {
    alert(`删除失败: ${error.message}`);
  }
}

newSessionBtn.addEventListener("click", () => {
  createNewSession();
});

renameSessionBtn.addEventListener("click", () => {
  renameCurrentSession();
});

deleteSessionBtn.addEventListener("click", () => {
  deleteCurrentSession();
});

input.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    if (send.dataset.state === "stop") {
      cancelRun();
    } else if (!send.disabled) {
      form.requestSubmit();
    }
  }
});

applyTheme(localStorage.getItem(THEME_KEY) || "dark");
clearEvents();
updateSessionButtons();
loadHistory("current").catch((error) => {
  setStatus("failed", "error");
  renderEmptyConversation(String(error));
});
