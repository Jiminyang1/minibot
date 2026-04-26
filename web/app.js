const form = document.querySelector("#form");
const input = document.querySelector("#input");
const session = document.querySelector("#session");
const sessionSelect = document.querySelector("#sessionSelect");
const reloadHistory = document.querySelector("#reloadHistory");
const feed = document.querySelector("#feed");
const conversation = document.querySelector("#conversation");
const statusEl = document.querySelector("#status");
const themeToggle = document.querySelector("#themeToggle");
const runTag = document.querySelector("#runTag");
const sessionTag = document.querySelector("#sessionTag");
const send = document.querySelector("#send");

const THEME_KEY = "minibot.theme";

let activeSessionId = "current";

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

function scrollConversation() {
  conversation.scrollTop = conversation.scrollHeight;
}

function setActiveSession(sessionId) {
  activeSessionId = sessionId || "current";
  session.value = activeSessionId;
  sessionTag.textContent = activeSessionId;
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

  const role = document.createElement("div");
  role.className = "message-role";
  role.textContent = messageLabel(message);

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = messageContent(message);

  row.append(role, content);
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
  if (payload.current_session_id) {
    setActiveSession(payload.current_session_id);
    sessionSelect.value = payload.current_session_id;
  }
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
  if (event.type === "run.failed") return `${p.error_type || "error"}: ${p.message || ""}`;
  return JSON.stringify(p);
}

function appendEvent(event) {
  if (feed.querySelector(".empty")) feed.innerHTML = "";
  runTag.textContent = event.run_id || "run";
  if (event.session_id) setActiveSession(event.session_id);

  const row = document.createElement("div");
  row.className = "event";

  const type = document.createElement("div");
  type.className = "event-type";
  type.textContent = event.type;

  const body = document.createElement("div");
  body.className = "event-body";
  body.textContent = summarize(event);

  if (event.type === "approval.required") {
    const actions = document.createElement("span");
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

  row.append(type, body);
  feed.append(row);
  feed.scrollTop = feed.scrollHeight;

  if (event.type === "message.completed" && event.payload && event.payload.content) {
    appendMessage({
      role: "assistant",
      content: event.payload.content,
      created_at: event.created_at,
    });
  }
  if (event.type === "run.completed") setStatus("done", "done");
  if (event.type === "run.failed") setStatus("failed", "error");
}

async function readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\\n\\n");
    buffer = chunks.pop() || "";
    for (const chunk of chunks) {
      const lines = chunk.split("\\n");
      const dataLine = lines.find((line) => line.startsWith("data: "));
      if (!dataLine) continue;
      appendEvent(JSON.parse(dataLine.slice(6)));
    }
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  clearEvents();
  appendMessage({ role: "user", content: text });
  input.value = "";
  send.disabled = true;
  setStatus("running", "running");
  const requestedSession = session.value.trim() || "current";
  try {
    const response = await fetch("/runs/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify({
        input: text,
        session_id: requestedSession,
      }),
    });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    await readSSE(response);
    await loadSessions();
  } catch (error) {
    setStatus("failed", "error");
    appendEvent({
      id: "client-error",
      run_id: "client",
      session_id: activeSessionId,
      seq: 0,
      type: "run.failed",
      created_at: new Date().toISOString(),
      payload: { error_type: "ClientError", message: String(error) },
    });
  } finally {
    send.disabled = false;
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

applyTheme(localStorage.getItem(THEME_KEY) || "dark");
clearEvents();
loadHistory("current").catch((error) => {
  setStatus("failed", "error");
  renderEmptyConversation(String(error));
});
