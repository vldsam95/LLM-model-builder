const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chatForm");
const inputEl = document.getElementById("messageInput");
const systemEl = document.getElementById("systemPrompt");
const checkpointEl = document.getElementById("checkpoint");
const temperatureEl = document.getElementById("temperature");
const topKEl = document.getElementById("topK");
const maxNewTokensEl = document.getElementById("maxNewTokens");
const jsonModeEl = document.getElementById("jsonMode");
const clearBtn = document.getElementById("clearBtn");
const regenBtn = document.getElementById("regenBtn");
const sendBtn = document.getElementById("sendBtn");

let lastPayload = null;

function looksRTL(text) {
  return /[\u0590-\u05FF]/.test(text);
}

function formatJsonIfPossible(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return null;
  }
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const formattedJson = role === "assistant" ? formatJsonIfPossible(text) : null;
  if (formattedJson) {
    div.classList.add("json");
    div.textContent = formattedJson;
  } else {
    if (looksRTL(text)) {
      div.classList.add("rtl");
    }
    div.textContent = text;
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage(payload) {
  sendBtn.disabled = true;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      addMessage("assistant", `Error: ${data.error || "request failed"}`);
      return;
    }
    addMessage("assistant", data.response);
  } finally {
    sendBtn.disabled = false;
  }
}

function currentPayload() {
  return {
    checkpoint: checkpointEl.value,
    system_prompt: systemEl.value,
    message: inputEl.value.trim(),
    temperature: Number(temperatureEl.value),
    top_k: Number(topKEl.value),
    max_new_tokens: Number(maxNewTokensEl.value),
    json_mode: jsonModeEl.checked,
  };
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = currentPayload();
  if (!payload.message) {
    return;
  }
  addMessage("user", payload.message);
  lastPayload = payload;
  inputEl.value = "";
  await sendMessage(payload);
});

clearBtn.addEventListener("click", () => {
  messagesEl.innerHTML = "";
});

regenBtn.addEventListener("click", async () => {
  if (!lastPayload) {
    return;
  }
  await sendMessage(lastPayload);
});

document.querySelectorAll(".preset-btn").forEach((button) => {
  button.addEventListener("click", () => {
    systemEl.value = button.dataset.system || "";
  });
});

addMessage("assistant", "Tiny bilingual demo is ready. Try: Hello, מה זה SEO?, or Translate 'house' to Hebrew.");
