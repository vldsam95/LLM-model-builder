const builderMessagesEl = document.getElementById("messages");
const builderFormEl = document.getElementById("chatForm");
const builderInputEl = document.getElementById("messageInput");
const builderSystemEl = document.getElementById("systemPrompt");
const builderTemperatureEl = document.getElementById("temperature");
const builderTopKEl = document.getElementById("topK");
const builderMaxTokensEl = document.getElementById("maxNewTokens");
const builderJsonModeEl = document.getElementById("jsonMode");
const builderClearBtn = document.getElementById("clearBtn");
const builderRegenBtn = document.getElementById("regenBtn");
const builderSendBtn = document.getElementById("sendBtn");

let builderLastPayload = null;

function builderLooksRTL(text) {
  return /[\u0590-\u05FF]/.test(text);
}

function builderFormatJsonIfPossible(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return null;
  }
}

function builderAddMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const formattedJson = role === "assistant" ? builderFormatJsonIfPossible(text) : null;
  if (formattedJson) {
    div.classList.add("json");
    div.textContent = formattedJson;
  } else {
    if (builderLooksRTL(text)) {
      div.classList.add("rtl");
    }
    div.textContent = text;
  }
  builderMessagesEl.appendChild(div);
  builderMessagesEl.scrollTop = builderMessagesEl.scrollHeight;
}

async function builderSend(payload) {
  builderSendBtn.disabled = true;
  try {
    const response = await fetch(`/api/builder/projects/${window.BUILDER_CHAT_BOOTSTRAP.slug}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      builderAddMessage("assistant", `Error: ${data.error || "request failed"}`);
      return;
    }
    builderAddMessage("assistant", data.response);
  } finally {
    builderSendBtn.disabled = false;
  }
}

function builderCurrentPayload() {
  return {
    message: builderInputEl.value.trim(),
    system_prompt: builderSystemEl.value,
    temperature: Number(builderTemperatureEl.value),
    top_k: Number(builderTopKEl.value),
    max_new_tokens: Number(builderMaxTokensEl.value),
    json_mode: builderJsonModeEl.checked,
  };
}

builderFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = builderCurrentPayload();
  if (!payload.message) {
    return;
  }
  builderAddMessage("user", payload.message);
  builderLastPayload = payload;
  builderInputEl.value = "";
  await builderSend(payload);
});

builderClearBtn.addEventListener("click", () => {
  builderMessagesEl.innerHTML = "";
});

builderRegenBtn.addEventListener("click", async () => {
  if (!builderLastPayload) {
    return;
  }
  await builderSend(builderLastPayload);
});

builderAddMessage("assistant", `Project preview is ready for ${window.BUILDER_CHAT_BOOTSTRAP.projectName}.`);
