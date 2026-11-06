async function clearModelCache() {
  const buttons = Array.from(document.querySelectorAll(".clear-cache-btn"));
  if (!buttons.length) {
    return;
  }

  const originalLabels = buttons.map((button) => button.textContent);
  buttons.forEach((button) => {
    button.disabled = true;
    button.textContent = "Clearing...";
  });

  try {
    const response = await fetch("/api/system/clear-model-cache", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "cache clear failed");
    }

    const before = data.rss_before_mb;
    const after = data.rss_after_mb;
    const label = before != null && after != null
      ? `Cache Cleared (${before} -> ${after} MB)`
      : "Cache Cleared";

    buttons.forEach((button) => {
      button.textContent = label;
      button.title = data.note || "";
    });
  } catch (error) {
    buttons.forEach((button) => {
      button.textContent = `Cache Error`;
      button.title = error.message;
    });
  } finally {
    setTimeout(() => {
      buttons.forEach((button, index) => {
        button.disabled = false;
        button.textContent = originalLabels[index];
      });
    }, 2500);
  }
}

function setupFieldHelp() {
  const helpPanels = Array.from(document.querySelectorAll(".field-help"));
  if (!helpPanels.length) {
    return;
  }

  const closeOthers = (activePanel) => {
    helpPanels.forEach((panel) => {
      if (panel !== activePanel) {
        panel.removeAttribute("open");
      }
    });
  };

  helpPanels.forEach((panel) => {
    panel.addEventListener("toggle", () => {
      if (panel.open) {
        closeOthers(panel);
      }
    });
  });

  document.addEventListener("click", (event) => {
    const insideHelp = event.target.closest(".field-help");
    if (insideHelp) {
      return;
    }

    helpPanels.forEach((panel) => panel.removeAttribute("open"));
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }

    helpPanels.forEach((panel) => panel.removeAttribute("open"));
  });
}

document.querySelectorAll(".clear-cache-btn").forEach((button) => {
  button.addEventListener("click", clearModelCache);
});

setupFieldHelp();
