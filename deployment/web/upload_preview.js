// Runs only inside the authenticated page. Preview uses a local Blob URL; no
// second upload, file-serving endpoint, server decoder, or checkpoint is used.
(function () {
  const video = element.querySelector("video");
  const note = element.querySelector(".av-preview-note");
  let url = null;
  function clear() {
    video.pause(); video.removeAttribute("src"); video.load(); video.hidden = true;
    if (url) URL.revokeObjectURL(url);
    url = null;
    note.textContent = "Choose a clip to preview it here. Preview playback is not a verification result.";
  }
  function change(event) {
    const input = event.target;
    const dropped = event.type === "drop" && input.closest && input.closest("#av-upload-source");
    if (!dropped && (!input.matches || !input.matches("#av-upload-source input[type=file]"))) return;
    clear();
    const files = dropped ? event.dataTransfer.files : input.files;
    const file = files && files[0];
    const limit = Number(element.querySelector(".av-upload-preview").dataset.limit);
    if (!file) return;
    if (!/\.(mp4|mov|mkv|webm)$/i.test(file.name) || file.size > limit || file.size === 0) {
      note.textContent = "Choose a supported video within the upload limit. Verify checks the file on the server.";
      return;
    }
    url = URL.createObjectURL(file); video.src = url; video.hidden = false;
    note.textContent = "Local preview only. Press Verify to check the upload; predictions remain disabled.";
  }
  video.addEventListener("error", function () {
    note.textContent = "This browser cannot preview this clip. You can still press Verify for server validation.";
  });
  // Gradio can clear a File without firing the native input's change event.
  // A capture-phase click sees its Clear control before the framework removes it.
  function click(event) {
    const target = event.target.closest && event.target.closest("#av-upload-source button");
    if (target && /clear|remove/i.test((target.getAttribute("aria-label") || "") +
        (target.getAttribute("title") || "") + target.textContent)) clear();
  }
  document.addEventListener("change", change, true);
  document.addEventListener("drop", change, true);
  document.addEventListener("click", click, true);
  window.addEventListener("pagehide", clear);
  const observer = new MutationObserver(function () {
    if (!element.isConnected) {
      clear(); observer.disconnect();
      document.removeEventListener("change", change, true);
      document.removeEventListener("drop", change, true);
      document.removeEventListener("click", click, true);
      window.removeEventListener("pagehide", clear);
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
  clear();
})();
