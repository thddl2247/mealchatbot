document.addEventListener("DOMContentLoaded", () => {
  const panel = document.getElementById("chatbot-panel");
  const fab = document.querySelector(".chatbot-fab");
  const iframe = panel.querySelector(".chatbot-frame");
  const closeButton = panel.querySelector(".chatbot-panel-close");
  const openTriggers = document.querySelectorAll("[data-chatbot-open]");

  function openChatbot() {
    if (!iframe.src) {
      iframe.src = iframe.dataset.src;
    }
    panel.hidden = false;
    fab.hidden = true;
  }

  function closeChatbot() {
    panel.hidden = true;
    fab.hidden = false;
  }

  openTriggers.forEach((trigger) => trigger.addEventListener("click", openChatbot));
  closeButton.addEventListener("click", closeChatbot);
});
