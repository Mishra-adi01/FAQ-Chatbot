const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const chatWidget = document.getElementById("chatWidget");
const quickReplies = document.getElementById("quickReplies");

// ------------------------------
// LOAD PREVIOUS CHAT
// ------------------------------
window.onload = () => {
    // Don't load previous chat - always start fresh
    // Show only the initial welcome message
    chatBox.innerHTML = `
        <div class="message bot">
            Hello 👋 Welcome to Athenura Internship.
        </div>
        <div class="quick-suggestion">
            <button class="suggestion-btn">Internship Info</button>
        </div>
    `;
};

// ------------------------------
// ENTER KEY SEND
// ------------------------------
userInput.addEventListener("keypress", function(e) {
    if (e.key === "Enter") sendMessage();
});

// ------------------------------
// OPEN / CLOSE CHAT
// ------------------------------
function toggleChat() {
    if (chatWidget.style.display === "none" || chatWidget.style.display === "") {
        chatWidget.style.display = "block";
    } else {
        chatWidget.style.display = "none";
    }
}

// ------------------------------
// QUICK REPLY SELECTION
// ------------------------------
function selectQuickReply(text) {
    userInput.value = text;
    sendMessage();
    quickReplies.style.display = "none";
}



// ------------------------------
// SEND MESSAGE
// ------------------------------
function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, "user");
    userInput.value = "";

    // Typing indicator
    const typing = addMessage("Typing...", "bot");

    fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    })
    .then(res => {
        if (!res.ok) throw new Error('Network response was not ok');
        return res.json();
    })
    .then(data => {
        typing.remove();
        addMessage(data.response, "bot");
    })
    .catch((error) => {
        console.error('Error:', error);
        typing.remove();
        addMessage("Server error. Please try again.", "bot");
    });
}

// ------------------------------
// ADD MESSAGE
// ------------------------------
function addMessage(text, sender) {
    const msg = document.createElement("div");
    msg.classList.add("message", sender);
    msg.innerText = text;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Don't save to localStorage - keep chat fresh on reload

    return msg;
}
