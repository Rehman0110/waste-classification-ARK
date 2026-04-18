# =============================================================
# utils/voice_output.py
# Text-to-speech + waste suggestion system (2 classes)
# =============================================================

import threading


# ── Class → Suggestion mapping ────────────────────────────────
SUGGESTIONS = {
    "Recyclable": "Please send it to the recycling bin.",
    "Organic": "Please compost this waste."
}

ICONS = {
    "Recyclable": "♻️",
    "Organic": "🌿"
}


# ── Voice Output ───────────────────────────────────────────────
def speak(text: str, rate: int = 160, volume: float = 0.9):
    """Speak text using offline TTS (non-blocking)."""

    def _speak():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"⚠️ Voice output error: {e}")

    threading.Thread(target=_speak, daemon=True).start()


# ── Announcement ──────────────────────────────────────────────
def announce_prediction(class_name: str, confidence: float):
    """Speak prediction result."""

    suggestion = SUGGESTIONS.get(
        class_name,
        "Please dispose waste properly."
    )

    conf_pct = int(confidence * 100)

    message = (
        f"This is {class_name.lower()} waste "
        f"with {conf_pct} percent confidence. "
        f"{suggestion}"
    )

    print("🔊", message)
    speak(message)

    return message


# ── Suggestion System ─────────────────────────────────────────
def get_suggestion(class_name: str) -> dict:
    """Return icon, suggestion, action, and tip."""

    data = {
        "Recyclable": {
            "suggestion": SUGGESTIONS["Recyclable"],
            "icon": ICONS["Recyclable"],
            "color": "#2196F3",
            "action": "♻️ Drop in recycling bin",
            "tip": "Clean items before recycling."
        },
        "Organic": {
            "suggestion": SUGGESTIONS["Organic"],
            "icon": ICONS["Organic"],
            "color": "#4CAF50",
            "action": "🌿 Compost it",
            "tip": "Avoid plastic contamination."
        }
    }

    return data.get(class_name, {
        "suggestion": "Unknown waste type",
        "icon": "❓",
        "color": "#9E9E9E",
        "action": "Check manually",
        "tip": ""
    })