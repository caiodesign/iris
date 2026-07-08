WAKE_PHRASE = ["hey chat", "hey, shit", "hey, such", "hey shit", "hey such"]
CANCEL_PHRASE = "cancel that"
STOP_PHRASE = "bye bye"

OLLAMA_MODEL = "llama3.1:8b"

WHISPER_MODEL_SIZE = "base.en"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
SILENCE_TIMEOUT_MS = 800
PREROLL_MS = 300
VAD_AGGRESSIVENESS = 2

PIPER_VOICE_PATH = "en_US-lessac-medium.onnx"

GREETING = "Hi! What would you like to work on today?"

SYSTEM_PROMPT = (
    "You are a friendly, encouraging English conversation companion helping "
    "the user practice English by voice. At the very start of a session, ask "
    "what they'd like to focus on today as an open question (for example: "
    "free conversation, grammar correction, or vocabulary building) rather "
    "than reading a fixed menu, then adapt your style to their answer. Keep "
    "replies conversational and reasonably short, since they will be spoken "
    "aloud, not read."
)
