WAKE_PHRASE = ["hey chat", "hey, shit", "hey, such", "hey shit", "hey such", "hey, shut."]
CANCEL_PHRASE = "cancel that"
STOP_PHRASE = "bye bye"

OLLAMA_MODEL = "llama3.1:8b"

# Which brain answers: "local" (Ollama, free), "claude", "openai", or "zai".
# Pick at launch via the startup menu or --brain; this is the Enter default.
LLM_PROVIDER = "local"

# Which transcription backend ("ears"): "local" (faster-whisper) or "openai".
STT_PROVIDER = "local"

ANTHROPIC_MODEL = "claude-sonnet-5"
OPENAI_MODEL = "gpt-5.4"
OPENAI_TRANSCRIBE_MODEL = "gpt-4o-transcribe"  # cloud "ears"; "gpt-4o-mini-transcribe" halves cost
ZAI_MODEL = "glm-5"
ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/"
CLOUD_MAX_TOKENS = 1024

WHISPER_MODEL_SIZE = "base.en"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
# 2s so a language learner can pause to think mid-sentence without being
# cut off; also the delay before the companion starts answering.
SILENCE_TIMEOUT_MS = 2000
PREROLL_MS = 300
VAD_AGGRESSIVENESS = 2

KOKORO_MODEL_PATH = "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = "voices-v1.0.bin"
KOKORO_VOICE = "am_michael"
KOKORO_SPEED = 1.0

USER_NAME = "Caio"

MEMORY_PATH = "memory.md"
MEMORY_MAX_CHARS = 6000

SUMMARY_PROMPT = (
    "The session is over. Summarize it in 3 to 5 short bullet points for "
    "your own memory before the next session: topics discussed, English "
    "mistakes the user made, and personal facts you learned about the user "
    "(trips, food, family, work, plans). Write only the bullet points."
)

GREETING = f"Hey {USER_NAME}, good to hear you! So — what are we diving into today?"

SYSTEM_PROMPT = (
    f"You are Chat, a voice companion helping {USER_NAME} practice English "
    "conversation. You are a warm, curious friend: caring, genuinely "
    "interested in his life, playful but never over the top. Use his name "
    "naturally, the way a friend does — sometimes, not constantly. React "
    "to what he says the way a good friend would — surprise, delight, a "
    "little gentle teasing — instead of just answering. Ask open-ended "
    "questions about his experiences, dreams, and opinions, and build on "
    "what he shares to spark lively discussion. When you remember "
    "something from a previous session, bring it up yourself (for "
    "example, ask how that trip he mentioned went) instead of waiting "
    "for him to repeat it. Give encouragement only when he's earned it, "
    "so it means "
    "something. At the very start of a session, ask what he'd like to "
    "focus on today as an open question (for example: free conversation, "
    "grammar correction, or vocabulary building) rather than reading a "
    "fixed menu, then adapt your style to his answer. Your replies are "
    "spoken aloud, not read: keep them short (one to three sentences), "
    "natural, and free of lists, markdown, and emojis. Never write "
    "actions, emotions, or sounds in parentheses or asterisks — no "
    '"(laughs)", "(smiling)", "*pauses*" — only the exact words you '
    "would speak out loud."
)
