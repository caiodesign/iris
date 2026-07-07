# English Voice Companion — Design Spec

## Purpose

A locally-run, hands-free voice companion for English practice. It listens continuously for a wake phrase, holds a spoken conversation to help the user practice English (conversation, grammar correction, or vocabulary — chosen by the assistant asking at the start of each session), and goes back to sleep on a stop phrase. Runs entirely on the user's own machine (RTX 4070 Super, 12GB VRAM / 32GB RAM) — no cloud dependency.

## Success Criteria (v1)

- User launches the app in a terminal window.
- Says "Hey Chat" → app wakes up and starts a spoken conversation.
- Assistant asks what the user wants to focus on today (open question, not a fixed menu) and adapts accordingly.
- User converses naturally by voice; assistant replies naturally by voice.
- Says "Bye Bye" → app ends the session and returns to sleep, still running and listening for "Hey Chat" again.
- User closes the terminal window to fully exit.

## Architecture

```
[Microphone] → [Voice Detector] → [Speech-to-Text] → [State Check] → [Llama 3.1 8B via Ollama] → [Text-to-Speech] → [Speakers]
```

Five components, each with one job:

1. **Voice Detector** — continuously monitors the mic and cheaply detects when speech starts/stops (no transcription, just "someone is talking now"). Triggers the STT step on each detected speech segment.
2. **Speech-to-Text (STT)** — transcribes a detected speech segment into text. Used both for wake/stop-phrase detection and for real conversation input. Runs locally, GPU-accelerated.
3. **State Check** — plain application logic (not AI) that inspects the transcribed text:
   - If state is **Asleep** and text contains "hey chat" → transition to **Active**, greet the user, ask what they want to focus on.
   - If state is **Active** and text contains "bye bye" → transition to **Asleep**, stop sending anything further to the LLM.
   - If state is **Active** and text is anything else → forward to the LLM.
   - If state is **Asleep** and text doesn't contain "hey chat" → discard, do nothing.
4. **LLM (Llama 3.1 8B via Ollama)** — generates the conversational reply. Runs locally.
5. **Text-to-Speech (TTS)** — converts the LLM's reply into audio and plays it through the speakers.

The application itself is the glue code wiring these five pieces together — no custom AI model training is required for v1.

## Conversation State Machine

Two states only in v1:

- **Asleep** — Voice Detector + STT active; only checks for "hey chat"; nothing reaches the LLM.
- **Active** — STT output goes to the LLM (unless it's "bye bye"); LLM replies are spoken via TTS.

No pause/interrupt state in v1 (see Out of Scope — barge-in).

## Control Words

| Phrase | Effect |
|---|---|
| "Hey Chat" | Wakes the app, starts a conversation (Asleep → Active) |
| "Bye Bye" | Ends the conversation, returns to sleep (Active → Asleep) |

Chosen to be distinctive multi-word phrases unlikely to occur naturally during English-practice conversation, reducing false triggers.

## Conversation Focus Selection

At the start of each Active session, the assistant asks an open-ended question ("What would you like to work on today?") rather than reading a fixed menu, and adapts its behavior based on the free-form answer. Supported focus areas for v1:

- Free conversation practice
- Real-time grammar/word correction
- Vocabulary building

**Explicitly out of scope for v1:** pronunciation feedback. This requires analyzing the raw audio signal (phoneme-level analysis), not just the transcribed text, which is a fundamentally different capability from the text-based flow above. Revisit as a standalone feature once the core loop works.

## Wake/Stop Detection — Chosen Approach and Alternatives

**Chosen for v1 — Option A: Voice Detector + STT phrase matching.** The Voice Detector flags speech segments; each segment is transcribed by the same local STT already needed for conversation; the State Check looks for "hey chat" / "bye bye" in the resulting text. Fully local, no extra accounts or custom model training, reuses infrastructure the app needs anyway. Tradeoff: introduces roughly half a second of latency between finishing the phrase and the app reacting, since it waits for transcription.

**Documented for future consideration — Option B: Dedicated wake-word engine.** A small, purpose-built always-on model trained to recognize one exact phrase (the mechanism behind "Hey Siri"-style assistants). Near-instant reaction, very low resource use. Would require a one-time setup step to train custom models for both "Hey Chat" and "Bye Bye" (typically via a free web-based tool), adding an extra moving part to the system. Worth revisiting if Option A's latency proves annoying in practice.

**Documented for future consideration — Option C: Push-to-talk.** Skip voice activation entirely; hold a key to talk. Trivial to build and immune to false triggers, but abandons the hands-free requirement that is the point of this app. Kept only as a fallback if Option A proves unreliable in real use.

## Interrupting the Assistant ("Barge-in")

**Out of scope for v1.** While the assistant is speaking (TTS is playing), the app does not listen for interruptions — the mic is effectively ignored until playback finishes, then it returns to the Active listening state. A "stop mid-sentence" word (candidate: "Enough") was discussed and deferred: implementing true barge-in requires the app to distinguish the user's live voice from the assistant's own voice coming out of the speakers (echo cancellation), which is meaningfully more complex than the rest of v1. Revisit once the core loop is proven.

## Memory (Documented Now, Built Later)

**Out of scope for v1 build**, but designed here so the architecture doesn't need to change shape later:

- When the user says "Bye Bye," before returning to Asleep, the app asks the LLM to summarize the session (topics covered, mistakes corrected, anything learned about the user) as a few bullet points, and appends them with a timestamp to a single `memory.md` file on disk.
- When the user next says "Hey Chat," the app reads `memory.md` and injects a condensed version into the LLM's context (e.g., via the system prompt), so the assistant is aware of prior sessions and can avoid re-covering the same ground.
- **Not yet designed:** how `memory.md` is pruned or summarized once it grows large after extended use. To be addressed when this feature is actually built.

## Tech Stack

- **Language:** Python (first-class local support for every tool below).
- **LLM serving:** Ollama, running `llama3.1:8b`.
- **Voice detection + Speech-to-Text:** faster-whisper — handles both short wake/stop-phrase snippets and full conversational transcription.
- **Text-to-Speech:** Piper — lightweight, fast, natural-sounding, CPU-based so it doesn't compete with the LLM for GPU/VRAM.
- **Runtime mode:** manually launched from a terminal window; runs until the window is closed. Not a background/tray app in v1.

## Error Handling (v1)

- **Ollama not running / unreachable:** app prints a clear error on startup and exits rather than failing silently mid-conversation.
- **No microphone / mic permission denied:** app detects this at startup and exits with a clear message.
- **STT produces empty/garbage transcription:** treated as "nothing said," discarded — no LLM call, no crash.
- **TTS playback failure:** logged, session continues (falls back to no audio for that turn) rather than crashing the whole app.

## Testing Approach

- **State machine logic** (Asleep/Active transitions, wake/stop phrase matching): unit-testable in isolation from audio — feed known transcript strings in, assert resulting state and whether the LLM would be called.
- **Full pipeline (audio in → audio out):** manual end-to-end testing, since real microphone/speaker behavior isn't practically unit-testable. Verify: wake phrase reliably wakes it, normal conversation flows, stop phrase reliably ends it, false triggers are rare during normal English-practice speech.

## Out of Scope for v1 (Summary)

- Pronunciation feedback (needs audio-level analysis, not just transcription)
- Barge-in / mid-sentence interruption ("Enough")
- Memory read/write (`memory.md`) — designed above, not built
- Dedicated wake-word engine (Option B)
- Push-to-talk fallback (Option C)
- Background/tray runtime mode (starts with Windows, always running)
