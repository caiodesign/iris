# Thinking Room + No Stage Directions (v1.2.1) — Design

**Date:** 2026-07-08
**Status:** Approved by user
**Builds on:** v1.2 (`2026-07-08-memory-and-warmth-design.md`)

## Problems (from the user's first real v1.2 sessions)

1. **Pressure to talk fast:** the VAD treats 800 ms of silence as
   end-of-utterance, so thinking pauses send half-finished sentences to the
   model. Bad for a language learner.
2. **Spoken stage directions:** llama3.1 writes roleplay actions —
   "(laughs)", "(smiling)", "(pausing for a moment)" — despite the prompt
   asking for none, and the TTS reads them aloud verbatim.

## Decisions

### 1. Thinking pause: `SILENCE_TIMEOUT_MS` 800 → 2000

User chose 2 seconds (options were 1.5/2/3). Config-only; `VoiceDetector`
already consumes this constant. Accepted trade-off: the companion also
waits 2 s after the user finishes before responding.

### 2. Stage directions: two layers

- **Prompt hardening** (`SYSTEM_PROMPT`): replace the trailing "free of
  lists, markdown, emojis, and stage directions" clause with an explicit,
  example-bearing ban: never write actions, emotions, or sounds in
  parentheses or asterisks — no "(laughs)", "(smiling)", "*pauses*" — only
  the exact words to be spoken aloud.
- **Code stripping** (the guarantee): `LLMClient.send()` cleans each reply
  before storing and returning it:
  - remove every `(...)` and `*...*` span (regex `\([^)]*\)|\*[^*]*\*`),
  - collapse runs of whitespace left behind,
  - remove space before punctuation (`"amazing ." → "amazing."`),
  - strip leading/trailing whitespace.
  Cleaning happens BEFORE the reply enters `self.history`, so the model
  never sees its own stage directions and can't reinforce the habit.

**Accepted trade-off:** legitimate parenthetical content in replies is also
removed — in spoken conversation this is a non-loss.

## Testing

- New tests in `tests/test_llm_client.py`: send() strips stage directions
  (parentheses and asterisks, tidy punctuation/whitespace) and stores the
  cleaned reply in history; a clean reply passes through unchanged.
- Existing 28 tests keep passing.
- Acceptance: user has a session, pauses mid-sentence ~1.5 s without being
  cut off, and never hears "(laughs)".

## Out of scope

Adaptive/endpoint-aware silence detection, barge-in, everything on the v2
list.
