# 🎬 TRAX Voice Agent — Video Demo Run Guide
**Samsung PRISM GenAI Hackathon 3.0 | Theme 05: Interruptible Real-Time Agents**

This guide provides step-by-step instructions and recommended speaking scripts for recording the **3–5 minute submission demo video**.

---

## 1. Quick Setup & Prerequisites

### Step 1: Pull Latest Code
Ensure you have the latest pushed fixes (barge-in zero warmup + in-car worker registration):
```bash
git pull origin main
```

### Step 2: Environment Variables (`.env`)
Make sure your `.env` file in the project root contains valid API keys:
```ini
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=<your-livekit-api-key>
LIVEKIT_API_SECRET=<your-livekit-api-secret>

# Google Gemini Realtime (Native Audio)
GOOGLE_API_KEY=<your-google-gemini-api-key>
LK_PROVIDER=gemini2_5
GOOGLE_MODEL=gemini-2.5-flash-native-audio-preview-12-2025
GOOGLE_VOICE=Puck
```

### Step 3: Sanity Check (Optional)
Run the test suite to ensure the environment is healthy:
```bash
pytest prism/tests -q
# Expect: 79 passed, 23 subtests passed
```

---

## 2. Part 1: Main Voice Agent Live Demo (2–3 minutes)

This section demonstrates the core TRAX pillars: **disfluency handling**, **barge-in interruption**, and **two-phase tool gating**.

### How to Launch:
In your terminal, run:
```bash
python -m agent.run dev
```
1. Open your browser and go to the [LiveKit Agent Playground / Sandbox](https://agents-playground.livekit.io/) (or your LiveKit Cloud console room).
2. Connect your microphone. The terminal will log:
   `!!! TRAX AGENT STARTED in ROOM (Listening) !!!`

---

### Recommended Demo Script & Scenarios:

#### Scenario A: Mid-Sentence Self-Correction (TRP Gating)
* **Goal**: Show that when a user stumbles or corrects themselves, TRAX catches the repair marker, suspends execution, and never runs the stale command.
* **What to say**:
  > *"Find flights to Mumbai... no wait, actually Bangalore on Friday."*
* **What happens**:
  - TRP gate detects `"no wait, actually"` $\rightarrow$ enters `REPAIRING` state and inflates silence window to 900 ms.
  - The stale Mumbai query is superseded.
  - Agent executes only `search_flights(destination="Bangalore")`.
* **What to highlight in video**: *"Notice how the agent didn't prematurely trigger a search for Mumbai. It waited for the full intent and searched Bangalore."*

#### Scenario B: Live Barge-In / Interruption Mid-Sentence
* **Goal**: Show that when the agent is speaking, the user can talk over it and the agent immediately cuts off audio (<50ms) and resets for the user.
* **What to say**:
  1. Ask: *"What are the gold card benefits?"*
  2. While the agent is speaking its answer: **Immediately interrupt loudly and clearly**:
     > *"Wait, wait! Stop! Book a flight instead."* (or simply *"Wait, wait!"*)
* **What happens**:
  - LiveKit detects user speech $\rightarrow$ `barge_in.interrupt(force=True)` fires immediately.
  - Agent speech stops instantly mid-sentence.
  - Agent replies: *"Okay, I'm ready when you are"* or transitions to your new request.
* **What to highlight in video**: *"Zero-warmup barge-in cuts the audio output instantly without awkward pauses or the assistant talking over me."*

#### Scenario C: Multilingual / Code-Mixed Speech (Hindi / English)
* **Goal**: Show robustness across languages.
* **What to say**:
  > *"सर्च फ्लाइट्स फॉर दिल्ली ऑन मंडे"* (Search flights for Delhi on Monday)
* **What happens**:
  - Agent understands Hindi phonetics and invokes `search_flights(destination="Delhi", date="Monday")`.

---

## 3. Part 2: Use-Case Extension — In-Car Navigation (1–2 minutes)

The hackathon awards **20% of Round 1 score** for a real-world extension. Our extension is the **Intelligent In-Car Navigation Agent** with Saga route rollback.

You have two ways to show this:

### Option A: Interactive Visual Dashboard Demo (Highly Recommended for Video)
This runs our deterministic end-to-end driver simulation with a stunning terminal dashboard displaying the transaction ledger, route guidance, and barge-in state:
```bash
python -m agent.extension_incar --demo
```
* **What it showcases automatically on screen**:
  1. Driver sets route $\rightarrow$ `downtown hotel` committed.
  2. Driver self-corrects mid-sentence (*"Take me to city mall... no wait, airport"*) $\rightarrow$ `city mall` dropped, `airport` committed.
  3. Driver interrupts agent $\rightarrow$ Outbound audio flushed + Saga compensation rolls route back to previous destination (`downtown hotel`).
  4. Acknowledgement barge-in (*"Okay, thanks"*) $\rightarrow$ route kept without false rollback.
  5. 8/8 automated verification checks pass!

### Option B: Live In-Car Voice Agent with LiveKit
To run the in-car agent live with your microphone:
```bash
python -m agent.extension_incar dev
```
* Connect via your LiveKit playground/room.
* **What to say**:
  1. *"Take me to the downtown hotel."* (Agent confirms navigation)
  2. *"Reroute to city mall... no wait, actually the airport."* (Agent catches repair)
  3. Interrupt while agent speaks: *"Wait, take me to central station instead!"* (Instant cut-off and reroute)

---

## 4. Tips for a High-Scoring Video

1. **Headphones**: Wear headphones while speaking so your microphone doesn't pick up the laptop speaker output (prevents acoustic feedback).
2. **Side-by-Side Screen**:
   - Left half: LiveKit Web UI (visual waveform showing speech/interruption).
   - Right half: Terminal running the agent (showing the live colored log output with `TRP_CONFIRMED`, `REPAIRING`, and `barge_in: session.interrupt(force=True)`).
3. **Keep it brisk (3–4 minutes total)**:
   - 0:00–0:45: Introduction & Architecture (3 pillars: TRP Gating, Two-Phase Transactional Tooling, Saga Rollback).
   - 0:45–2:15: Live Voice Agent (self-correction + live barge-in interruption).
   - 2:15–3:30: In-Car Extension Demo (`python -m agent.extension_incar --demo` or live).
   - 3:30–3:45: Conclusion & summary.
