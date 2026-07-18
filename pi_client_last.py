"""
pi_client_optimized.py
======================
Wanees AI — Raspberry Pi Client (Fixed Version + Reminder Polling)

Fix: Pre-buffer timeout split into two phases:
  - Phase A: wait up to 15s for FIRST chunk (XTTS synthesis delay)
  - Phase B: wait only 0.5s per chunk after first arrives

Addition: background thread polls the backend for pending medication
reminders and plays them through the same speaker, using a shared
lock so reminder audio never overlaps with conversation audio.

Wake word engine: openWakeWord (ONNX runtime), threshold=0.002
"""

import math
import queue
import struct
import threading
import time

import numpy as np
import pyaudio
from openwakeword.model import Model as OWWModel
import requests
import sounddevice as sd
import torch

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

SERVER_URL      = "https://7501-01khejztq1640gknejy92er4gq.cloudspaces.litng.ai/transcribe"
USER_ID         = "pi-living-room-001"

# openWakeWord config (replaces PPN_ACCESS_KEY / PPN_MODEL_PATH)
OWW_MODEL_PATH   = "wa-nees_en_raspberry-pi_v4_0_0.onnx"
OWW_THRESHOLD    = 0.002
OWW_INFERENCE_FW = "onnx"

# Audio
RATE             = 16_000
CHANNELS         = 1
TTS_SAMPLE_RATE  = 24_000

# Beep
BEEP_FREQUENCY   = 880
BEEP_DURATION    = 0.18
BEEP_VOLUME      = 0.6
BEEP_SAMPLE_RATE = 44_100
BEEP_FADE_MS     = 15

# VAD
VAD_THRESHOLD         = 0.5
VAD_MIN_SILENCE_MS    = 600
VAD_SPEECH_PAD_MS     = 100
PRE_SPEECH_BUFFER_SEC = 0.3
MAX_SPEECH_SEC        = 10.0
NO_SPEECH_TIMEOUT_SEC = 5.0

# Follow-up conversation
FOLLOWUP_WINDOW_SEC = 15.0

# HTTP
HTTP_TIMEOUT_CONNECT  = 10    # was 5 — give cloud more time to accept
HTTP_TIMEOUT_READ     = 60
HTTP_CHUNK_BYTES      = 4_096

# Audio player
SAMPLES_PER_FRAME  = 1024
BYTES_PER_FRAME    = SAMPLES_PER_FRAME * 2    # int16 = 2 bytes
QUEUE_MAXSIZE      = 50

# Pre-buffer — FIXED VALUES
PRE_BUFFER_CHUNKS        = 3
FIRST_CHUNK_TIMEOUT      = 20.0   # wait up to 20s for XTTS to synthesize first chunk
SUBSEQUENT_CHUNK_TIMEOUT = 0.5    # after first chunk arrives, 0.5s per chunk is plenty

# Medication reminder polling
REMINDER_POLL_INTERVAL_SEC = 15


# ═══════════════════════════════════════════════════════════════
#  BEEP GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_beep(
    frequency   : float = BEEP_FREQUENCY,
    duration    : float = BEEP_DURATION,
    volume      : float = BEEP_VOLUME,
    sample_rate : int   = BEEP_SAMPLE_RATE,
    fade_ms     : int   = BEEP_FADE_MS,
) -> np.ndarray:
    num_samples  = int(sample_rate * duration)
    t            = np.linspace(0, duration, num_samples, endpoint=False)
    wave         = np.sin(2 * math.pi * frequency * t).astype(np.float32)
    fade_samples = min(int(sample_rate * fade_ms / 1000), num_samples // 2)
    wave[:fade_samples]  *= np.linspace(0.0, 1.0, fade_samples)
    wave[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    wave *= volume
    return (wave * 32767).astype(np.int16)


BEEP_HIGH = generate_beep(frequency=880, duration=0.18)
BEEP_LOW  = generate_beep(frequency=440, duration=0.12)


def play_beep(audio: np.ndarray = None) -> None:
    try:
        target = audio if audio is not None else BEEP_HIGH
        sd.play(target, samplerate=BEEP_SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"⚠ Beep error: {e}")


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════

print("⏳ Loading openWakeWord wake word engine (ONNX)…")
oww_model = OWWModel(
    wakeword_models=[OWW_MODEL_PATH],
    inference_framework=OWW_INFERENCE_FW,
)
# openWakeWord's default chunk size is 1280 samples (80ms @ 16kHz).
# We keep this as the frame length used everywhere frame_length was
# previously referenced (VAD recording, mic reads, etc.).
FRAME_LEN = 1280
# Resolve the model's key name (basename without extension) so we can
# read its score out of the prediction dict below.
import os
OWW_MODEL_KEY = os.path.splitext(os.path.basename(OWW_MODEL_PATH))[0]
print(f"✅ openWakeWord ready  (frame_length={FRAME_LEN}, threshold={OWW_THRESHOLD})")

print("⏳ Loading Silero VAD…")
vad_model, vad_utils = torch.hub.load(
    "snakers4/silero-vad", "silero_vad", force_reload=False
)
(get_speech_timestamps, _, _, VADIterator, _) = vad_utils
vad_model.eval()

vad_iter = VADIterator(
    vad_model,
    threshold=VAD_THRESHOLD,
    sampling_rate=RATE,
    min_silence_duration_ms=VAD_MIN_SILENCE_MS,
    speech_pad_ms=VAD_SPEECH_PAD_MS,
)
print("✅ Silero VAD ready")

pa = pyaudio.PyAudio()

http_session = requests.Session()
http_session.headers.update({"Content-Type": "audio/octet-stream"})

# Reminder polling hits JSON endpoints, so use a separate session
# with default headers (don't force audio/octet-stream on GET requests)
reminder_session = requests.Session()

# Base URL for reminder endpoints — same host as SERVER_URL, different path
REMINDER_BASE_URL = SERVER_URL.rsplit("/transcribe", 1)[0]

# Shared lock: ensures reminder audio and conversation audio never play
# at the same time on the same speaker.
playback_lock = threading.Lock()

print("✅ All models loaded — ready\n")


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def int2float(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


# ═══════════════════════════════════════════════════════════════
#  VAD RECORDING
# ═══════════════════════════════════════════════════════════════

def record_speech_with_vad(
    stream,
    no_speech_timeout: float = NO_SPEECH_TIMEOUT_SEC,
) -> list:
    try:
        vad_iter.reset_states()
    except AttributeError:
        vad_model.reset_states()

    speech_frames  : list = []
    pre_buffer     : list = []
    speech_started  = False
    speech_ended    = False

    pre_buffer_max   = int(PRE_SPEECH_BUFFER_SEC * RATE / FRAME_LEN)
    max_frames       = int(MAX_SPEECH_SEC         * RATE / FRAME_LEN)
    no_speech_frames = int(no_speech_timeout      * RATE / FRAME_LEN)
    total_frames     = 0

    while True:
        raw    = stream.read(FRAME_LEN, exception_on_overflow=False)
        f32    = int2float(raw)
        tensor = torch.from_numpy(f32).unsqueeze(0)
        total_frames += 1

        with torch.no_grad():
            vad_out = vad_iter(tensor, return_seconds=False)

        if not speech_started:
            pre_buffer.append(raw)
            if len(pre_buffer) > pre_buffer_max:
                pre_buffer.pop(0)

        if vad_out and "start" in vad_out:
            speech_started = True
            speech_frames.extend(pre_buffer)
            pre_buffer = []
            print("🗣  Speech started…")

        if speech_started and not speech_ended:
            speech_frames.append(raw)

        if vad_out and "end" in vad_out:
            speech_ended = True
            print("🔇 Speech ended.")
            break

        if len(speech_frames) >= max_frames:
            print(f"⏱ Max duration ({MAX_SPEECH_SEC}s) reached.")
            break

        if not speech_started and total_frames >= no_speech_frames:
            return []

    return speech_frames


# ═══════════════════════════════════════════════════════════════
#  STREAMING TTS PLAYBACK  — FIXED
# ═══════════════════════════════════════════════════════════════

def stream_audio_to_server(pcm_frames: list) -> bool:
    full_pcm = b"".join(pcm_frames)
    duration  = len(full_pcm) / (RATE * 2)
    print(f"📤 Sending {len(full_pcm)/1024:.1f} KB  ({duration:.1f}s)…")

    audio_queue : queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    got_audio   = threading.Event()
    player_done = threading.Event()
    error_event = threading.Event()

    # ─────────────────────────────────────────────────────────
    #  Audio Player Thread — FIXED pre-buffer logic
    # ─────────────────────────────────────────────────────────
    def audio_player():
        print("⏳ Waiting for server to synthesize audio…")

        # ── Phase A: wait for FIRST chunk ─────────────────────
        # XTTS needs 5-10s to synthesize before any bytes flow.
        # We wait up to FIRST_CHUNK_TIMEOUT seconds here.
        try:
            first_chunk = audio_queue.get(timeout=FIRST_CHUNK_TIMEOUT)
        except queue.Empty:
            print("⚠ No audio received from server — synthesis may have failed.")
            player_done.set()
            return

        if first_chunk is None:
            print("⚠ No audio received from server.")
            player_done.set()
            return

        print(f"✅ First chunk arrived ({len(first_chunk)} bytes) — buffering…")

        # ── Phase B: collect remaining pre-buffer chunks ───────
        # Network is flowing now, so 0.5s per chunk is enough.
        pre_buffer_data = first_chunk
        chunks_collected = 1

        while chunks_collected < PRE_BUFFER_CHUNKS:
            try:
                chunk = audio_queue.get(timeout=SUBSEQUENT_CHUNK_TIMEOUT)
            except queue.Empty:
                break   # fewer chunks than target is fine — start playing

            if chunk is None:
                audio_queue.put(None)   # put sentinel back for drain loop
                break

            pre_buffer_data  += chunk
            chunks_collected += 1

        print(f"🔊 Pre-buffered {len(pre_buffer_data)/1024:.1f} KB "
              f"({chunks_collected} chunks) — starting playback…")

        # ── Phase C: open PyAudio output stream ───────────────
        # Acquire the shared lock so a medication reminder can't start
        # playing over this conversation's audio.
        with playback_lock:
            pa_out = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=TTS_SAMPLE_RATE,
                output=True,
                frames_per_buffer=SAMPLES_PER_FRAME,
            )

            leftover = b""

            def write_aligned(data: bytes) -> bytes:
                """Write only complete frames; return leftover bytes."""
                combined   = leftover + data
                n_complete = (len(combined) // BYTES_PER_FRAME) * BYTES_PER_FRAME
                if n_complete > 0:
                    pa_out.write(combined[:n_complete])
                return combined[n_complete:]

            try:
                # Play pre-buffered data first
                leftover = write_aligned(pre_buffer_data)

                # Drain the queue until sentinel or timeout
                while True:
                    try:
                        chunk = audio_queue.get(timeout=2.0)
                    except queue.Empty:
                        if got_audio.is_set():
                            break
                        continue

                    if chunk is None:
                        break

                    leftover = write_aligned(chunk)

                # Flush remaining bytes (pad to full frame with silence)
                if leftover:
                    pad = BYTES_PER_FRAME - (len(leftover) % BYTES_PER_FRAME)
                    if pad != BYTES_PER_FRAME:
                        leftover += b"\x00" * pad
                    pa_out.write(leftover)

            except Exception as e:
                print(f"❌ Player error: {e}")
                error_event.set()
            finally:
                pa_out.stop_stream()
                pa_out.close()
                player_done.set()

    # ─────────────────────────────────────────────────────────
    #  HTTP Reader  (main thread)
    # ─────────────────────────────────────────────────────────
    player_thread = threading.Thread(target=audio_player, daemon=True)
    player_thread.start()

    success = False
    try:
        resp = http_session.post(
            f"{SERVER_URL}/{USER_ID}",
            data=full_pcm,
            stream=True,
            timeout=(HTTP_TIMEOUT_CONNECT, HTTP_TIMEOUT_READ),
        )
        resp.raise_for_status()

        chunk_count = 0
        for chunk in resp.iter_content(chunk_size=HTTP_CHUNK_BYTES):
            if not chunk:
                continue
            chunk_count += 1
            if chunk_count == 1:
                print(f"📦 First HTTP chunk received ({len(chunk)} bytes)")
            audio_queue.put(chunk)   # blocks if queue full (backpressure)

        success = chunk_count > 0
        if not success:
            print("⚠ Server returned 200 but sent 0 bytes of audio.")

    except requests.exceptions.Timeout:
        print("❌ Server timeout — XTTS may still be loading.\n")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot reach server — check SERVER_URL.\n")
    except Exception as e:
        print(f"❌ HTTP error: {e}\n")
    finally:
        got_audio.set()
        audio_queue.put(None)   # sentinel to unblock player

    player_done.wait(timeout=30.0)

    if success and not error_event.is_set():
        print("✅ TTS playback complete.\n")

    return success and not error_event.is_set()


# ═══════════════════════════════════════════════════════════════
#  MEDICATION REMINDER POLLING
# ═══════════════════════════════════════════════════════════════

def play_pcm_stream(resp) -> None:
    """Blocking playback of a streamed PCM response (used for reminders)."""
    pa_out = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=TTS_SAMPLE_RATE,
        output=True,
        frames_per_buffer=SAMPLES_PER_FRAME,
    )
    try:
        for chunk in resp.iter_content(chunk_size=HTTP_CHUNK_BYTES):
            if chunk:
                pa_out.write(chunk)
    finally:
        pa_out.stop_stream()
        pa_out.close()


def reminder_poll_loop():
    """
    Background loop: periodically asks the backend if a medication
    reminder is pending for this device, fetches its audio, and plays
    it. Uses playback_lock so it never talks over an active conversation.
    """
    print(f"💊 Reminder polling started (every {REMINDER_POLL_INTERVAL_SEC}s)")

    while True:
        try:
            r = reminder_session.get(
                f"{REMINDER_BASE_URL}/remind/pending/{USER_ID}",
                timeout=(HTTP_TIMEOUT_CONNECT, HTTP_TIMEOUT_READ),
            )
            r.raise_for_status()
            data = r.json()

            if data.get("pending"):
                text = data["text"]
                print(f"💊 Reminder pending: {text}")

                audio_resp = reminder_session.get(
                    f"{REMINDER_BASE_URL}/remind/audio",
                    params={"text": text},
                    stream=True,
                    timeout=(HTTP_TIMEOUT_CONNECT, HTTP_TIMEOUT_READ),
                )
                audio_resp.raise_for_status()

                # Wait for any in-progress conversation audio to finish
                # before playing the reminder.
                with playback_lock:
                    play_beep(BEEP_HIGH)
                    play_pcm_stream(audio_resp)

        except requests.exceptions.Timeout:
            print("⚠ Reminder poll timeout — will retry.")
        except requests.exceptions.ConnectionError:
            print("⚠ Reminder poll: cannot reach server — will retry.")
        except Exception as e:
            print(f"⚠ Reminder poll error: {e}")

        time.sleep(REMINDER_POLL_INTERVAL_SEC)


# ═══════════════════════════════════════════════════════════════
#  FOLLOW-UP WINDOW
# ═══════════════════════════════════════════════════════════════

def followup_listen(stream) -> list:
    print(f"💬 Follow-up window open ({FOLLOWUP_WINDOW_SEC}s) — speak freely…\n")

    frames = record_speech_with_vad(
        stream,
        no_speech_timeout=FOLLOWUP_WINDOW_SEC,
    )

    if not frames:
        play_beep(BEEP_LOW)
        print("⏱ Follow-up window expired — returning to wake word.\n")

    return frames


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

mic_stream = pa.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAME_LEN,
)

print("👂 Listening for 'يا ونيس'…\n")

# Start the reminder polling thread in the background, independent
# of the wake-word loop below.
threading.Thread(target=reminder_poll_loop, daemon=True).start()

try:
    while True:

        # ══════════════════════════════════════════════════════
        #  PHASE 1 — Wake Word Detection (openWakeWord / ONNX)
        # ══════════════════════════════════════════════════════
        raw       = mic_stream.read(FRAME_LEN, exception_on_overflow=False)
        pcm_frame = np.frombuffer(raw, dtype=np.int16)

        prediction = oww_model.predict(pcm_frame)
        score      = prediction.get(OWW_MODEL_KEY, 0.0)

        if score < OWW_THRESHOLD:
            continue

        print(f"🔔 Wake word detected! (score={score:.4f})")

        beep_thread = threading.Thread(target=play_beep, daemon=True)
        beep_thread.start()
        beep_thread.join()

        # Reset openWakeWord's internal buffers so residual audio from
        # the trigger doesn't cause an immediate re-trigger later.
        oww_model.reset()

        # ══════════════════════════════════════════════════════
        #  PHASE 2 — First Utterance
        # ══════════════════════════════════════════════════════
        speech_frames = record_speech_with_vad(mic_stream)

        if not speech_frames:
            print("⚠ No speech captured — back to wake word.\n")
            continue

        got_response = stream_audio_to_server(speech_frames)

        # ══════════════════════════════════════════════════════
        #  PHASE 3 — Follow-up Conversation Loop
        # ══════════════════════════════════════════════════════
        if got_response:
            while True:
                followup_frames = followup_listen(mic_stream)

                if not followup_frames:
                    break

                print("🔄 Follow-up detected — sending…")
                play_beep(BEEP_HIGH)

                got_response = stream_audio_to_server(followup_frames)

                if not got_response:
                    print("⚠ No server response — closing conversation.\n")
                    break

        print("👂 Back to wake word detection…\n")

except KeyboardInterrupt:
    print("\n\n--- Stopped by user ---")

finally:
    mic_stream.stop_stream()
    mic_stream.close()
    pa.terminate()
    http_session.close()
    reminder_session.close()
    print("Cleanup done. Goodbye!")
