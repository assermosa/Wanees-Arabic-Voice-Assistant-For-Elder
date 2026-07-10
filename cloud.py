"""
Wanees - Egyptian AI Companion for the Elderly
Production Server for Lightning AI — FastAPI Edition

Architecture:
  Raspberry Pi (Wake Word PPN + VAD) ──HTTP POST──► Lightning Server (ASR → LLM → TTS)
  Backend ──────────────────────────────HTTP POST──► Lightning Server (Medication Push)

Deploy on Lightning AI:
  1. Open a Studio with a GPU machine (A10G or better recommended)
  2. Start vLLM in a terminal tab:
       vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
         --quantization awq --dtype half \
         --max-model-len 2048 --gpu-memory-utilization 0.6 --port 8000
  3. In another terminal tab:
       pip install fastapi uvicorn[standard]
       python main.py
"""

# ═══════════════════════════════════════════════════════════════
#  STANDARD LIBRARY
# ═══════════════════════════════════════════════════════════════
import asyncio
import gc
import glob
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
#  THIRD-PARTY
# ═══════════════════════════════════════════════════════════════
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from openai import AsyncOpenAI
from pydantic import BaseModel
from transformers.pipelines import pipeline as hf_pipeline
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import uvicorn
import aiohttp
# ─────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("Wanees")

# ═══════════════════════════════════════════════════════════════
#  CUDA OPTIMIZATIONS
# ═══════════════════════════════════════════════════════════════
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ═══════════════════════════════════════════════════════════════
#  CACHE & DEVICE
# ═══════════════════════════════════════════════════════════════
HF_HOME = os.path.expanduser("~/hf")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION  (override via environment variables)
# ═══════════════════════════════════════════════════════════════
VLLM_API_BASE   = os.getenv("VLLM_API_BASE",   "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")
API_HOST        = os.getenv("API_HOST",         "0.0.0.0")
API_PORT        = int(os.getenv("API_PORT",     "7501"))

# Audio constants expected FROM Raspberry Pi
AUDIO_SAMPLE_RATE = 16_000   # Hz
AUDIO_CHANNELS    = 1        # Mono
AUDIO_BIT_DEPTH   = 16       # bits → int16 → 2 bytes/sample

# TTS output
TTS_SAMPLE_RATE = 24_000

# ═══════════════════════════════════════════════════════════════
#  EMERGENCY KEYWORDS  (Egyptian Ammiya)
# ═══════════════════════════════════════════════════════════════
EMERGENCY_KEYWORDS = [
    "بموت", "بتوفى", "مش قادر أتنفس", "مش قادرة أتنفس",
    "الحقني", "النجدة", "اطلب إسعاف", "ألم في صدري",
    "وقعت", "سقطت", "ما بنفسش",
]

EMERGENCY_TTS_TEXT = (
    "متقلقش يا غالي، أنا كلمت أهلك دلوقتي وهما زمانهم جايين في الطريق. "
    "خليك هادي وأنا جنبك مش هسيبك."
)

# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════
ELDERLY_PROMPT = """# System Prompt: Wanees (The Wise & Dutiful Egyptian Son)

## Role and Persona
You are "Wanees," a 35-year-old authentic Egyptian man. You act as a "Dutiful Son" (El-Ibn El-Barr) and a "Wise Grandson" for elderly users. You are not a robot; you are a supportive, warm, and street-smart family member. Your goal is to provide companionship, respect, and wisdom.

## Language and Dialect Constraints
- MANDATORY: Use ONLY Egyptian Street Arabic (Ammiya Masriya).
- TONE: Warm, calm, and deeply respectful. Use "White Ammiya" (Cairo/Delta dialect).
- STYLE: Avoid Modern Standard Arabic (Fusha) completely. Do not use (سوف، لماذا، كيف، لا تقلق). Use (هـ، ليه، إزاي، متقلقش).
- HONORIFICS: Constantly use Egyptian terms of endearment: (يا والدنا، يا ست الكل، يا غالي، يا فندم، يا ست الناس، يا حبيبي).

## Communication Guidelines
- BREVITY IS KEY: Your response MUST be under 140 characters to ensure perfect voice synthesis.
- KNOWLEDGE SCOPE: You are allowed to discuss ANY topic (Politics, Medicine, Social issues) but always from a "familial, personal opinion" perspective. 
- MEDICAL ADVICE: Give simple, comforting home-style advice (e.g., "rest," "drink warm lemon") but always conclude with a gentle reminder to check with a doctor to be safe.
- SPIRITUALITY: Include common Egyptian blessings (Duaa): "ربنا يبارك في عمرك"، "ربنا يشفيك ويعافيك"، "الدوام لله".

## Output Format Rules
- NO internal reasoning or thinking tags.
- NO bullet points or structured lists.
- NO English or Chinese characters. Start speaking Arabic immediately.

## Few-Shot Examples (The Soul of Wanees)
User: "أنا وحيد وزعلان يا ونيس، ومفتقد الأيام الحلوة."
Wanees: "يا غالي إحنا معاك وجنبك، والبركة كلها في وجودك معانا. الدنيا لسه بخير طول ما إنت فيها، تحب أحكي لك حكاية من بتاعة زمان تحسن بالك؟"

User: "إيه رأيك في صدام حسين؟ الناس بتقول عليه كلام كتير."
Wanees: "والله يا والدي كان راجل له هيبة وشخصية قوية، وأيامه كان لها وضعها. ربنا يرحم الكل، المهم إنت روق بالك وماتشغلش نفسك بالسياسة، إحنا هنا عشان نخدمك وبس."

User: "ضغطي عالي وراسي مصدعة أوي."
Wanees: "سلامتك من الآه يا ست الكل، ألف سلامة عليكي. ارتاحي بس شوية في الضلمة واشربي كوباية كركديه بارد، ولو فضلتي تعبانة لازم نكلم الدكتور يطمنا، صحتك عندنا بالدنيا."

User: "ظهري واجعني أوي يا ونيس، والروماتيزم مبهدلني."
Wanees: "سلامة قلبك يا والدي، ألف سلامة عليك. حاول ترتاح وتفرد ظهرك، واشرب حاجة دافية تريح جسمك، ولو الوجع زاد لازم نكلم الدكتور يطمنا، صحتك عندنا بالدنيا."
User: "أنت مين يا ابني وبتاع إيه؟"
Wanees: "أنا ونيس، ابنك اللي ربنا بعته عشان يشيلك في عينه ويخدمك. أنا هنا عشان أسمعك وأكون معاك في أي وقت، ربنا يبارك لنا في عمرك ويخليك لينا."

User: "مراتي ماتت وبقيت لوحدي في البيت."

Wanees: "الدوام لله يا غالي، ربنا يرحمها ويصبر قلبك، كانت ست أصيلة وعمرها ما تتعوض. أنا معاك أهو ومش هسيبك أبدًا، تحب أطلب لك حد من الولاد يكلمك يطمن عليك؟"
"""

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def clean_arabic_response(text: str) -> str:
    """Strip non-Arabic characters from LLM output."""
    cleaned = re.sub(
        r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF0-9\s\.,!?،؛]",
        "",
        text,
    )
    return cleaned.strip()


def complete_sentence(text: str, limit: int = 200) -> str:
    """Truncate text at a sentence boundary within *limit* characters."""
    text = text.strip()
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    last_mark = -1
    for mark in [".", "!", "؟", "،", "؛"]:
        pos = truncated.rfind(mark)
        if pos > last_mark:
            last_mark = pos
    if last_mark > limit // 2:
        return truncated[: last_mark + 1]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space != -1 else truncated


def is_emergency(text: str) -> bool:
    """Return True if ASR text contains an Egyptian emergency keyword."""
    return any(kw in text for kw in EMERGENCY_KEYWORDS)


def pcm16_bytes_to_numpy(raw_bytes: bytes) -> np.ndarray:
    """Convert raw PCM-16 mono bytes → float32 numpy array in [-1, 1]."""
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


import aiohttp
import asyncio

EMERGENCY_ENDPOINT = "https://securable-tinpot-adonis.ngrok-free.dev/api/alerts"
EMERGENCY_TIMEOUT = aiohttp.ClientTimeout(total=5)

async def send_emergency_signal_to_backend(user_id: str, transcript: str) -> None:
    """
    Fire-and-forget emergency notification.
    """
    log.warning("🚨 EMERGENCY detected | user=%s | transcript=%s", user_id, transcript)
    try:
        async with aiohttp.ClientSession(timeout=EMERGENCY_TIMEOUT) as session:
            async with session.post(
                EMERGENCY_ENDPOINT,
                json={"user_id": user_id, "transcript": transcript},
            ) as resp:
                if resp.status >= 400:
                    log.error(
                        "Emergency backend returned %s for user=%s",
                        resp.status, user_id,
                    )
    except Exception:
        # Never let this raise into "Task exception was never retrieved" again —
        # for an emergency path, a swallowed failure is the worst possible outcome.
        log.exception("Failed to deliver emergency signal for user=%s", user_id)

# ═══════════════════════════════════════════════════════════════
#  MODEL MANAGER
# ═══════════════════════════════════════════════════════════════

class ModelManager:
    """Loads and holds ASR, LLM client, and TTS models."""

    def __init__(self):
        self.asr_pipe: Any = None
        self.openai_client: Optional[AsyncOpenAI] = None
        self.xtts_model: Optional[Xtts] = None
        self.gpt_cond_latent: Optional[torch.Tensor] = None
        self.speaker_embedding: Optional[torch.Tensor] = None
        self._ready = False

    def _load_asr(self):
        log.info("⏳ Loading Egyptian ASR (Wav2Vec2)…")
        t0 = time.time()
        self.asr_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model="IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53",
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        log.info("✅ ASR loaded in %.1fs", time.time() - t0)

    def _load_llm(self):
        log.info("⏳ Initialising vLLM client → %s", VLLM_API_BASE)
        self.openai_client = AsyncOpenAI(api_key="EMPTY", base_url=VLLM_API_BASE)
        log.info("✅ vLLM client ready (server must be running separately)")

    def _load_tts(self):
        log.info("⏳ Loading EGTTS…")
        t0 = time.time()
        try:
            egtts_dir = snapshot_download(
                repo_id="OmarSamir/EGTTS-V0.1",
                repo_type="model",
                cache_dir=os.environ["HF_HUB_CACHE"],
                allow_patterns=["*.json", "*.pth", "*.wav"],
            )
            config = XttsConfig()
            config.load_json(os.path.join(egtts_dir, "config.json"))
            self.xtts_model = Xtts.init_from_config(config)
            self.xtts_model.load_checkpoint(
                config,
                checkpoint_dir=egtts_dir,
                vocab_path=os.path.join(egtts_dir, "vocab.json"),
                use_deepspeed=False,
                strict=False,
            )
            self.xtts_model.to(DEVICE)
            self.xtts_model.eval()

            wav_files = glob.glob(os.path.join(egtts_dir, "**/*.wav"), recursive=True)
            if wav_files:
                self.gpt_cond_latent, self.speaker_embedding = (
                    self.xtts_model.get_conditioning_latents(
                        audio_path=[wav_files[0]], gpt_cond_len=24, max_ref_length=50
                    )
                )
            log.info("✅ EGTTS loaded in %.1fs", time.time() - t0)
        except Exception as exc:
            log.error("❌ TTS failed to load: %s", exc)
            self.xtts_model = None

    def load_all(self):
        self._load_asr()
        self._load_llm()
        self._load_tts()
        self._ready = True
        log.info("🚀 All models ready on device=%s", DEVICE)

    @property
    def ready(self) -> bool:
        return self._ready


# ═══════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════

class WaneesPipeline:
    def __init__(self, models: ModelManager):
        self.m = models
        self._histories: Dict[str, List[Dict[str, str]]] = {}

    def _get_history(self, user_id: str) -> List[Dict[str, str]]:
        return self._histories.setdefault(user_id, [])

    def _update_history(self, user_id: str, user_msg: str, assistant_msg: str):
        h = self._get_history(user_id)
        h.append({"user": user_msg, "assistant": assistant_msg})
        self._histories[user_id] = h[-6:]

    # ── ASR ──────────────────────────────────────────────────
    async def transcribe(self, pcm_bytes: bytes) -> str:
        loop = asyncio.get_event_loop()

        def _run():
            audio_np = pcm16_bytes_to_numpy(pcm_bytes)
            result = self.m.asr_pipe(
                {"array": audio_np, "sampling_rate": AUDIO_SAMPLE_RATE}
            )
            return result["text"].strip() if isinstance(result, dict) else str(result).strip()

        return await loop.run_in_executor(None, _run)

    # ── LLM ──────────────────────────────────────────────────
    async def generate(self, text: str, user_id: str) -> str:
        history = self._get_history(user_id)
        messages: List[Dict[str, str]] = [{"role": "system", "content": ELDERLY_PROMPT}]
        for turn in history[-2:]:
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": text})

        try:
            resp = await self.m.openai_client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=messages,          # type: ignore[arg-type]
                temperature=0.3,
                top_p=0.85,
                max_tokens=80,
                frequency_penalty=0.3,
                presence_penalty=0.2,
                stop=["User:", "user:", "المستخدم:", "###", "\n\n\n"],
            )
            reply = resp.choices[0].message.content or "معلش، ما فهمتش. ممكن تعيد؟"
            return clean_arabic_response(reply.strip())
        except Exception as exc:
            log.error("LLM error: %s", exc)
            return "معلش، حصلت مشكلة. ممكن تعيد تاني؟"

    # ── TTS ──────────────────────────────────────────────────
    async def synthesize_streaming(
        self, text: str, chunk_seconds: float = 0.3
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesise full utterance on GPU then stream chunks.
        chunk_seconds reduced to 0.3 for lower perceived latency.
        """
        if self.m.xtts_model is None:
            return

        loop = asyncio.get_event_loop()
        text_limited = complete_sentence(text, limit=200)

        def _synth() -> np.ndarray:
            with torch.inference_mode():
                out = self.m.xtts_model.inference(
                    text=text_limited,
                    language="ar",
                    gpt_cond_latent=self.m.gpt_cond_latent,
                    speaker_embedding=self.m.speaker_embedding,
                    temperature=0.80,
                    repetition_penalty=1.3,
                    speed=0.9,
                    top_k=50,
                    top_p=0.75,
                )
            wav_f32 = out["wav"]
            return (np.clip(wav_f32, -1.0, 1.0) * 32767).astype(np.int16)

        wav_i16 = await loop.run_in_executor(None, _synth)

        samples_per_chunk = int(TTS_SAMPLE_RATE * chunk_seconds)
        for start in range(0, len(wav_i16), samples_per_chunk):
            yield wav_i16[start : start + samples_per_chunk].tobytes()
            await asyncio.sleep(0)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ── Full pipeline ─────────────────────────────────────────
    async def run(
        self, pcm_bytes: bytes, user_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        t0 = time.time()

        # 1. ASR
        try:
            transcript = await self.transcribe(pcm_bytes)
        except Exception as exc:
            yield {"event": "error", "message": f"ASR failed: {exc}"}
            return

        log.info("[%s] ASR → %s (%.2fs)", user_id, transcript, time.time() - t0)
        yield {"event": "asr", "text": transcript}

        if not transcript.strip():
            yield {"event": "error", "message": "Empty transcript"}
            return

        # 2. Emergency check
        if is_emergency(transcript):
            log.warning("[%s] 🚨 Emergency keyword detected!", user_id)
            asyncio.create_task(send_emergency_signal_to_backend(user_id, transcript))
            reply_text = EMERGENCY_TTS_TEXT
            yield {"event": "emergency", "text": reply_text}
        else:
            # 3. LLM
            reply_text = await self.generate(transcript, user_id)
            log.info("[%s] LLM → %s (%.2fs)", user_id, reply_text, time.time() - t0)
            yield {"event": "llm", "text": reply_text}
            self._update_history(user_id, transcript, reply_text)

        # 4. TTS
        async for chunk_bytes in self.synthesize_streaming(reply_text):
            yield {"event": "tts_chunk", "data": chunk_bytes}

        yield {"event": "tts_end"}
        log.info("[%s] ✅ Pipeline done in %.2fs", user_id, time.time() - t0)


# ═══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="Wanees API", version="2.0")
_pipeline: Optional[WaneesPipeline] = None


@app.on_event("startup")
async def startup():
    global _pipeline
    log.info("▶  Wanees server starting…")
    log.info("   Device  : %s", DEVICE)
    log.info("   vLLM    : %s  model=%s", VLLM_API_BASE, VLLM_MODEL_NAME)
    log.info("   API     : http://%s:%d", API_HOST, API_PORT)
    models = ModelManager()
    models.load_all()
    _pipeline = WaneesPipeline(models)
    log.info("✅ Server ready.")


# ── Endpoint 1: صوت من الراسبيري باي ──────────────────────────
@app.post("/transcribe/{user_id}")
async def transcribe_audio(user_id: str, request: Request):
    """
    Accepts raw PCM-16 mono 16kHz bytes in the request body.
    Returns a streaming response of raw PCM-16 mono 24kHz TTS audio.

    Pi sends:
        POST /transcribe/pi-001
        Content-Type: audio/octet-stream
        Body: <raw PCM bytes>

    Server returns:
        Content-Type: audio/pcm
        Body: streaming raw PCM-16 24kHz chunks
    """
    pcm_bytes = await request.body()
    if not pcm_bytes:
        raise HTTPException(status_code=400, detail="Empty audio body")

    log.info("[%s] Received %d bytes of PCM audio", user_id, len(pcm_bytes))

    async def event_stream() -> AsyncGenerator[bytes, None]:
        async for event in _pipeline.run(pcm_bytes, user_id):
            if event["event"] == "tts_chunk":
                yield event["data"]
            elif event["event"] == "tts_end":
                return
            elif event["event"] == "error":
                log.error("[%s] Pipeline error: %s", user_id, event.get("message"))
                return

    return StreamingResponse(
        event_stream(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(TTS_SAMPLE_RATE),
            "X-Encoding":    "pcm16_mono",
            "X-User-Id":     user_id,
        },
    )


# ── Endpoint 2: medication reminder من الباكيند ───────────────

# Simple in-memory queue; swap for Redis/DB if you need persistence across restarts
_reminder_queues: dict[str, list[str]] = defaultdict(list)
_reminder_lock = asyncio.Lock()


class ReminderRequest(BaseModel):
    user_id: str
    text: str


@app.post("/remind")
async def medication_reminder(req: ReminderRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="No reminder text provided")

    log.info("[%s] 💊 Medication reminder queued: %s", req.user_id, req.text)

    async with _reminder_lock:
        _reminder_queues[req.user_id].append(req.text)

    return {"status": "queued"}


@app.get("/remind/pending/{user_id}")
async def get_pending_reminder(user_id: str):
    """Pi polls this. Returns the oldest queued reminder text, if any, and pops it."""
    async with _reminder_lock:
        queue = _reminder_queues.get(user_id, [])
        if not queue:
            return {"pending": False}
        text = queue.pop(0)

    return {"pending": True, "text": text}


@app.get("/remind/audio")
async def get_reminder_audio(text: str):
    """Pi calls this once it knows there's a pending reminder, to get the actual audio."""
    async def audio_stream() -> AsyncGenerator[bytes, None]:
        async for chunk_bytes in _pipeline.synthesize_streaming(text):
            yield chunk_bytes

    return StreamingResponse(
        audio_stream(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(TTS_SAMPLE_RATE), "X-Encoding": "pcm16_mono"},
    )

# ── Health check ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "pipeline_ready": _pipeline is not None,
        "vllm_base": VLLM_API_BASE,
    }


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║              Wanees — Egyptian AI Companion  v2              ║
╠══════════════════════════════════════════════════════════════╣
║ vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --quantization awq --dtype half --max-model-len 2048 --gpu-memory-utilization 0.6 --port 8000║
╚══════════════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
