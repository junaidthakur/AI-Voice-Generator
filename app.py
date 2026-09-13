from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

import base64
import io
import re
import time
import uuid
import threading
import wave


# ============================================================
# APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"

SUPPORTED_MODELS = {
    "gemini-2.5-flash-preview-tts": "Gemini 2.5 Flash TTS",
    "gemini-2.5-pro-preview-tts": "Gemini 2.5 Pro TTS",
}

LANGUAGE_CODE = "bn-IN"

# Gemini TTS returns 24kHz PCM audio.
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2

# Approximate chunk size.
# Actual duration depends on speaking speed.
TARGET_CHARS_PER_PART = 14000

# Temporary jobs kept in RAM.
# Nothing is permanently written to an output folder.
JOBS = {}

JOBS_LOCK = threading.Lock()

JOB_EXPIRY_SECONDS = 30 * 60


# ============================================================
# GEMINI PREBUILT VOICES
# ============================================================

VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "service": "AI Voice Generator",
        "provider": "Google Gemini",
        "tts": "Gemini 2.5 TTS"
    })


# ============================================================
# MODELS
# ============================================================

@app.route("/api/models", methods=["GET"])
def get_models():

    models = []

    for model_id, model_name in SUPPORTED_MODELS.items():

        models.append({
            "id": model_id,
            "name": model_name
        })

    return jsonify({
        "success": True,
        "models": models,
        "voices": VOICES,
        "language": LANGUAGE_CODE
    })


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "status": "healthy"
    })


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Maximum 2 empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text, max_chars=TARGET_CHARS_PER_PART):

    text = clean_text(text)

    if not text:
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    chunks = []
    current = ""

    def push_piece(piece):

        nonlocal current

        piece = piece.strip()

        if not piece:
            return

        if not current:

            current = piece
            return

        combined = current + "\n\n" + piece

        if len(combined) <= max_chars:

            current = combined

        else:

            chunks.append(current.strip())
            current = piece

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Paragraph already fits
        if len(paragraph) <= max_chars:

            push_piece(paragraph)
            continue

        # Large paragraph:
        # split by sentences
        sentences = re.split(
            r"(?<=[.!?।！？])\s+",
            paragraph
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) <= max_chars:

                push_piece(sentence)

            else:

                # Extremely long sentence.
                # Split by words as a last resort.
                words = sentence.split()

                word_buffer = ""

                for word in words:

                    if not word_buffer:

                        word_buffer = word

                    elif len(
                        word_buffer + " " + word
                    ) <= max_chars:

                        word_buffer += " " + word

                    else:

                        push_piece(word_buffer)

                        word_buffer = word

                if word_buffer:

                    push_piece(word_buffer)

    if current:

        chunks.append(current.strip())

    return chunks


# ============================================================
# PCM -> WAV
# ============================================================

def pcm_to_wav(pcm_data):

    output = io.BytesIO()

    with wave.open(output, "wb") as wav:

        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)

        wav.writeframes(pcm_data)

    return output.getvalue()


# ============================================================
# WAV MERGE
# ============================================================

def merge_wav_files(wav_files):

    if not wav_files:

        raise ValueError(
            "No audio files to merge."
        )

    output = io.BytesIO()

    with wave.open(output, "wb") as output_wav:

        output_wav.setnchannels(CHANNELS)
        output_wav.setsampwidth(SAMPLE_WIDTH)
        output_wav.setframerate(SAMPLE_RATE)

        for wav_bytes in wav_files:

            input_buffer = io.BytesIO(
                wav_bytes
            )

            with wave.open(
                input_buffer,
                "rb"
            ) as input_wav:

                frames = input_wav.readframes(
                    input_wav.getnframes()
                )

                output_wav.writeframes(frames)

    return output.getvalue()


# ============================================================
# TEMPERATURE -> STYLE
# ============================================================

def temperature_instruction(temperature):

    if temperature <= 0.3:

        return (
            "Speak calmly, slowly and clearly. "
            "Keep the delivery controlled and peaceful."
        )

    if temperature <= 0.7:

        return (
            "Speak naturally with a warm, smooth "
            "and expressive Bengali narration style."
        )

    return (
        "Speak with expressive emotion, natural variation "
        "in tone and pacing, while keeping the Bengali "
        "pronunciation clear and natural."
    )


# ============================================================
# GEMINI TTS
# ============================================================

def generate_gemini_audio(
    api_key,
    text,
    model,
    voice,
    temperature
):

    client = genai.Client(
        api_key=api_key
    )

    style = temperature_instruction(
        temperature
    )

    prompt = f"""
Read the following Bengali text aloud.

{style}

Important instructions:

- Speak in natural Bengali.
- Do not translate the text.
- Do not summarize the text.
- Do not add explanations.
- Do not add extra words.
- Do not remove words.
- Preserve the meaning and wording.
- Use natural pauses at punctuation.
- Make the narration sound human and suitable for a professional voice-over.

TEXT:

{text}
"""

    response = client.models.generate_content(

        model=model,

        contents=prompt,

        config=types.GenerateContentConfig(

            response_modalities=["AUDIO"],

            speech_config=types.SpeechConfig(

                language_code=LANGUAGE_CODE,

                voice_config=types.VoiceConfig(

                    prebuilt_voice_config=(
                        types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                )
            )
        )
    )

    # --------------------------------------------------------
    # Extract audio
    # --------------------------------------------------------

    try:

        audio_data = (
            response
            .candidates[0]
            .content
            .parts[0]
            .inline_data
            .data
        )

    except Exception as error:

        raise RuntimeError(
            "Gemini returned no audio data."
        ) from error

    if not audio_data:

        raise RuntimeError(
            "Empty audio response from Gemini."
        )

    # Usually bytes.
    # Handle Base64 string as fallback.
    if isinstance(audio_data, str):

        audio_data = base64.b64decode(
            audio_data
        )

    return pcm_to_wav(
        audio_data
    )


# ============================================================
# API KEY ERROR
# ============================================================

def should_try_next_key(error):

    message = str(error).lower()

    key_error_patterns = [

        "429",
        "quota",
        "rate limit",
        "rate_limit",
        "resource exhausted",
        "too many requests",

        "api key",
        "invalid api key",
        "invalid_argument",

        "permission denied",
        "unauthorized",
        "authentication",

        "403"
    ]

    return any(
        pattern in message
        for pattern in key_error_patterns
    )


# ============================================================
# GENERATE
# ============================================================

@app.route("/api/generate", methods=["POST"])
def generate():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        api_keys = data.get(
            "api_keys",
            []
        )

        text = data.get(
            "text",
            ""
        ).strip()

        model = data.get(
            "model",
            DEFAULT_MODEL
        )

        voice = data.get(
            "voice",
            "Kore"
        )

        temperature = data.get(
            "temperature",
            0.7
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not isinstance(
            api_keys,
            list
        ):

            return jsonify({
                "success": False,
                "error": "API keys must be a list."
            }), 400

        api_keys = [
            str(key).strip()
            for key in api_keys
            if str(key).strip()
        ]

        if not api_keys:

            return jsonify({
                "success": False,
                "error": "কমপক্ষে একটি Gemini API Key দিন।"
            }), 400

        if not text:

            return jsonify({
                "success": False,
                "error": "Text input খালি।"
            }), 400

        if model not in SUPPORTED_MODELS:

            return jsonify({
                "success": False,
                "error": "Invalid Gemini TTS model."
            }), 400

        if voice not in VOICES:

            return jsonify({
                "success": False,
                "error": "Invalid voice."
            }), 400

        try:

            temperature = float(
                temperature
            )

        except Exception:

            temperature = 0.7

        temperature = max(
            0.0,
            min(1.0, temperature)
        )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        chunks = split_text(
            text
        )

        if not chunks:

            return jsonify({
                "success": False,
                "error": "Text থেকে কোনো অংশ তৈরি করা যায়নি।"
            }), 400

        # ----------------------------------------------------
        # Create job
        # ----------------------------------------------------

        job_id = str(
            uuid.uuid4()
        )

        job = {

            "job_id": job_id,

            "created_at": time.time(),

            "status": "generating",

            "total_parts": len(chunks),

            "completed_parts": 0,

            "parts": [],

            "model": model,

            "voice": voice
        }

        with JOBS_LOCK:

            JOBS[job_id] = job

        generated_parts = []

        # ----------------------------------------------------
        # Generate chunks
        # ----------------------------------------------------

        for index, chunk in enumerate(chunks):

            audio_bytes = None

            used_key = None

            errors = []

            # ------------------------------------------------
            # Try every API key
            # ------------------------------------------------

            for key_index, api_key in enumerate(
                api_keys
            ):

                try:

                    audio_bytes = (
                        generate_gemini_audio(

                            api_key=api_key,

                            text=chunk,

                            model=model,

                            voice=voice,

                            temperature=temperature
                        )
                    )

                    used_key = (
                        key_index + 1
                    )

                    break

                except Exception as error:

                    errors.append({
                        "key": key_index + 1,
                        "error": str(error)
                    })

                    # Try next key.
                    continue

            # ------------------------------------------------
            # All keys failed
            # ------------------------------------------------

            if audio_bytes is None:

                with JOBS_LOCK:

                    if job_id in JOBS:

                        JOBS[job_id][
                            "status"
                        ] = "failed"

                return jsonify({

                    "success": False,

                    "job_id": job_id,

                    "error": (
                        f"Part {index + 1} generate "
                        f"করা যায়নি।"
                    ),

                    "details": errors

                }), 500

            # ------------------------------------------------
            # Base64
            # ------------------------------------------------

            audio_base64 = base64.b64encode(
                audio_bytes
            ).decode("utf-8")

            part = {

                "part": index + 1,

                "text": chunk,

                "audio_base64": audio_base64,

                "mime_type": "audio/wav",

                "api_key_used": used_key
            }

            generated_parts.append(
                part
            )

            # ------------------------------------------------
            # Update progress
            # ------------------------------------------------

            with JOBS_LOCK:

                if job_id in JOBS:

                    JOBS[job_id][
                        "completed_parts"
                    ] = index + 1

                    JOBS[job_id][
                        "parts"
                    ] = generated_parts

        # ----------------------------------------------------
        # Finished
        # ----------------------------------------------------

        with JOBS_LOCK:

            if job_id in JOBS:

                JOBS[job_id][
                    "status"
                ] = "completed"

        return jsonify({

            "success": True,

            "job_id": job_id,

            "total_parts": len(
                generated_parts
            ),

            "parts": generated_parts

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# PROGRESS
# ============================================================

@app.route(
    "/api/progress/<job_id>",
    methods=["GET"]
)
def progress(job_id):

    with JOBS_LOCK:

        job = JOBS.get(
            job_id
        )

    if not job:

        return jsonify({

            "success": False,

            "error": "Job পাওয়া যায়নি।"

        }), 404

    return jsonify({

        "success": True,

        "job_id": job_id,

        "status": job["status"],

        "total_parts": job[
            "total_parts"
        ],

        "completed_parts": job[
            "completed_parts"
        ]

    })


# ============================================================
# MERGE
# ============================================================

@app.route(
    "/api/merge",
    methods=["POST"]
)
def merge():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        job_id = data.get(
            "job_id"
        )

        if not job_id:

            return jsonify({

                "success": False,

                "error": "job_id পাওয়া যায়নি।"

            }), 400

        with JOBS_LOCK:

            job = JOBS.get(
                job_id
            )

        if not job:

            return jsonify({

                "success": False,

                "error": (
                    "Job পাওয়া যায়নি "
                    "অথবা expired."
                )

            }), 404

        if job["status"] != "completed":

            return jsonify({

                "success": False,

                "error": (
                    "Generation এখনো শেষ হয়নি।"
                )

            }), 400

        wav_files = []

        for part in job["parts"]:

            encoded = part.get(
                "audio_base64"
            )

            if not encoded:
                continue

            wav_bytes = (
                base64.b64decode(
                    encoded
                )
            )

            wav_files.append(
                wav_bytes
            )

        if not wav_files:

            return jsonify({

                "success": False,

                "error": (
                    "Merge করার মতো audio নেই।"
                )

            }), 400

        merged_audio = merge_wav_files(
            wav_files
        )

        merged_base64 = base64.b64encode(
            merged_audio
        ).decode("utf-8")

        return jsonify({

            "success": True,

            "job_id": job_id,

            "mime_type": "audio/wav",

            "audio_base64": merged_base64

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# DELETE JOB
# ============================================================

@app.route(
    "/api/job/<job_id>",
    methods=["DELETE"]
)
def delete_job(job_id):

    with JOBS_LOCK:

        existed = (
            job_id in JOBS
        )

        JOBS.pop(
            job_id,
            None
        )

    return jsonify({

        "success": True,

        "deleted": existed

    })


# ============================================================
# CLEANUP OLD JOBS
# ============================================================

def cleanup_jobs():

    while True:

        time.sleep(300)

        now = time.time()

        with JOBS_LOCK:

            expired = []

            for job_id, job in JOBS.items():

                age = (
                    now
                    - job["created_at"]
                )

                if age > JOB_EXPIRY_SECONDS:

                    expired.append(
                        job_id
                    )

            for job_id in expired:

                JOBS.pop(
                    job_id,
                    None
                )


cleanup_thread = threading.Thread(
    target=cleanup_jobs,
    daemon=True
)

cleanup_thread.start()


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
    )
