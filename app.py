# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import base64
import io
import re
import wave
import uuid
import time
import threading

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"

# Gemini TTS output is PCM, 24kHz, mono, 16-bit.
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2

# Approximately 2.5 minutes.
# This is an estimate because actual duration depends on speaking speed.
TARGET_CHARS_PER_PART = 15000

# Temporary generated jobs.
# Nothing is permanently saved to an output folder.
JOBS = {}

JOBS_LOCK = threading.Lock()

# ============================================================
# AVAILABLE VOICES
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
# HEALTH CHECK
# ============================================================

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "service": "AI Voice Generator",
        "tts": "Gemini 2.5 TTS"
    })


# ============================================================
# MODELS
# ============================================================

@app.route("/api/models", methods=["GET"])
def get_models():

    return jsonify({
        "models": [
            {
                "id": "gemini-2.5-flash-preview-tts",
                "name": "Gemini 2.5 Flash TTS"
            },
            {
                "id": "gemini-2.5-pro-preview-tts",
                "name": "Gemini 2.5 Pro TTS"
            }
        ]
    })


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text, max_chars=TARGET_CHARS_PER_PART):
    """
    Split long text into chunks without unnecessarily
    cutting sentences.

    Priority:
    1. Paragraph
    2. Sentence
    3. Word
    """

    text = text.strip()

    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current = ""

    def add_piece(piece):
        nonlocal current

        piece = piece.strip()

        if not piece:
            return

        if len(current) + len(piece) + 1 <= max_chars:
            if current:
                current += "\n\n" + piece
            else:
                current = piece

        else:
            if current:
                chunks.append(current.strip())

            current = piece

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) <= max_chars:
            add_piece(paragraph)
            continue

        # Paragraph is too large.
        # Split into sentences.
        sentences = re.split(
            r"(?<=[.!?।！？])\s+",
            paragraph
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) <= max_chars:
                add_piece(sentence)

            else:
                # Extremely long sentence.
                words = sentence.split()

                current_sentence = ""

                for word in words:

                    if (
                        len(current_sentence)
                        + len(word)
                        + 1
                        <= max_chars
                    ):
                        if current_sentence:
                            current_sentence += " " + word
                        else:
                            current_sentence = word

                    else:

                        if current_sentence:
                            add_piece(current_sentence)

                        current_sentence = word

                if current_sentence:
                    add_piece(current_sentence)

    if current:
        chunks.append(current.strip())

    return chunks


# ============================================================
# WAV CREATOR
# ============================================================

def pcm_to_wav(pcm_data):
    """
    Convert raw PCM audio returned by Gemini
    into a playable WAV byte stream.
    """

    output = io.BytesIO()

    with wave.open(output, "wb") as wav_file:

        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)

        wav_file.writeframes(pcm_data)

    return output.getvalue()


# ============================================================
# GEMINI TTS
# ============================================================

def generate_with_gemini(
    api_key,
    text,
    model,
    voice,
    temperature
):

    client = genai.Client(api_key=api_key)

    # Temperature is not a native TTS generation parameter
    # in the same way as standard text generation.
    #
    # We therefore use it to slightly modify the
    # performance instruction.

    if temperature <= 0.3:
        style = (
            "Speak calmly, steadily and naturally. "
            "Keep the delivery controlled and clear."
        )

    elif temperature <= 0.7:
        style = (
            "Speak naturally with a warm, expressive "
            "and smooth narration style."
        )

    else:
        style = (
            "Speak expressively and emotionally, "
            "with natural variation in pacing and tone."
        )

    prompt = f"""
You are a professional Bengali narrator.

{style}

Read the following text exactly as written.
Do not translate it.
Do not summarize it.
Do not add extra words.
Do not remove words.

TEXT:

{text}
"""

    interaction = client.interactions.create(
        model=model,
        input=prompt,
        response_format={
            "type": "audio"
        },
        generation_config={
            "speech_config": [
                {
                    "voice": voice
                }
            ]
        }
    )

    if not hasattr(interaction, "output_audio"):
        raise RuntimeError(
            "Gemini did not return audio."
        )

    audio_data = interaction.output_audio.data

    if isinstance(audio_data, str):
        pcm_data = base64.b64decode(audio_data)
    else:
        pcm_data = base64.b64decode(audio_data)

    return pcm_to_wav(pcm_data)


# ============================================================
# API KEY ERROR DETECTION
# ============================================================

def is_key_error(error):

    message = str(error).lower()

    key_errors = [
        "quota",
        "rate limit",
        "rate_limit",
        "resource exhausted",
        "too many requests",
        "429",
        "api key",
        "permission denied",
        "unauthorized",
        "invalid api key",
        "authentication",
        "403"
    ]

    return any(
        phrase in message
        for phrase in key_errors
    )


# ============================================================
# GENERATE AUDIO
# ============================================================

@app.route("/api/generate", methods=["POST"])
def generate_audio():

    try:

        data = request.get_json(silent=True) or {}

        api_keys = data.get("api_keys", [])
        text = data.get("text", "").strip()

        model = data.get(
            "model",
            DEFAULT_MODEL
        )

        voice = data.get(
            "voice",
            "Kore"
        )

        temperature = float(
            data.get(
                "temperature",
                0.7
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not api_keys:
            return jsonify({
                "error": "কমপক্ষে একটি Gemini API Key দিন।"
            }), 400

        api_keys = [
            key.strip()
            for key in api_keys
            if key and key.strip()
        ]

        if not api_keys:
            return jsonify({
                "error": "Valid API Key পাওয়া যায়নি।"
            }), 400

        if not text:
            return jsonify({
                "error": "Text input দিন।"
            }), 400

        if voice not in VOICES:
            return jsonify({
                "error": f"Unknown voice: {voice}"
            }), 400

        allowed_models = [
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts"
        ]

        if model not in allowed_models:
            return jsonify({
                "error": "Invalid Gemini TTS model."
            }), 400

        # ----------------------------------------------------
        # Split text
        # ----------------------------------------------------

        chunks = split_text(text)

        if not chunks:
            return jsonify({
                "error": "Text থেকে কোনো chunk তৈরি করা যায়নি।"
            }), 400

        job_id = str(uuid.uuid4())

        generated_parts = []

        # ----------------------------------------------------
        # Generate every chunk
        # ----------------------------------------------------

        for index, chunk in enumerate(chunks):

            audio_bytes = None
            successful_key = None
            last_error = None

            # Try API keys one by one.
            for key_index, api_key in enumerate(api_keys):

                try:

                    audio_bytes = generate_with_gemini(
                        api_key=api_key,
                        text=chunk,
                        model=model,
                        voice=voice,
                        temperature=temperature
                    )

                    successful_key = key_index + 1

                    break

                except Exception as error:

                    last_error = error

                    # If this key has quota/rate-limit/auth issue,
                    # automatically move to the next key.
                    if is_key_error(error):
                        continue

                    # Other error:
                    # still try next key so the app remains resilient.
                    continue

            if audio_bytes is None:

                return jsonify({
                    "error": (
                        f"Part {index + 1} generate করা যায়নি. "
                        f"শেষ error: {str(last_error)}"
                    )
                }), 500

            # ------------------------------------------------
            # Convert to Base64
            # ------------------------------------------------

            encoded_audio = base64.b64encode(
                audio_bytes
            ).decode("utf-8")

            generated_parts.append({
                "part": index + 1,
                "text": chunk,
                "audio_base64": encoded_audio,
                "mime_type": "audio/wav",
                "api_key_used": successful_key
            })

        # ----------------------------------------------------
        # Store temporarily in memory
        # ----------------------------------------------------

        with JOBS_LOCK:

            JOBS[job_id] = {
                "created_at": time.time(),
                "parts": generated_parts,
                "model": model,
                "voice": voice
            }

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "job_id": job_id,
            "total_parts": len(generated_parts),
            "parts": generated_parts
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# MERGE WAV FILES
# ============================================================

def merge_wav_files(wav_files):

    if not wav_files:
        raise ValueError("No audio files to merge.")

    output = io.BytesIO()

    with wave.open(output, "wb") as output_wav:

        output_wav.setnchannels(CHANNELS)
        output_wav.setsampwidth(SAMPLE_WIDTH)
        output_wav.setframerate(SAMPLE_RATE)

        for wav_bytes in wav_files:

            with wave.open(
                io.BytesIO(wav_bytes),
                "rb"
            ) as input_wav:

                frames = input_wav.readframes(
                    input_wav.getnframes()
                )

                output_wav.writeframes(frames)

    return output.getvalue()


# ============================================================
# MERGE ALL VOICES
# ============================================================

@app.route("/api/merge", methods=["POST"])
def merge_audio():

    try:

        data = request.get_json(silent=True) or {}

        job_id = data.get("job_id")

        if not job_id:
            return jsonify({
                "error": "job_id পাওয়া যায়নি।"
            }), 400

        with JOBS_LOCK:

            job = JOBS.get(job_id)

        if not job:
            return jsonify({
                "error": "Job পাওয়া যায়নি বা expired."
            }), 404

        wav_files = []

        for part in job["parts"]:

            encoded = part.get(
                "audio_base64"
            )

            if not encoded:
                continue

            wav_bytes = base64.b64decode(
                encoded
            )

            wav_files.append(wav_bytes)

        if not wav_files:
            return jsonify({
                "error": "Merge করার মতো audio নেই।"
            }), 400

        merged_audio = merge_wav_files(
            wav_files
        )

        encoded_merged = base64.b64encode(
            merged_audio
        ).decode("utf-8")

        return jsonify({
            "success": True,
            "mime_type": "audio/wav",
            "audio_base64": encoded_merged
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# DELETE TEMPORARY JOB
# ============================================================

@app.route("/api/job/<job_id>", methods=["DELETE"])
def delete_job(job_id):

    with JOBS_LOCK:

        if job_id in JOBS:
            del JOBS[job_id]

    return jsonify({
        "success": True
    })


# ============================================================
# AUTOMATIC CLEANUP
# ============================================================

def cleanup_old_jobs():

    while True:

        time.sleep(600)  # 10 minutes

        now = time.time()

        with JOBS_LOCK:

            expired_jobs = [
                job_id
                for job_id, job in JOBS.items()
                if now - job["created_at"] > 1800
            ]

            for job_id in expired_jobs:
                del JOBS[job_id]


cleanup_thread = threading.Thread(
    target=cleanup_old_jobs,
    daemon=True
)

cleanup_thread.start()


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
