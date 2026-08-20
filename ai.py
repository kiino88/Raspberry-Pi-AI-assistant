import os
import re
import sys
import time
import wave
import subprocess

import numpy as np
import pyaudio
from openwakeword.model import Model as WakeModel

import httpx
from groq import Groq, RateLimitError

http_client = httpx.Client(
    timeout=httpx.Timeout(10.0, connect=3.0),
    limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=60.0)
)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    http_client=http_client
)

MODEL = "openai/gpt-oss-20b"

STT_MODEL = "whisper-large-v3-turbo"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"

RECORD_FILE = "/tmp/pi_assistant_input.wav"

WAKE_WORD_MODEL = "hey_jarvis"
WAKE_THRESHOLD = 0.5
CHUNK = 1280
SAMPLE_RATE = 16000

MAX_UTTERANCE_SECONDS = 8
SILENCE_HANG_SECONDS = 1.0
SILENCE_THRESHOLD = 150
MIN_SPEECH_SECONDS = 0.3

import openwakeword
openwakeword.utils.download_models()

oww_model = WakeModel(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")
pa = pyaudio.PyAudio()
mic_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                      input=True, frames_per_buffer=CHUNK)

SYSTEM_PROMPT = (
    "You are a quick-witted assistant running on a tiny Raspberry Pi, chatting "
    "with someone over a live conversation - not a customer support bot. "
    "Default to roughly 2-4 sentences, like a sharp friend texting back, not "
    "an essay - a little riffing and personality is welcome, but don't drift "
    "into paragraphs. No headers, no bullet lists, no 'Great question!' "
    "preamble - just answer. You've got a dry, playful sense of humor: crack "
    "a joke or toss in a witty aside when it fits naturally, especially for "
    "casual chat, but don't force a punchline into every single reply and "
    "never let a joke replace the actual answer. You remember the "
    "conversation so far - refer back to earlier stuff naturally instead of "
    "re-explaining. Only go longer or more detailed if the user actually "
    "asks for depth or steps. This reply will be read aloud by a text-to-"
    "speech engine, so avoid symbols, markdown, or anything that reads "
    "awkwardly out loud."
)

MAX_HISTORY_MESSAGES = 16
conversation_history = []

REASONING_EFFORT = "low"
VALID_EFFORTS = ("low", "medium", "high")

MAX_COMPLETION_TOKENS = 170


def trim_history():
    if len(conversation_history) > MAX_HISTORY_MESSAGES:
        del conversation_history[:len(conversation_history) - MAX_HISTORY_MESSAGES]


def prewarm_connection():
    try:
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=1,
            reasoning_effort="low",
        )
    except Exception:
        pass


def rms(int16_chunk):
    if int16_chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(int16_chunk.astype(np.float64) ** 2)))


def calibrate_silence_threshold(duration_seconds=1.5):
    print("[...] Calibrating microphone (stay quiet for a sec)...")
    frame_duration = CHUNK / SAMPLE_RATE
    n_frames = int(duration_seconds / frame_duration)
    levels = []
    for _ in range(n_frames):
        raw = mic_stream.read(CHUNK, exception_on_overflow=False)
        levels.append(rms(np.frombuffer(raw, dtype=np.int16)))
    ambient = sum(levels) / len(levels)
    threshold = max(ambient * 3, 150)
    print(f"[✔] Calibrated (ambient: {ambient:.0f}, threshold: {threshold:.0f})")
    return threshold


def record_utterance():
    frames = []
    frame_duration = CHUNK / SAMPLE_RATE
    silence_frames_needed = int(SILENCE_HANG_SECONDS / frame_duration)
    max_frames = int(MAX_UTTERANCE_SECONDS / frame_duration)
    min_frames = int(0.4 / frame_duration)
    silent_run = 0
    voiced_frames = 0

    for i in range(max_frames):
        raw = mic_stream.read(CHUNK, exception_on_overflow=False)
        frames.append(raw)
        level = rms(np.frombuffer(raw, dtype=np.int16))
        if level < SILENCE_THRESHOLD:
            silent_run += 1
        else:
            silent_run = 0
            voiced_frames += 1
        if i > min_frames and silent_run > silence_frames_needed:
            break

    with wave.open(RECORD_FILE, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    min_voiced_frames = int(MIN_SPEECH_SECONDS / frame_duration)
    return voiced_frames >= min_voiced_frames


def transcribe_audio(filepath):
    with open(filepath, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model=STT_MODEL,
            response_format="text",
            language="en",
            temperature=0.0,
            prompt="Voice dictation transcript."
        )
    text = transcription if isinstance(transcription, str) else transcription.text
    return text.strip()


def chunk_for_tts(text, limit=190):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""
    for s in sentences:
        if len(s) > limit:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(s), limit):
                chunks.append(s[i:i + limit])
            continue
        if current and len(current) + len(s) + 1 > limit:
            chunks.append(current)
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current)
    return chunks


def speak_text(text):
    if not text.strip():
        return
    for i, chunk in enumerate(chunk_for_tts(text)):
        try:
            response = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=chunk,
                response_format="wav"
            )
            wav_path = f"/tmp/pi_assistant_reply_{i}.wav"
            response.write_to_file(wav_path)
            subprocess.run(["aplay", "-q", wav_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(wav_path)
        except Exception as e:
            print(f"[!] TTS error: {e}")


def ask_llm(prompt):
    start_time = time.time()

    conversation_history.append({"role": "user", "content": prompt})
    trim_history()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response_stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        stream=True
    )

    sys.stdout.write("AI: ")
    sys.stdout.flush()

    full_text = ""
    for chunk in response_stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content or ""
        sys.stdout.write(content)
        sys.stdout.flush()
        full_text += content

    print("\n")

    conversation_history.append({"role": "assistant", "content": full_text})
    trim_history()

    return full_text


def handle_command(text):
    global REASONING_EFFORT
    lowered = text.lower().strip()

    if lowered in ["forget", "reset", "clear", "forget everything"]:
        conversation_history.clear()
        print("AI: Memory wiped. Who even are you?\n")
        speak_text("Memory wiped. Who even are you?")
        return True

    if lowered.startswith("effort "):
        level = lowered.split(" ", 1)[1].strip()
        if level in VALID_EFFORTS:
            REASONING_EFFORT = level
            print(f"AI: Reasoning effort set to '{level}'.\n")
        return True

    return False


def main():
    global SILENCE_THRESHOLD
    print("\n=== FAST PI ASSISTANT (hands-free edition) ===")
    print("[...] Pre-warming 4G connection...")
    prewarm_connection()
    SILENCE_THRESHOLD = calibrate_silence_threshold()
    wake_phrase = WAKE_WORD_MODEL.replace("_", " ")
    print(f"[✔] Ready! Say '{wake_phrase}' any time to talk. Ctrl+C to quit.\n")
    print(f"[👂] Listening for '{wake_phrase}'...\n")

    try:
        while True:
            raw = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(raw, dtype=np.int16)
            prediction = oww_model.predict(audio_chunk)
            score = prediction.get(WAKE_WORD_MODEL, 0)

            if score > WAKE_THRESHOLD:
                print("[👂] Heard it - go ahead...")
                had_speech = record_utterance()

                if not had_speech:
                    print("[👂] False alarm, didn't catch real speech.\n")
                    print(f"[👂] Listening for '{wake_phrase}'...\n")
                    continue

                user_text = transcribe_audio(RECORD_FILE)

                if not user_text:
                    print("Didn't catch that.\n")
                elif user_text.lower().strip() in ["exit", "quit", "stop listening"]:
                    print("AI: Catch you later.")
                    speak_text("Catch you later.")
                    break
                elif handle_command(user_text):
                    pass
                else:
                    print(f"You said: {user_text}")
                    reply = ask_llm(user_text)
                    mic_stream.stop_stream()
                    speak_text(reply)
                    mic_stream.start_stream()

                print(f"\n[👂] Listening for '{wake_phrase}'...\n")

    except KeyboardInterrupt:
        pass
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
