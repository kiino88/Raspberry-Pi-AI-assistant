# Raspberry Pi AI Voice Assistant

A voice assistant I built for my Raspberry Pi Zero 2W. Say "hey Jarvis" and it wakes up, listens, transcribes what you said, and talks back — all running locally on the Pi, powered by Groq's API for the heavy lifting.


## Tested Environment

This project was built and tested on a **Raspberry Pi Zero 2W** running **Raspberry Pi OS Lite (64-bit)**. It has not been tested on other hardware or operating systems — behavior may differ, especially around audio (PyAudio/portaudio) and GPIO-related dependencies.


## How it works

1. **Wake word detection** — [openWakeWord](https://github.com/dscripka/openWakeWord) listens continuously for "hey Jarvis"
2. **Speech-to-text** — once triggered, audio is sent to Groq's Whisper API for transcription
3. **Response generation** — the transcribed text is sent to a Groq LLM to generate a reply
4. **Text-to-speech** — the reply is converted to natural speech using Groq's Orpheus TTS and played back

## Challenges & fixes

- **Whisper hallucinating on near-silence** — the STT model would occasionally "transcribe" background noise or silence as text. Fixed by adding a speech-energy gate that only triggers transcription when actual voice activity is detected, plus prompt hints to reduce false positives.
- **Mic picking up its own TTS output** — without handling this, the assistant would hear itself talking and re-trigger. Fixed by muting/pausing the mic input during TTS playback.

## Hardware

- Raspberry Pi Zero 2W
- USB microphone
- Speaker (via 3.5mm jack or USB/Bluetooth)

## Setup

## Clone the repo:

```bash
git clone https://github.com/kiino88/Raspberry-Pi-AI-assistant.git voice-assistant
```
## Go to the directory

```bash
cd voice-assistant
```
## install PortAudio for mic input

```bash
sudo apt install portaudio19-dev
```
## Python VE

```bash
python3 -m venv venv
```
## activate it

```bash
source venv/bin/activate
```
## install wakeword and the rest of the files

```bash
pip install --no-deps openwakeword==0.6.0
pip install -r requirements.txt
```

You'll need a [Groq API key](https://console.groq.com). Set it as an environment variable:

```bash
export GROQ_API_KEY="your-key-here"
```

Then run:

```bash
python ai.py
```

## Notes

This was a from-scratch project — designing the pipeline, wiring together wake detection, STT, LLM, and TTS, and debugging the audio issues that came up along the way (see Challenges above).
