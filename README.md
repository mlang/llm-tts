# llm-tts

> Streaming playback for Text-to-Speech APIs and local models

Provides a subcommand `llm tts` as well as a `tts` tool.

Uses FFmpeg or GStreamer for real-time playback.

## Installation

If you have FFmpeg installed:

```bash
llm install git+https://github.com/mlang/llm-tts
```

If you prefer to use GStreamer:

```bash
sudo apt install cairo-1.0 gstreamer-1.0 girepository-2.0
llm install 'git+https://github.com/mlang/llm-tts#[gstreamer]'
```

## Usage

```bash
llm tts 'Hello, World!'
llm "A short poem" | llm tts --instructions poetic
```

By default, `llm-tts` will try to use FFmpeg for real-time playback.
If FFmpeg doesn’t work for you, try GStreamer instead:

```bash
llm tts --play gstreamer:alsasink "Advanced Linux Sound Architecture"
```

If you want the chat model to be able to pass instructions along to the tts model, you can do so with a schema specification and the `--json` flag:

```bash
llm --schema "text: Text to speak, instructions: How to speak the given text" --save tts
llm -t tts "A lovely poem" | llm tts --json


## Supported TTS back-ends

### OpenAI
• Model names: `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`  
• Requires `OPENAI_API_KEY`.

### ElevenLabs

* Model name: `eleven_multilingual_v2`  
* `pip install elevenlabs` and set `ELEVENLABS_API_KEY`.

### Hugging Face / transformers (local)

* Model names: `facebook/mms-tts-eng`, `suno/bark`, `suno/bark-small`  
* `pip install transformers sentencepiece`.

### Piper / Mimic3 (fully offline)

* Model names start with `piper/`, e.g. `piper/en_US-amy-low`  
* `pip install piper-tts` – the first run auto-downloads the voice file.

### OuteTTS

* Model names: `OuteTTS-1.0-0.6B`, `Llama-OuteTTS-1.0-1B`  
* `pip install outetts`.

### Silero (offline)

• Model names such as `silero/v3_en`, `silero/v4_ru`, …  
• `pip install torch` (models are fetched via `torch.hub`).

Run

```bash
llm tts --list-models
```

to see every identifier currently available.  
If you omit `--model`, the default is **gpt-4o-mini-tts**.

---

## Playback back-ends

### FFmpeg (default on Linux, macOS, Windows)

```bash
--play ffmpeg:FORMAT:DEVICE   # e.g. --play ffmpeg:alsa:hw:0
```

### GStreamer (default on other systems)

```bash
--play gstreamer[:SINK]       # list sinks with --list-sinks
```

You can also skip playback and write the result to a file instead:

```bash
llm tts --output-file out.mp3 "Save to disk instead of playing"
```
```
