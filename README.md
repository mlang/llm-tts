# llm-tts

> Streaming playback for the OpenAI Text-to-Speech API

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
llm --schema "input: Text to speak, instructions: How to speak the given text" --save tts
llm -t tts "A lovely poem" | llm tts --json
```
