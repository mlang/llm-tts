# llm-tts

A LLM plugin for text to speech using OpenAI APIs.

Provides a subcommand "llm tts" as well as a "tts" tool.

## Installation

```bash
sudo apt install cairo-1.0 gstreamer-1.0 girepository-2.0
llm install git+https://github.com/mlang/llm-tts
```

## Usage

```bash
llm tts 'Hello, World!'
llm "A short poem" | llm tts --instructions poetic
```
