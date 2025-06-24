from contextlib import contextmanager
import json
from sys import platform, stdin
from typing import cast, Literal, Optional

from click import argument, command, echo, option, Choice, ParamType, UsageError
from dataclasses import dataclass
from subprocess import Popen, PIPE
from llm import get_key, hookimpl
from openai import OpenAI


@hookimpl
def register_commands(cli):
    cli.add_command(tts_cmd)


@hookimpl
def register_tools(register):
    register(tts)


@dataclass(frozen=True)
class AudioFormat:
    gst_caps: str
    ffmpeg_args: list[str]


AUDIO_FORMATS: dict[str, AudioFormat] = {
    "mp3": AudioFormat(
        gst_caps="audio/mpeg, mpegversion=1",
        ffmpeg_args=["-f", "mp3"],
    ),
    "opus": AudioFormat(
        gst_caps="audio/x-opus",
        ffmpeg_args=["-f", "opus"],
    ),
    "aac": AudioFormat(
        gst_caps="audio/mpeg, mpegversion=4",
        ffmpeg_args=["-f", "aac"],
    ),
    "flac": AudioFormat(
        gst_caps="audio/x-flac",
        ffmpeg_args=["-f", "flac"],
    ),
    "wav": AudioFormat(
        gst_caps="audio/x-wav",
        ffmpeg_args=["-f", "wav"],
    ),
    # Raw 24 kHz, 16-bit, mono PCM
    "pcm": AudioFormat(
        gst_caps="audio/x-raw, format=S16LE, channels=1, rate=24000, layout=interleaved",
        ffmpeg_args=["-f", "s16le", "-ar", "24000", "-ac", "1"],
    )
}


@dataclass
class FFmpeg:
    out_fmt: str
    device: str

@dataclass
class GStreamer:
    sink: str

PlayerCfg = FFmpeg | GStreamer

def default_player() -> PlayerCfg:
    if platform.startswith("linux"):
        return FFmpeg(out_fmt="alsa", device="default")
    if platform == "darwin":
        return FFmpeg(out_fmt="avfoundation", device=":0")
    if platform.startswith(("win32", "cygwin")):
        return FFmpeg(out_fmt="dshow", device='audio="default"')

    return GStreamer(sink="autoaudiosink")


class PlayerSpec(ParamType):
    name = "PLAYER"

    def convert(self, value, param, ctx) -> PlayerCfg:
        backend, *parts = value.split(sep=":", maxsplit=2)

        if backend == 'gstreamer':
            if len(parts) > 1:
                self.fail("Use gstreamer[:SINK]", param, ctx)
            sink = parts[0] if len(parts) == 1 else "autoaudiosink"
            return GStreamer(sink=sink)

        if backend == 'ffmpeg':
            if len(parts) != 2:
                self.fail("Use ffmpeg:FORMAT:DEVICE", param, ctx)
            out_fmt, device = parts
            return FFmpeg(out_fmt=out_fmt, device=device)

        self.fail("Backend must be 'gstreamer' or 'ffmpeg'", param, ctx)


@command()
@argument("text", required=False)
@option('-m', '--model', show_default=True, metavar="NAME")
@option('-v', '--voice', metavar="NAME")
@option('-i', '--instructions')
@option('-f', '--format', 'fmt',
    type=Choice(AUDIO_FORMATS.keys()), show_choices=True,
    default='aac', show_default=True
)
@option('json_mode', '--json', is_flag=True,
    help="Read the initial TTS request from stdin"
)
@option('--play', type=PlayerSpec(),
    metavar="gstreamer[:SINK] | ffmpeg:FORMAT:DEVICE",
    help="Playback backend. Examples: gstreamer, gstreamer:pulsesink, ffmpeg:alsa:hw:0"
)
@option("--list-sinks", is_flag=True,
    help="List available GStreamer audio sinks and exit"
)
@option('-o', '--output-file', metavar="FILE",
    help="Don't play, write to file"
)
@option('--api-key', metavar="SECRET", help='OpenAI API key')
def tts_cmd(
    text: Optional[str],
    model: str,
    voice: str,
    instructions: Optional[str],
    fmt: str,
    json_mode: bool,
    api_key: Optional[str],
    output_file: Optional[str],
    play: Optional[PlayerCfg],
    list_sinks: bool
):
    """Text to speech"""

    # Handle --list-sinks option
    if list_sinks:
        echo("Available GStreamer audio sinks:")
        for name in sorted(audio_sinks()):
            echo(f"  {name}")
        return

    if play and output_file:
        raise UsageError("--play and --output-file are mutually exclusive")

    if play is None and output_file is None:
        play = default_player()

    openai = OpenAI(api_key=get_key(api_key, 'openai', 'OPENAI_API_KEY'))

    tts_request = json.loads(stdin.read()) if json_mode else {}
    if not text and not json_mode:
        text = stdin.read()

    if text:
        tts_request['input'] = text
    if model or 'model' not in tts_request:
        tts_request['model'] = model or DEFAULT_TTS_MODEL
    if voice or 'voice' not in tts_request:
        tts_request['voice'] = voice or DEFAULT_TTS_VOICE
    if instructions:
        tts_request['instructions'] = instructions
    tts_request['response_format'] = fmt

    with openai.audio.speech.with_streaming_response.create(
        **tts_request
    ) as response:
        if output_file:
            response.stream_to_file(output_file)
        else:
            with player(cast(PlayerCfg, play), fmt) as pipe:
                for chunk in response.iter_bytes(CHUNK_SIZE):
                    pipe.write(chunk)


Voice = Literal[
    'alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx',
    'sage', 'shimmer', 'verse'
]


def tts(
    text: str,
    voice: Voice = "nova",
    instructions: Optional[str] = None
):
    """Convert the given text to speech and play it back for the user.  Instructions lets you change the way the text is spoken."""

    openai = OpenAI(api_key=get_key(None, 'openai', 'OPENAI_API_KEY'))
    response_format: Literal['aac'] = 'aac'

    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts", voice=voice, input=text,
        response_format=response_format,
        **({"instructions": instructions} if instructions else {})
    ) as response:
        with player(default_player(), response_format) as pipe:
            for chunk in response.iter_bytes(CHUNK_SIZE):
                pipe.write(chunk)

    return "OK"


@contextmanager
def gstreamer_pipeline(
    fmt: str, sink: str = "alsasink"
):
    import gi # type: ignore
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst # type: ignore
    Gst.init(None)

    class Pipeline:
        def __init__(self, appsrc):
            self._appsrc = appsrc

        def write(self, data: bytes) -> int:
            self._appsrc.emit('push-buffer', Gst.Buffer.new_wrapped(data))
            return len(data)

    SOURCE = 'appsrc name=audio_source'
    DECODE = 'decodebin ! audioconvert ! audioresample'
    pipeline = Gst.parse_launch(f'{SOURCE} ! {DECODE} ! {sink}')
    appsrc = pipeline.get_by_name('audio_source')
    appsrc.set_property(
        'caps',
        Gst.Caps.from_string(AUDIO_FORMATS[fmt].gst_caps)
    )
    appsrc.set_property('block', True)
    appsrc.set_property('max-bytes', 2097152)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        yield Pipeline(appsrc)

        appsrc.emit('end-of-stream')
        msg = pipeline.get_bus().timed_pop_filtered(Gst.CLOCK_TIME_NONE,
            Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"ERROR: {err}, {debug}")

    finally:
        pipeline.set_state(Gst.State.NULL)


@contextmanager
def ffmpeg_pipeline(
    fmt: str = "aac",
    *,
    out_fmt: str,
    device: str
):
    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        *AUDIO_FORMATS[fmt].ffmpeg_args, "-i", "pipe:0",
        "-f", out_fmt, device
    ]
    proc = Popen(cmd, stdin=PIPE)

    try:
        yield proc.stdin

        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    finally:
        if proc.poll() is None:
            proc.kill()


# Dispatcher returning the correct playback context-manager
def player(cfg: PlayerCfg, input_fmt: str):
    if isinstance(cfg, GStreamer):
        return gstreamer_pipeline(fmt=input_fmt, sink=cfg.sink)
    else:
        return ffmpeg_pipeline(
            fmt=input_fmt, out_fmt=cfg.out_fmt, device=cfg.device
        )


def audio_sinks():
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst

    Gst.init(None)
    registry = Gst.Registry.get()
    factories = registry.get_feature_list(Gst.ElementFactory)

    return (
        factory.get_name()
        for factory in factories
        if {"Audio", "Sink"}.issubset(factory.get_metadata('klass').split("/"))
    )


CHUNK_SIZE = 4096
DEFAULT_TTS_MODEL = 'gpt-4o-mini-tts'
DEFAULT_TTS_VOICE = 'nova'
