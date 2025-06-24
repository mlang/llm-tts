from contextlib import contextmanager
from sys import stdin
from typing import Literal, Optional

from click import argument, command, option
import gi
from llm import get_key, hookimpl
from openai import OpenAI


@hookimpl
def register_commands(cli):
    cli.add_command(tts_cmd)


@hookimpl
def register_tools(register):
    register(tts)


@command()
@argument("input", required=False)
@option('-m', '--model', default='gpt-4o-mini-tts', show_default=True,
    metavar="NAME"
)
@option('-v', '--voice', metavar="NAME",
    default="nova", show_default=True
)
@option('-i', '--instructions')
@option('-f', '--format', metavar="FORMAT", default='aac', show_default=True)
@option("-s", "--sink", default="alsasink", show_default=True)
@option('-o', '--output-file', metavar="FILE",
    help="Don't play, write to file"
)
@option('--api-key', metavar="SECRET", help='OpenAI API key')
def tts_cmd(
    input: Optional[str],
    model: str,
    voice: str,
    instructions: Optional[str],
    format: str,
    api_key: Optional[str],
    output_file: Optional[str],
    sink: str
):
    """Text to speech"""

    openai = OpenAI(api_key=get_key(api_key, 'openai', 'OPENAI_API_KEY'))

    if not input: input = stdin.read()

    with openai.audio.speech.with_streaming_response.create(
        model=model, voice=voice, input=input,
        response_format=format,
        **({"instructions": instructions} if instructions else {})
    ) as response:
        if output_file:
            response.stream_to_file(output_file)
        else:
            with gstreamer_pipeline(format=format, sink=sink) as pipe:
                for chunk in response.iter_bytes(CHUNK_SIZE):
                    pipe.push_bytes(chunk)


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
    format = 'aac'

    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts", voice=voice, input=text,
        response_format=format,
        **({"instructions": instructions} if instructions else {})
    ) as response:
        with gstreamer_pipeline(format=format) as pipe:
            for chunk in response.iter_bytes(CHUNK_SIZE):
                pipe.push_bytes(chunk)

    return "OK"


@contextmanager
def gstreamer_pipeline(format: str, sink: str = "alsasink"):
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)

    class Pipeline:
        def __init__(self, appsrc):
            self._appsrc = appsrc

        def push_bytes(self, data: bytes) -> None:
            self._appsrc.emit('push-buffer', Gst.Buffer.new_wrapped(data))

    SOURCE = 'appsrc name=audio_source'
    DECODE = 'decodebin ! audioconvert ! audioresample'
    pipeline = Gst.parse_launch(f'{SOURCE} ! {DECODE} ! {sink}')
    appsrc = pipeline.get_by_name('audio_source')
    appsrc.set_property('caps', Gst.Caps.from_string(CAPS[format]))
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


CHUNK_SIZE = 4096


CAPS = {
    'mp3': 'audio/mpeg, mpegversion=1',
    'opus': 'audio/x-opus',
    'aac': 'audio/mpeg, mpegversion=4',
    'flac': 'audio/x-flac',
    'wav': 'audio/x-wav',
    'pcm': 'audio/x-raw, format=S16LE, channels=1, rate=24000, layout=interleaved'
}
