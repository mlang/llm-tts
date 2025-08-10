from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
from subprocess import Popen, PIPE
from sys import platform, stdin
from types import TracebackType
from typing import cast, Callable, ContextManager, IO, Iterator, Literal, Optional, Protocol, Type, ClassVar

from click import argument, command, echo, option, ParamType, UsageError
from llm import get_key, hookimpl
from openai import OpenAI, NOT_GIVEN


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

    # Dispatcher returning the correct playback context-manager
    def play(self, cfg: "PlayerCfg") -> ContextManager[IO[bytes]]:
        if isinstance(cfg, GStreamer):
            return gstreamer_pipeline(self.gst_caps, sink=cfg.sink)
        else:
            return ffmpeg_pipeline(
                self.ffmpeg_args, out_fmt=cfg.out_fmt, device=cfg.device
            )

    @classmethod
    def mp3(cls) -> "AudioFormat":
        return cls("audio/mpeg, mpegversion=1", ["-f", "mp3"])

    @classmethod
    def opus(cls) -> "AudioFormat":
        return cls("audio/x-opus", ["-f", "opus"])

    @classmethod
    def aac(cls) -> "AudioFormat":
        return cls("audio/mpeg, mpegversion=4", ["-f", "aac"])

    @classmethod
    def flac(cls) -> "AudioFormat":
        return cls("audio/x-flac", ["-f", "flac"])

    @classmethod
    def wav(cls) -> "AudioFormat":
        return cls("audio/x-wav", ["-f", "wav"])

    @classmethod
    def pcm(cls, sr: int) -> "AudioFormat":
        return cls(
            f"audio/x-raw, format=S16LE, channels=1, rate={sr}, layout=interleaved",
            ["-f", "s16le", "-ar", str(sr), "-ac", "1"],
        )

    @classmethod
    def alaw(cls) -> "AudioFormat":
        return cls("audio/x-alaw, rate=8000, channels=1", ["-f", "alaw", "-ar", "8000", "-ac", "1"])

    @classmethod
    def ulaw(cls) -> "AudioFormat":
        return cls("audio/x-mulaw, rate=8000, channels=1", ["-f", "mulaw", "-ar", "8000", "-ac", "1"])


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
@option('-f', '--format', 'fmt')
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
@option("--list-models", is_flag=True,
    help="List available TTS models and exit"
)
@option('-o', '--output-file', metavar="FILE",
    help="Don't play, write to file"
)
@option('--api-key', metavar="SECRET", help='OpenAI API key')
def tts_cmd(
    text: Optional[str],
    model: str,
    voice: Optional[str],
    instructions: Optional[str],
    fmt: Optional[str],
    json_mode: bool,
    api_key: Optional[str],
    output_file: Optional[str],
    play: Optional[PlayerCfg],
    list_sinks: bool,
    list_models: bool
):
    """Text to speech"""

    # Handle --list-models option
    if list_models:
        echo("Available TTS models:")
        for name in sorted(_models.keys()):
            echo(f"  {name}")
        return

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

    backend = cast(PlayerCfg, play)

    tts_model = get_tts_model(model)

    if fmt is None:
        fmt = tts_model.preferred_audio_format

    if fmt not in tts_model.audio_formats:
        raise UsageError(f"Audio format {fmt} not supported by {model}.  Supported formats are: {', '.join(tts_model.audio_formats.keys())}")

    tts_request = json.loads(stdin.read()) if json_mode else {}
    if not text and not json_mode:
        text = stdin.read()

    if text:
        tts_request['text'] = text
    if 'model' in tts_request:
        model = tts_request['model']
        del tts_request['model']
    if model is None:
        model = DEFAULT_TTS_MODEL
    if voice or 'voice' not in tts_request:
        tts_request['voice'] = voice or tts_model.default_voice
    if instructions:
        tts_request['instructions'] = instructions
    tts_request['response_format'] = fmt

    with tts_model.synthesize(**tts_request) as audio:
        if output_file:
            audio.stream_to_file(output_file)
        else:
            with tts_model.audio_formats[fmt].play(backend) as pipe:
                for chunk in audio.iter_bytes(CHUNK_SIZE):
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

    tts_model = get_tts_model(DEFAULT_TTS_MODEL)
    response_format = tts_model.preferred_audio_format

    with tts_model.synthesize(
        text=text, voice=voice, instructions=instructions,
        response_format=response_format
    ) as response:
        fmt_spec = tts_model.audio_formats[response_format]
        with tts_model.audio_formats[response_format].play(default_player()) as pipe:
            for chunk in response.iter_bytes(CHUNK_SIZE):
                pipe.write(chunk)

    return "OK"


@contextmanager
def gstreamer_pipeline(
    gst_caps: str, *, sink: str = "alsasink"
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
        Gst.Caps.from_string(gst_caps)
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
    ffmpeg_args: list[str],
    *,
    out_fmt: str,
    device: str
):
    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        *ffmpeg_args, "-i", "pipe:0",
        "-f", out_fmt, device
    ]
    proc = Popen(cmd, stdin=PIPE, bufsize=0)

    try:
        yield proc.stdin

        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    finally:
        if proc.poll() is None:
            proc.kill()


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


class StreamingAudio(Protocol):
    def stream_to_file(self, filename: str):
        ...

    def iter_bytes(self, chunk_size: int = 4096) -> Iterator[bytes]:
        ...


class TextToSpeechModel(ABC):
    key_name: ClassVar[Optional[str]] = None
    key_envvar: ClassVar[Optional[str]] = None
    audio_formats: ClassVar[dict[str, AudioFormat]]
    preferred_audio_format: ClassVar[str]
    default_voice: ClassVar[Optional[str]] = None

    @classmethod
    def get_key(cls):
        return get_key(None, cls.key_name, cls.key_envvar)

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        response_format: Optional[str] = None
    ) -> ContextManager[StreamingAudio]:
        pass


class OpenAITextToSpeechModel(TextToSpeechModel):
    key_name = "openai"
    key_envvar = "OPENAI_API_KEY"

    audio_formats = {
        "mp3": AudioFormat.mp3(),
        "opus": AudioFormat.opus(),
        "aac": AudioFormat.aac(),
        "flac": AudioFormat.flac(),
        "wav": AudioFormat.wav(),
        "pcm": AudioFormat.pcm(24000),
    }
    preferred_audio_format = 'aac'
    default_voice = 'nova'

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=self.get_key())

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        response_format: Optional[str] = None
    ):
        if response_format:
            ResponseFormat = Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']
            if response_format not in self.audio_formats:
                raise RuntimeError(f"Unsupported format {response_format}")

        return self.client.audio.speech.with_streaming_response.create(
            model=self.model_name, input=text, voice=voice or self.default_voice,
            instructions=instructions or NOT_GIVEN,
            response_format=cast(ResponseFormat, response_format) or NOT_GIVEN
        )

class ElevenLabsTextToSpeechModel(TextToSpeechModel):
    key_name = "elevenlabs"
    key_envvar = "ELEVENLABS_API_KEY"

    audio_formats = {
        **{name: AudioFormat.mp3() for name in (
            'mp3_22050_32', 'mp3_44100_32', 'mp3_44100_64',
            'mp3_44100_96', 'mp3_44100_128', 'mp3_44100_192')},
        **{name: AudioFormat.opus() for name in (
            'opus_48000_32', 'opus_48000_64', 'opus_48000_96',
            'opus_48000_128', 'opus_48000_192')},
        **{f"pcm_{rate}": AudioFormat.pcm(rate) for rate in (
            8000, 16000, 22050, 24000, 44100, 48000)},
        "ulaw_8000": AudioFormat.ulaw(),
        "alaw_8000": AudioFormat.alaw()
    }
    preferred_audio_format = 'mp3_44100_96'
    default_voice = 'JBFqnCBsd6RMkjVDRZzb'

    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def client(self):
        from elevenlabs.client import ElevenLabs
        return ElevenLabs(api_key=self.get_key())

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        response_format: Optional[str] = None
    ) -> ContextManager[StreamingAudio]:
        if instructions:
            raise RuntimeError("ElevenLabs does not support instructions")

        return ElevenLabsAudioStream(
            self.client.text_to_speech.stream(
                model_id=self.model_name, text=text, voice_id=voice or self.default_voice,
                output_format=response_format
            )
        )

class ElevenLabsAudioStream:
    def __init__(self, inner):
        self.inner = inner

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type:  Optional[Type[BaseException]],
        exc_val:   Optional[BaseException],
        exc_tb:    Optional[TracebackType]
    ) -> Optional[bool]:
        self.inner.close()
        return None

    def iter_bytes(self, chunk_size: int = 1024) -> Iterator[bytes]:
        buffer = bytearray()
        for chunk in self.inner:
            buffer.extend(chunk)
            while len(buffer) >= chunk_size:
                yield bytes(buffer[:chunk_size])
                buffer = buffer[chunk_size:]
        if buffer:
            yield bytes(buffer)

    def stream_to_file(self, filename: str):
        with open(filename, "wb") as file:
            for chunk in self.inner:
                file.write(chunk)


_models: dict[str, TextToSpeechModel | Callable[[], TextToSpeechModel]] = {}

def register_tts_model(name: str, factory: TextToSpeechModel | Callable[[], TextToSpeechModel]):
    _models[name] = factory

def get_tts_model(name: str) -> TextToSpeechModel:
    if name not in _models:
        raise RuntimeError(f"No TTS Model named {name}")
    model_or_factory = _models[name]
    if callable(model_or_factory):
        model_or_factory = _models[name] = model_or_factory()
    return model_or_factory


register_tts_model('tts-1', partial(OpenAITextToSpeechModel, 'tts-1'))
register_tts_model('tts-1-hd', partial(OpenAITextToSpeechModel, 'tts-1-hd'))
register_tts_model('gpt-4o-mini-tts', partial(OpenAITextToSpeechModel, 'gpt-4o-mini-tts'))
register_tts_model('eleven_multilingual_v2', partial(ElevenLabsTextToSpeechModel, 'eleven_multilingual_v2'))

_has_transformers = importlib.util.find_spec("transformers") is not None

if _has_transformers:
    class TransformersTextToSpeechModel(TextToSpeechModel):
        """
        A local TTS model via 🤗 transformers pipeline("text-to-speech").
        Renders to WAV in RAM, then streams in chunks.
        """
        audio_formats = { "wav": AudioFormat.wav() }
        preferred_audio_format = "wav"

        def __init__(self, model_name: str):
            self.model_name = model_name

        @cached_property
        def _pipeline(self):
            from transformers import pipeline as _hf_pipeline
            return _hf_pipeline("text-to-speech", model=self.model_name)

        def synthesize(
            self,
            text: str,
            *,
            voice: Optional[str] = None,
            instructions: Optional[str] = None,
            response_format: Optional[str] = None
        ) -> ContextManager[StreamingAudio]:
            """
            - voice/instructions are ignored here.
            - response_format must be "wav" (our only entry).
            """
            import wave
            import numpy as np
            fmt = response_format or self.preferred_audio_format
            if fmt not in self.audio_formats:
                raise RuntimeError(f"Unsupported format {fmt}")

            # run the TTS pipeline → {"array": np.ndarray, "sampling_rate": int}
            speech = self._pipeline(text)
            s16: np.ndarray = (speech["audio"] * 32767).astype(np.int16)

            # write to a WAV in RAM
            buf = BytesIO()
            with wave.open(buf, 'wb') as wav:
                wav.setnchannels(s16.shape[0])
                wav.setsampwidth(2)  # 2 bytes for 16-bit audio
                wav.setframerate(speech["sampling_rate"])
                wav.writeframes(s16.tobytes())

            raw_bytes = buf.getvalue()

            return _InMemoryAudio(raw_bytes)

    register_tts_model('facebook/mms-tts-eng', partial(TransformersTextToSpeechModel, 'facebook/mms-tts-eng'))
    register_tts_model('suno/bark', partial(TransformersTextToSpeechModel, 'suno/bark'))
    register_tts_model('suno/bark-small', partial(TransformersTextToSpeechModel, 'suno/bark-small'))


# ----------------------------------------------------------------------
#  Piper / Mimic3 – fully offline TTS via `piper-tts`
# ----------------------------------------------------------------------
try:
    import piper.download_voices
    from piper import PiperVoice
    _has_piper = True

except ModuleNotFoundError:
    _has_piper = False

if _has_piper:
    _PIPER_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "llm-tts" / "piper-tts"
    _PIPER_CACHE.mkdir(parents=True, exist_ok=True)

    class PiperTextToSpeechModel(TextToSpeechModel):
        """
        Local synthesis with piper-tts.
        A *model name* corresponds to a specific Piper voice file.
        """
        audio_formats = { "wav": AudioFormat.wav() }
        preferred_audio_format = "wav"

        def __init__(self, voice_name: str):
            # voice_name is like "en_US-amy-low"
            self.voice_path = _PIPER_CACHE / f"{voice_name}.onnx"
            if not self.voice_path.exists():
                # download_voice will create <voice>.onnx inside the target dir
                print(f"Downloading {voice_name} to {_PIPER_CACHE}")
                piper.download_voices.download_voice(voice_name, _PIPER_CACHE)
            self.voice = PiperVoice.load(self.voice_path)

        def synthesize(
            self,
            text: str,
            *,
            voice: Optional[str] = None,        # ignored
            instructions: Optional[str] = None, # ignored
            response_format: Optional[str] = None
        ) -> ContextManager[StreamingAudio]:
            import wave
            buf = BytesIO()
            with wave.open(buf, "wb") as wav:
                self.voice.synthesize_wav(text, wav)

            return _InMemoryAudio(buf.getvalue())

    # Register every voice the library knows about
    for voice in (
        "ar_JO-kareem-low",
        "ar_JO-kareem-medium",
        "ca_ES-upc_ona-medium",
        "ca_ES-upc_ona-x_low",
        "ca_ES-upc_pau-x_low",
        "cs_CZ-jirka-low",
        "cs_CZ-jirka-medium",
        "cy_GB-bu_tts-medium",
        "cy_GB-gwryw_gogleddol-medium",
        "da_DK-talesyntese-medium",
        "de_DE-eva_k-x_low",
        "de_DE-karlsson-low",
        "de_DE-kerstin-low",
        "de_DE-mls-medium",
        "de_DE-pavoque-low",
        "de_DE-ramona-low",
        "de_DE-thorsten-high",
        "de_DE-thorsten-low",
        "de_DE-thorsten-medium",
        "de_DE-thorsten_emotional-medium",
        "el_GR-rapunzelina-low",
        "en_GB-alan-low",
        "en_GB-alan-medium",
        "en_GB-alba-medium",
        "en_GB-aru-medium",
        "en_GB-cori-high",
        "en_GB-cori-medium",
        "en_GB-jenny_dioco-medium",
        "en_GB-northern_english_male-medium",
        "en_GB-semaine-medium",
        "en_GB-southern_english_female-low",
        "en_GB-vctk-medium",
        "en_US-amy-low",
        "en_US-amy-medium",
        "en_US-arctic-medium",
        "en_US-bryce-medium",
        "en_US-danny-low",
        "en_US-hfc_female-medium",
        "en_US-hfc_male-medium",
        "en_US-joe-medium",
        "en_US-john-medium",
        "en_US-kathleen-low",
        "en_US-kristin-medium",
        "en_US-kusal-medium",
        "en_US-l2arctic-medium",
        "en_US-lessac-high",
        "en_US-lessac-low",
        "en_US-lessac-medium",
        "en_US-libritts-high",
        "en_US-libritts_r-medium",
        "en_US-ljspeech-high",
        "en_US-ljspeech-medium",
        "en_US-norman-medium",
        "en_US-reza_ibrahim-medium",
        "en_US-ryan-high",
        "en_US-ryan-low",
        "en_US-ryan-medium",
        "en_US-sam-medium",
        "es_AR-daniela-high",
        "es_ES-carlfm-x_low",
        "es_ES-davefx-medium",
        "es_ES-mls_10246-low",
        "es_ES-mls_9972-low",
        "es_ES-sharvard-medium",
        "es_MX-ald-medium",
        "es_MX-claude-high",
        "fa_IR-amir-medium",
        "fa_IR-ganji-medium",
        "fa_IR-ganji_adabi-medium",
        "fa_IR-gyro-medium",
        "fa_IR-reza_ibrahim-medium",
        "fi_FI-harri-low",
        "fi_FI-harri-medium",
        "fr_FR-gilles-low",
        "fr_FR-mls-medium",
        "fr_FR-mls_1840-low",
        "fr_FR-siwis-low",
        "fr_FR-siwis-medium",
        "fr_FR-tom-medium",
        "fr_FR-upmc-medium",
        "hi_IN-pratham-medium",
        "hi_IN-priyamvada-medium",
        "hu_HU-anna-medium",
        "hu_HU-berta-medium",
        "hu_HU-imre-medium",
        "is_IS-bui-medium",
        "is_IS-salka-medium",
        "is_IS-steinn-medium",
        "is_IS-ugla-medium",
        "it_IT-paola-medium",
        "it_IT-riccardo-x_low",
        "ka_GE-natia-medium",
        "kk_KZ-iseke-x_low",
        "kk_KZ-issai-high",
        "kk_KZ-raya-x_low",
        "lb_LU-marylux-medium",
        "lv_LV-aivars-medium",
        "ml_IN-arjun-medium",
        "ml_IN-meera-medium",
        "ne_NP-chitwan-medium",
        "ne_NP-google-medium",
        "ne_NP-google-x_low",
        "nl_BE-nathalie-medium",
        "nl_BE-nathalie-x_low",
        "nl_BE-rdh-medium",
        "nl_BE-rdh-x_low",
        "nl_NL-mls-medium",
        "nl_NL-mls_5809-low",
        "nl_NL-mls_7432-low",
        "nl_NL-pim-medium",
        "nl_NL-ronnie-medium",
        "no_NO-talesyntese-medium",
        "pl_PL-darkman-medium",
        "pl_PL-gosia-medium",
        "pl_PL-mc_speech-medium",
        "pl_PL-mls_6892-low",
        "pt_BR-cadu-medium",
        "pt_BR-edresson-low",
        "pt_BR-faber-medium",
        "pt_BR-jeff-medium",
        "pt_PT-tugão-medium",
        "ro_RO-mihai-medium",
        "ru_RU-denis-medium",
        "ru_RU-dmitri-medium",
        "ru_RU-irina-medium",
        "ru_RU-ruslan-medium",
        "sk_SK-lili-medium",
        "sl_SI-artur-medium",
        "sr_RS-serbski_institut-medium",
        "sv_SE-lisa-medium",
        "sv_SE-nst-medium",
        "sw_CD-lanfrica-medium",
        "tr_TR-dfki-medium",
        "tr_TR-fahrettin-medium",
        "tr_TR-fettah-medium",
        "uk_UA-lada-x_low",
        "uk_UA-ukrainian_tts-medium",
        "vi_VN-25hours_single-low",
        "vi_VN-vais1000-medium",
        "vi_VN-vivos-x_low",
        "zh_CN-huayan-medium",
        "zh_CN-huayan-x_low"
    ):
        register_tts_model(f"piper/{voice}", partial(PiperTextToSpeechModel, voice))


class _InMemoryAudio:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self) -> "_InMemoryAudio":
        return self

    def __exit__(
        self,
        exc_type:  Optional[Type[BaseException]],
        exc_val:   Optional[BaseException],
        exc_tb:    Optional[TracebackType]
    ) -> Optional[bool]:
        return None

    def iter_bytes(self, chunk_size: int = 1024) -> Iterator[bytes]:
        # yield consecutive slices of the byte blob
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]

    def stream_to_file(self, filename: str):
        with open(filename, "wb") as f:
            f.write(self._data)


CHUNK_SIZE = 4096
DEFAULT_TTS_MODEL = 'gpt-4o-mini-tts'
