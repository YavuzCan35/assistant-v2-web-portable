import argparse
import collections
import ctypes
import os
import queue
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from kokoro import KPipeline
from openai import OpenAI


def env(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return default if value is None else value


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Drop ANSI escapes and control characters that can hide/overwrite console output.
    text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)
    chars: list[str] = []
    for ch in text:
        if ch in {"\r", "\n", "\t"}:
            chars.append(" ")
            continue
        if ch.isprintable():
            chars.append(ch)
        else:
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


@dataclass
class Settings:
    input_mode: str
    lm_base_url: str
    lm_model: str
    lm_temperature: float
    lm_max_tokens: int | None
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_language: str | None
    whisper_initial_prompt: str | None
    stt_preprocess: "STTPreprocessConfig"
    input_device_name: str | None
    output_device_name: str | None
    # 0 means auto-detect from device capabilities.
    input_sample_rate: int
    push_to_talk: bool
    push_to_talk_key: str
    kokoro_repo_id: str
    kokoro_lang_code: str
    kokoro_voice: str
    kokoro_speed: float
    kokoro_split_pattern: str | None
    tts_simple_playback: bool
    tts_max_chars: int
    replay_all_captured_audio: bool
    replay_all_captured_audio_max_sec: float
    replay_on_stt_fail: bool
    replay_on_stt_fail_max_sec: float
    assistant_prompt: str
    assistant_prompt_source: str
    no_tts: bool


@dataclass
class STTPreprocessConfig:
    enabled: bool
    denoise_strength: float
    noise_floor_percentile: float
    highpass_hz: float
    normalize_audio: bool
    target_rms: float
    max_gain: float
    peak_limit: float


@dataclass
class STTTranscription:
    text: str
    reason: str


class MicrophoneTurnDetector:
    def __init__(
        self,
        input_device: str | int | None = None,
        input_channels: int = 1,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        padding_ms: int = 450,
        vad_aggressiveness: int = 2,
        min_utterance_sec: float = 0.4,
        max_utterance_sec: float = 12.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.input_channels = max(1, int(input_channels))
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.padding_frames = max(1, int(padding_ms / frame_ms))
        self.min_utterance_sec = min_utterance_sec
        self.max_utterance_sec = max_utterance_sec
        self.vad = webrtcvad.Vad(vad_aggressiveness)

    def _extract_mono_frame(
        self,
        frame: np.ndarray,
        selected_channel: int | None,
    ) -> tuple[np.ndarray, int]:
        if frame.ndim == 1:
            return frame.astype(np.int16, copy=False), 0
        if frame.shape[1] <= 1:
            return frame[:, 0].astype(np.int16, copy=False), 0

        if selected_channel is None:
            energy = np.mean(np.abs(frame.astype(np.float32, copy=False)), axis=0)
            selected_channel = int(np.argmax(energy))
        mono = frame[:, selected_channel].astype(np.int16, copy=False)
        return mono, selected_channel

    def listen_for_turn(self) -> np.ndarray:
        ring: collections.deque[tuple[bytes, bool]] = collections.deque(
            maxlen=self.padding_frames
        )
        voiced_frames: list[bytes] = []
        triggered = False
        start_ts = 0.0
        selected_channel: int | None = None

        with sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=self.input_channels,
            dtype="int16",
            blocksize=self.frame_samples,
        ) as stream:
            while True:
                frame, _overflowed = stream.read(self.frame_samples)
                mono, selected_channel = self._extract_mono_frame(frame, selected_channel)
                pcm = mono.tobytes()
                is_speech = self.vad.is_speech(pcm, self.sample_rate)

                if not triggered:
                    ring.append((pcm, is_speech))
                    voiced_count = sum(1 for _, speech in ring if speech)
                    if voiced_count > 0.8 * ring.maxlen:
                        triggered = True
                        start_ts = time.time()
                        voiced_frames.extend(p for p, _ in ring)
                        ring.clear()
                else:
                    voiced_frames.append(pcm)
                    ring.append((pcm, is_speech))
                    unvoiced_count = sum(1 for _, speech in ring if not speech)
                    utterance_sec = len(voiced_frames) * self.frame_ms / 1000.0

                    if utterance_sec >= self.min_utterance_sec and (
                        unvoiced_count > 0.9 * ring.maxlen
                    ):
                        break
                    if utterance_sec >= self.max_utterance_sec:
                        break

                if triggered and (time.time() - start_ts) > (self.max_utterance_sec + 1.5):
                    break

        audio = np.frombuffer(b"".join(voiced_frames), dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    def listen_for_turn_push_to_talk(self, vk_code: int) -> np.ndarray:
        if os.name != "nt":
            raise RuntimeError("Push-to-talk key mode is currently supported on Windows only.")

        frames: list[np.ndarray] = []
        user32 = ctypes.windll.user32
        pressed_mask = 0x8000
        selected_channel: int | None = None

        while not (user32.GetAsyncKeyState(vk_code) & pressed_mask):
            time.sleep(0.01)

        with sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=self.input_channels,
            dtype="int16",
            blocksize=self.frame_samples,
        ) as stream:
            while user32.GetAsyncKeyState(vk_code) & pressed_mask:
                frame, _overflowed = stream.read(self.frame_samples)
                mono, selected_channel = self._extract_mono_frame(frame, selected_channel)
                frames.append(mono.copy())

            # Keep a short release tail so trailing syllables are not clipped.
            release_tail_frames = max(1, int(150 / self.frame_ms))
            for _ in range(release_tail_frames):
                frame, _overflowed = stream.read(self.frame_samples)
                mono, selected_channel = self._extract_mono_frame(frame, selected_channel)
                frames.append(mono.copy())

        if not frames:
            return np.array([], dtype=np.float32)
        audio = np.concatenate(frames).astype(np.float32)
        return audio / 32768.0


class FasterWhisperSTT:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        language: str | None,
        initial_prompt: str | None,
        preprocess: STTPreprocessConfig,
    ) -> None:
        self.language = language or None
        self.initial_prompt = initial_prompt
        self.preprocess = preprocess
        self.device = resolve_whisper_device(device)
        self.model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=compute_type,
        )

    def _decode_text(self, audio: np.ndarray, vad_filter: bool) -> str:
        segments, _info = self.model.transcribe(
            audio=audio,
            language=self.language,
            initial_prompt=self.initial_prompt,
            temperature=0.0,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=vad_filter,
        )
        parts = [normalize_text(seg.text) for seg in segments]
        parts = [p for p in parts if p]
        return normalize_text(" ".join(parts))

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        result = self.transcribe_with_reason(audio, sample_rate)
        return result.text

    def transcribe_with_reason(self, audio: np.ndarray, sample_rate: int) -> STTTranscription:
        if audio.size == 0:
            return STTTranscription(text="", reason="empty_audio_buffer")

        duration_sec = float(audio.shape[0]) / float(sample_rate) if sample_rate > 0 else 0.0
        if duration_sec < 0.20:
            return STTTranscription(
                text="",
                reason=f"audio_too_short ({duration_sec:.2f}s)",
            )

        work_audio = audio.astype(np.float32, copy=False)
        if self.preprocess.enabled:
            work_audio = preprocess_stt_audio(
                audio=work_audio,
                sample_rate=sample_rate,
                cfg=self.preprocess,
            )

        rms = float(np.sqrt(np.mean(np.square(work_audio), dtype=np.float64))) if work_audio.size else 0.0
        if rms < 0.0015:
            return STTTranscription(
                text="",
                reason=f"very_low_signal (rms={rms:.4f})",
            )

        if sample_rate != 16000:
            work_audio = resample_audio_linear(work_audio, src_rate=sample_rate, dst_rate=16000)

        merged = self._decode_text(work_audio, vad_filter=True)
        if not merged and duration_sec >= 0.8 and rms >= 0.0025:
            merged = self._decode_text(work_audio, vad_filter=False)
            if merged:
                return STTTranscription(
                    text=merged,
                    reason=f"ok_no_vad_fallback (duration={duration_sec:.2f}s rms={rms:.4f})",
                )

        if not merged:
            return STTTranscription(
                text="",
                reason=f"no_segments_after_vad (duration={duration_sec:.2f}s rms={rms:.4f})",
            )
        if not merged:
            return STTTranscription(
                text="",
                reason="segments_empty_after_normalization",
            )
        return STTTranscription(text=merged, reason="ok")


class LMStudioChat:
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int | None) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")

    def _request(
        self,
        messages: list[dict[str, object]],
        max_tokens: int | None,
        request_timeout_sec: float | None = None,
    ):
        request: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if max_tokens is not None and max_tokens > 0:
            request["max_tokens"] = max_tokens
        if request_timeout_sec is not None and request_timeout_sec > 0:
            request["timeout"] = float(request_timeout_sec)
        return self.client.chat.completions.create(**request)

    def complete(
        self,
        messages: list[dict[str, object]],
        request_timeout_sec: float | None = None,
    ) -> str:
        response = self._request(messages, self.max_tokens, request_timeout_sec=request_timeout_sec)
        choice = response.choices[0]
        message = choice.message
        raw_content = re.sub(r"<\|[^>]+?\|>", "", str(message.content or ""))
        content = normalize_text(raw_content)
        if content:
            return content

        reasoning = normalize_text(str(getattr(message, "reasoning_content", "") or ""))
        finish_reason = normalize_text(str(choice.finish_reason or "")).lower()
        if finish_reason == "length":
            current = self.max_tokens if (self.max_tokens and self.max_tokens > 0) else 0
            retry_max_tokens = min(max(current * 3, 180), 640)
            if retry_max_tokens > current:
                print(
                    f"[LLM warning] Empty content at token limit "
                    f"({current if current > 0 else 'auto'}); retrying with {retry_max_tokens}."
                )
                retry_response = self._request(
                    messages,
                    retry_max_tokens,
                    request_timeout_sec=request_timeout_sec,
                )
                retry_choice = retry_response.choices[0]
                retry_message = retry_choice.message
                retry_raw = re.sub(r"<\|[^>]+?\|>", "", str(retry_message.content or ""))
                retry_content = normalize_text(retry_raw)
                if retry_content:
                    return retry_content
                retry_reasoning = normalize_text(
                    str(getattr(retry_message, "reasoning_content", "") or "")
                )
                extracted_retry = extract_reply_from_reasoning(retry_reasoning)
                if extracted_retry:
                    print(
                        f"[LLM warning] Empty content after retry at {retry_max_tokens} tokens; "
                        "using extracted reasoning fallback."
                    )
                    return extracted_retry

        extracted = extract_reply_from_reasoning(reasoning)
        if extracted:
            print("[LLM warning] Empty content; using extracted reasoning fallback.")
            return extracted
        return ""


def extract_reply_from_reasoning(reasoning: str) -> str:
    text = normalize_text(reasoning)
    if not text:
        return ""

    patterns = [
        r'(?:final answer|answer|reply|like)\s*:\s*["\']([^"\']{4,280})["\']',
        r'(?:final answer|answer|reply|like)\s*:\s*(.+)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = normalize_text(match.group(1))
        if candidate:
            return candidate

    quoted = re.findall(r'["\']([^"\']{6,280})["\']', text)
    if quoted:
        candidate = normalize_text(quoted[-1])
        if candidate:
            return candidate

    sentences = [normalize_text(x) for x in re.split(r"(?<=[.!?])\s+", text) if normalize_text(x)]
    if sentences:
        tail = sentences[-1]
        if len(tail) > 4:
            return tail
    return ""


def register_extra_kokoro_espeak_language(lang_code: str) -> str:
    normalized = (lang_code or "").strip().lower()
    if not normalized:
        return normalized

    extra_map: dict[str, tuple[str, str]] = {
        "tr": ("t", "tr"),
        "tr-tr": ("t", "tr"),
        "turkish": ("t", "tr"),
        "t": ("t", "tr"),
        "de": ("g", "de"),
        "de-de": ("g", "de"),
        "german": ("g", "de"),
        "g": ("g", "de"),
    }
    mapping = extra_map.get(normalized)
    if mapping is None:
        return normalized

    short_code, espeak_lang = mapping
    try:
        from kokoro import pipeline as kokoro_pipeline
    except Exception:
        return normalized

    kokoro_pipeline.LANG_CODES.setdefault(short_code, espeak_lang)
    kokoro_pipeline.ALIASES.setdefault(espeak_lang, short_code)
    kokoro_pipeline.ALIASES.setdefault(normalized, short_code)
    return short_code


class KokoroTTS:
    def __init__(
        self,
        repo_id: str,
        lang_code: str,
        voice: str,
        speed: float,
        split_pattern: str | None,
        device: str,
        output_device: str | int | None,
        simple_playback: bool,
    ) -> None:
        self.pipeline_sample_rate = 24000
        self.voice = voice
        self.speed = speed
        self.split_pattern = split_pattern
        self.output_device = output_device
        self.simple_playback = simple_playback
        # "Simple" mode mirrors tts_serial_test.py and kokoro_us_uk_voices.py behavior.
        if self.simple_playback or self.output_device is None:
            self.output_sample_rate = self.pipeline_sample_rate
        else:
            self.output_sample_rate = resolve_output_sample_rate(
                output_device=self.output_device,
                requested_rate=self.pipeline_sample_rate,
            )
        resolved_lang_code = register_extra_kokoro_espeak_language(lang_code)
        if resolved_lang_code != (lang_code or "").strip().lower():
            print(
                f"[Kokoro] Mapping lang_code='{lang_code}' to internal code "
                f"'{resolved_lang_code}' via EspeakG2P."
            )
        self.pipeline = KPipeline(
            lang_code=resolved_lang_code,
            repo_id=repo_id,
            device=device,
        )

    def speak(self, text: str, cancel_event: threading.Event | None = None) -> bool:
        if not text:
            return True
        synth_start = time.perf_counter()
        chunks: list[np.ndarray] = []
        for result in self.pipeline(
            text=text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=self.split_pattern,
        ):
            if cancel_event is not None and cancel_event.is_set():
                print("[TTS] Interrupted during synthesis.")
                return False
            chunk_audio = result.audio
            if chunk_audio is None:
                continue
            if hasattr(chunk_audio, "detach"):
                chunk_np = chunk_audio.detach().cpu().numpy().astype(np.float32)
            else:
                chunk_np = np.asarray(chunk_audio, dtype=np.float32)
            chunks.append(chunk_np)

        if not chunks:
            return False

        audio = np.concatenate(chunks)
        synth_elapsed = time.perf_counter() - synth_start
        print(f"[TTS] Synthesis done in {synth_elapsed:.2f}s")
        if cancel_event is not None and cancel_event.is_set():
            print("[TTS] Interrupted before playback.")
            return False
        if self.simple_playback:
            print("[TTS] Playing response (simple mode)...")
            sd.play(audio, self.pipeline_sample_rate)
            duration_sec = float(audio.shape[0]) / float(self.pipeline_sample_rate)
            wait_for_playback(
                timeout_sec=max(8.0, duration_sec + 5.0),
                cancel_event=cancel_event,
            )
            return not (cancel_event is not None and cancel_event.is_set())

        if self.output_sample_rate != self.pipeline_sample_rate:
            audio = resample_audio_linear(
                audio,
                src_rate=self.pipeline_sample_rate,
                dst_rate=self.output_sample_rate,
            )
        play_start = time.perf_counter()
        print("[TTS] Playing response...")
        sd.play(audio, self.output_sample_rate, device=self.output_device)
        duration_sec = 0.0
        if self.output_sample_rate > 0:
            duration_sec = float(audio.shape[0]) / float(self.output_sample_rate)
        wait_for_playback(
            timeout_sec=max(8.0, duration_sec + 5.0),
            cancel_event=cancel_event,
        )
        play_elapsed = time.perf_counter() - play_start
        print(f"[TTS] Playback done in {play_elapsed:.2f}s")
        return not (cancel_event is not None and cancel_event.is_set())


def trim_history(messages: list[dict[str, str]], max_non_system_messages: int = 12) -> list[dict[str, str]]:
    if not messages:
        return messages
    if len(messages) <= max_non_system_messages + 1:
        return messages
    return [messages[0], *messages[-max_non_system_messages:]]


def env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_whisper_device(device: str) -> str:
    text = (device or "").strip().lower()
    if text in {"cpu", "cuda"}:
        if text == "cuda":
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "WHISPER_DEVICE is set to 'cuda' but CUDA is unavailable in this environment. "
                        "Install a CUDA-enabled torch build or set WHISPER_DEVICE=cpu."
                    )
            except ImportError as exc:
                raise RuntimeError("Torch is required for CUDA device detection.") from exc
        return text
    if text in {"auto", ""}:
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cuda"
    return "cuda"


def normalize_lm_openai_base_url(base_url: str) -> str:
    text = (base_url or "").strip().rstrip("/")
    if not text:
        return "http://localhost:1234/v1"
    # Force LM Studio OpenAI-compatible endpoint path.
    if text.endswith("/api/v1"):
        return text[:-7] + "/v1"
    if text.endswith("/v1"):
        return text
    if text.startswith("http://") or text.startswith("https://"):
        return text + "/v1"
    return "http://localhost:1234/v1"


def resample_audio_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if audio.size == 0 or src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"Invalid sample rate conversion: {src_rate} -> {dst_rate}")

    src_len = audio.shape[0]
    dst_len = max(1, int(round(src_len * (dst_rate / src_rate))))
    if src_len < 2 or dst_len < 2:
        return audio.astype(np.float32, copy=False)

    src_x = np.linspace(0.0, 1.0, num=src_len, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, dtype=np.float64)
    resampled = np.interp(dst_x, src_x, audio.astype(np.float64, copy=False))
    return resampled.astype(np.float32, copy=False)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_highpass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if audio.size < 2 or sample_rate <= 0 or cutoff_hz <= 0:
        return audio
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / float(sample_rate)
    alpha = rc / (rc + dt)

    x = audio.astype(np.float32, copy=False)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.shape[0]):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    return y


def apply_noise_gate(
    audio: np.ndarray,
    sample_rate: int,
    denoise_strength: float,
    noise_floor_percentile: float,
) -> np.ndarray:
    if audio.size == 0:
        return audio
    strength = _clamp(float(denoise_strength), 0.0, 1.0)
    if strength <= 0.0:
        return audio

    frame_len = max(1, int(round(sample_rate * 0.02)))  # 20ms
    hop = max(1, frame_len // 2)
    x = audio.astype(np.float32, copy=False)
    frame_rms: list[float] = []
    starts: list[int] = []
    for start in range(0, x.shape[0], hop):
        end = min(x.shape[0], start + frame_len)
        if end <= start:
            break
        frame = x[start:end]
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        frame_rms.append(rms)
        starts.append(start)
    if not frame_rms:
        return audio

    floor_pct = _clamp(float(noise_floor_percentile), 1.0, 60.0)
    noise_floor = float(np.percentile(np.asarray(frame_rms, dtype=np.float32), floor_pct))
    threshold = max(noise_floor * 2.5, 1e-4)
    attenuation = 1.0 - strength

    gain = np.ones_like(x, dtype=np.float32)
    for start, rms in zip(starts, frame_rms):
        end = min(x.shape[0], start + frame_len)
        if rms < threshold:
            gain[start:end] = np.minimum(gain[start:end], attenuation)

    smooth_len = max(5, int(round(sample_rate * 0.01)))  # 10ms smoothing
    if smooth_len % 2 == 0:
        smooth_len += 1
    kernel = np.hanning(smooth_len).astype(np.float32)
    denom = float(np.sum(kernel))
    if denom > 0:
        kernel /= denom
        gain = np.convolve(gain, kernel, mode="same").astype(np.float32, copy=False)

    return x * gain


def normalize_audio_level(
    audio: np.ndarray,
    target_rms: float,
    max_gain: float,
    peak_limit: float,
) -> np.ndarray:
    if audio.size == 0:
        return audio
    x = audio.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    if rms < 1e-7:
        return x

    target = _clamp(float(target_rms), 1e-4, 0.5)
    max_gain_clamped = _clamp(float(max_gain), 1.0, 50.0)
    gain = target / rms
    gain = _clamp(gain, 0.1, max_gain_clamped)

    y = x * gain
    peak = float(np.max(np.abs(y)))
    limit = _clamp(float(peak_limit), 0.1, 1.0)
    if peak > limit and peak > 0:
        y = y * (limit / peak)
    return y.astype(np.float32, copy=False)


def preprocess_stt_audio(audio: np.ndarray, sample_rate: int, cfg: STTPreprocessConfig) -> np.ndarray:
    if audio.size == 0:
        return audio
    x = audio.astype(np.float32, copy=False)
    # Remove DC offset before filtering and gain.
    x = x - np.mean(x, dtype=np.float64)
    x = apply_highpass_filter(x, sample_rate=sample_rate, cutoff_hz=cfg.highpass_hz)
    x = apply_noise_gate(
        x,
        sample_rate=sample_rate,
        denoise_strength=cfg.denoise_strength,
        noise_floor_percentile=cfg.noise_floor_percentile,
    )
    if cfg.normalize_audio:
        x = normalize_audio_level(
            x,
            target_rms=cfg.target_rms,
            max_gain=cfg.max_gain,
            peak_limit=cfg.peak_limit,
        )
    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


def describe_audio_device(device_ref: str | int | None, kind: str) -> str:
    if device_ref is None:
        return "default device"
    device_info = sd.query_devices(device_ref, kind)
    hostapi_idx = int(device_info.get("hostapi", -1))
    hostapi_name = "unknown"
    if hostapi_idx >= 0:
        hostapi_name = str(sd.query_hostapis(hostapi_idx).get("name", "unknown"))
    default_sr = int(round(float(device_info.get("default_samplerate", 0))))
    return f"{device_info.get('name', device_ref)} (hostapi: {hostapi_name}, default_sr: {default_sr})"


def list_audio_devices() -> None:
    devices = sd.query_devices()
    print("Available input devices:")
    for idx, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        hostapi_idx = int(device.get("hostapi", -1))
        hostapi_name = "unknown"
        if hostapi_idx >= 0:
            hostapi_name = sd.query_hostapis(hostapi_idx)["name"]
        default_sr = int(round(float(device.get("default_samplerate", 0))))
        print(f"  [{idx}] {device['name']}  (hostapi: {hostapi_name}, default_sr: {default_sr})")
    print("\nAvailable output devices:")
    for idx, device in enumerate(devices):
        if int(device.get("max_output_channels", 0)) <= 0:
            continue
        hostapi_idx = int(device.get("hostapi", -1))
        hostapi_name = "unknown"
        if hostapi_idx >= 0:
            hostapi_name = sd.query_hostapis(hostapi_idx)["name"]
        default_sr = int(round(float(device.get("default_samplerate", 0))))
        print(f"  [{idx}] {device['name']}  (hostapi: {hostapi_name}, default_sr: {default_sr})")


def resolve_input_device(device_hint: str | None) -> str | int | None:
    if not device_hint:
        return None

    text = device_hint.strip()
    if not text:
        return None

    devices = sd.query_devices()
    if text.isdigit():
        index = int(text)
        if index < 0 or index >= len(devices):
            raise ValueError(f"Input device index out of range: {index}")
        device = devices[index]
        if int(device.get("max_input_channels", 0)) <= 0:
            raise ValueError(f"Device [{index}] is not an input device: {device['name']}")
        return index

    lowered = text.lower()
    matches: list[tuple[int, int]] = []
    for idx, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        if lowered in str(device.get("name", "")).lower():
            hostapi_idx = int(device.get("hostapi", -1))
            hostapi_name = ""
            if hostapi_idx >= 0:
                hostapi_name = str(sd.query_hostapis(hostapi_idx).get("name", ""))
            score = {
                "Windows WASAPI": 40,
                "Windows WDM-KS": 30,
                "Windows DirectSound": 20,
                "MME": 10,
            }.get(hostapi_name, 0)
            matches.append((score, idx))
    if matches:
        matches.sort(reverse=True)
        return matches[0][1]
    raise ValueError(f"No input device matched '{device_hint}'. Use --list-devices.")


def resolve_output_device(device_hint: str | None) -> str | int | None:
    if not device_hint:
        return None

    text = device_hint.strip()
    if not text:
        return None

    devices = sd.query_devices()
    if text.isdigit():
        index = int(text)
        if index < 0 or index >= len(devices):
            raise ValueError(f"Output device index out of range: {index}")
        device = devices[index]
        if int(device.get("max_output_channels", 0)) <= 0:
            raise ValueError(f"Device [{index}] is not an output device: {device['name']}")
        return index

    lowered = text.lower()
    matches: list[tuple[int, int]] = []
    for idx, device in enumerate(devices):
        if int(device.get("max_output_channels", 0)) <= 0:
            continue
        if lowered in str(device.get("name", "")).lower():
            hostapi_idx = int(device.get("hostapi", -1))
            hostapi_name = ""
            if hostapi_idx >= 0:
                hostapi_name = str(sd.query_hostapis(hostapi_idx).get("name", ""))
            score = {
                "Windows WASAPI": 40,
                "Windows WDM-KS": 30,
                "Windows DirectSound": 20,
                "MME": 10,
            }.get(hostapi_name, 0)
            matches.append((score, idx))
    if matches:
        matches.sort(reverse=True)
        return matches[0][1]
    raise ValueError(f"No output device matched '{device_hint}'. Use --list-devices.")


def resolve_input_sample_rate(
    input_device: str | int | None,
    requested_rate: int,
    for_vad: bool,
) -> int:
    allowed_vad_rates = (8000, 16000, 32000, 48000)

    device_info = sd.query_devices(input_device, "input")
    default_sr = int(round(float(device_info.get("default_samplerate", 0))))

    candidates: list[int] = []
    if requested_rate > 0:
        candidates.append(requested_rate)
    if default_sr > 0:
        candidates.append(default_sr)

    if for_vad:
        candidates.extend([16000, 48000, 32000, 8000])
    else:
        candidates.extend([48000, 44100, 32000, 16000])

    seen: set[int] = set()
    for rate in candidates:
        if rate in seen or rate <= 0:
            continue
        seen.add(rate)
        if for_vad and rate not in allowed_vad_rates:
            continue
        try:
            sd.check_input_settings(
                device=input_device,
                samplerate=rate,
                channels=1,
                dtype="int16",
            )
            return rate
        except Exception:
            continue

    mode = "VAD/always-on" if for_vad else "push-to-talk"
    raise ValueError(
        f"Could not find a valid input sample rate for device '{device_info.get('name', input_device)}' "
        f"in {mode} mode. Try setting INPUT_SAMPLE_RATE explicitly."
    )


def resolve_input_channels(input_device: str | int | None) -> int:
    device_info = sd.query_devices(input_device, "input")
    max_input_channels = int(device_info.get("max_input_channels", 1))
    if max_input_channels <= 1:
        return 1
    # Capture at most stereo; detector picks the strongest channel.
    return min(2, max_input_channels)


def resolve_output_sample_rate(output_device: str | int | None, requested_rate: int) -> int:
    device_info = sd.query_devices(output_device, "output")
    default_sr = int(round(float(device_info.get("default_samplerate", 0))))

    candidates: list[int] = []
    if requested_rate > 0:
        candidates.append(requested_rate)
    if default_sr > 0:
        candidates.append(default_sr)
    candidates.extend([48000, 44100, 32000, 24000, 22050, 16000])

    seen: set[int] = set()
    for rate in candidates:
        if rate in seen or rate <= 0:
            continue
        seen.add(rate)
        try:
            sd.check_output_settings(
                device=output_device,
                samplerate=rate,
                channels=1,
                dtype="float32",
            )
            return rate
        except Exception:
            continue

    raise ValueError(
        f"Could not find a valid output sample rate for device "
        f"'{device_info.get('name', output_device)}'."
    )


def wait_for_playback(timeout_sec: float, cancel_event: threading.Event | None = None) -> None:
    completed = threading.Event()
    errors: list[Exception] = []

    def _waiter() -> None:
        try:
            sd.wait()
        except Exception as exc:
            errors.append(exc)
        finally:
            completed.set()

    thread = threading.Thread(target=_waiter, daemon=True)
    thread.start()
    start = time.perf_counter()
    while True:
        if completed.wait(0.05):
            break
        if cancel_event is not None and cancel_event.is_set():
            sd.stop()
            completed.wait(2.0)
            return
        if (time.perf_counter() - start) > timeout_sec:
            sd.stop()
            raise RuntimeError(f"TTS playback timed out after {timeout_sec:.1f}s and was stopped.")
    if errors:
        raise errors[0]


def replay_audio_for_debug(
    audio: np.ndarray,
    sample_rate: int,
    output_device: str | int | None,
    max_seconds: float,
) -> None:
    if audio.size == 0 or sample_rate <= 0:
        return
    clip = audio.astype(np.float32, copy=False)
    if max_seconds > 0:
        max_samples = int(round(max_seconds * sample_rate))
        if max_samples > 0 and clip.shape[0] > max_samples:
            clip = clip[:max_samples]
    sd.play(clip, sample_rate, device=output_device)
    clip_duration = float(clip.shape[0]) / float(sample_rate)
    wait_for_playback(timeout_sec=max(6.0, clip_duration + 3.0))


def shrink_text_for_tts(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    cleaned = normalize_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    sentence_cut = re.search(r"^(.{1," + str(max_chars) + r"}?[.!?])(?:\s|$)", cleaned)
    if sentence_cut:
        return sentence_cut.group(1).strip()
    clipped = cleaned[:max_chars].rstrip(" ,;:-")
    return f"{clipped}."


def normalize_text_for_tts(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    # Remove markdown, URLs, and formatting artifacts first.
    normalized = re.sub(r"```.+?```", " ", normalized)
    normalized = re.sub(r"`{3,}", " ", normalized)
    normalized = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+|www\.[^\s)]+)\)", r"\1", normalized)
    normalized = re.sub(r"https?://\S+|www\.\S+", " ", normalized)
    normalized = re.sub(r"<[^>]{1,120}>", " ", normalized)
    normalized = re.sub(r":[a-zA-Z0-9_+\-]{2,32}:", " ", normalized)
    # Remove markdown emphasis and convert list markers to pause-friendly separators.
    normalized = re.sub(r"\*\*(.*?)\*\*", r"\1", normalized)
    normalized = re.sub(r"__(.*?)__", r"\1", normalized)
    normalized = re.sub(r"`([^`]+)`", r"\1", normalized)
    normalized = re.sub(r"(^|[\s:;])\*\s+", r"\1", normalized)
    normalized = re.sub(r"\s+\*\s+", ". ", normalized)
    normalized = unicodedata.normalize("NFKC", normalized)

    cleaned_chars: list[str] = []
    for ch in normalized:
        if ch.isspace():
            cleaned_chars.append(" ")
            continue
        category = unicodedata.category(ch)
        if category in {"Cf", "Cs", "Co", "Cn"}:
            cleaned_chars.append(" ")
            continue
        # Keep letters/numbers in any language.
        if category[0] in {"L", "N"}:
            cleaned_chars.append(ch)
            continue
        # Keep speech-friendly punctuation, drop symbol-heavy noise (emoji/math/icons).
        if ch in ".,!?;:'\"-()/%+&":
            cleaned_chars.append(ch)
            continue
        if category[0] == "S":
            cleaned_chars.append(" ")
            continue
        if category[0] == "P":
            cleaned_chars.append(ch)
            continue
        cleaned_chars.append(" ")

    normalized = "".join(cleaned_chars)
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[|~_^*=#<>\[\]{}\\]+", " ", normalized)
    normalized = re.sub(r"([!?.,;:])\1+", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\.\.+", ".", normalized)
    return normalized


def resolve_vk_code(key_name: str) -> int:
    name = key_name.strip().upper()
    if len(name) == 1 and (name.isalnum() or name in {" ", ";", ",", ".", "-", "="}):
        return ord(name)
    if name.startswith("F") and name[1:].isdigit():
        fn = int(name[1:])
        if 1 <= fn <= 24:
            return 0x6F + fn

    vk_map = {
        "SPACE": 0x20,
        "TAB": 0x09,
        "ENTER": 0x0D,
        "ESC": 0x1B,
        "ESCAPE": 0x1B,
        "SHIFT": 0x10,
        "CTRL": 0x11,
        "CONTROL": 0x11,
        "ALT": 0x12,
        "LSHIFT": 0xA0,
        "RSHIFT": 0xA1,
        "LCTRL": 0xA2,
        "RCTRL": 0xA3,
        "LALT": 0xA4,
        "RALT": 0xA5,
        "CAPSLOCK": 0x14,
    }
    code = vk_map.get(name)
    if code is None:
        raise ValueError(f"Unsupported push-to-talk key: '{key_name}'.")
    return code


def build_settings(
    no_tts: bool,
    force_always_on: bool,
    cli_ptt_key: str | None,
    cli_input_mode: str | None,
) -> Settings:
    input_mode_raw = (cli_input_mode or env("INPUT_MODE", "auto")).strip().lower()
    input_mode = input_mode_raw if input_mode_raw in {"voice", "text", "auto"} else "auto"
    whisper_language = env("WHISPER_LANGUAGE", "").strip() or None
    whisper_initial_prompt = env("WHISPER_INITIAL_PROMPT", "").strip() or None
    stt_preprocess = STTPreprocessConfig(
        enabled=env_bool("STT_PREPROCESS", True),
        denoise_strength=float(env("STT_DENOISE_STRENGTH", "0.7")),
        noise_floor_percentile=float(env("STT_NOISE_FLOOR_PERCENTILE", "20")),
        highpass_hz=float(env("STT_HIGHPASS_HZ", "70")),
        normalize_audio=env_bool("STT_NORMALIZE", True),
        target_rms=float(env("STT_TARGET_RMS", "0.08")),
        max_gain=float(env("STT_MAX_GAIN", "8.0")),
        peak_limit=float(env("STT_PEAK_LIMIT", "0.95")),
    )
    input_device_name = env("INPUT_DEVICE_NAME", "").strip() or None
    output_device_name = env("OUTPUT_DEVICE_NAME", "").strip() or None
    push_to_talk = env_bool("PUSH_TO_TALK", True)
    if force_always_on:
        push_to_talk = False
    input_sample_rate_raw = env("INPUT_SAMPLE_RATE", "0").strip().lower()
    if input_sample_rate_raw in {"", "0", "auto"}:
        input_sample_rate = 0
    else:
        input_sample_rate = int(input_sample_rate_raw)
    push_to_talk_key = (cli_ptt_key or env("PUSH_TO_TALK_KEY", "F8")).strip() or "F8"
    kokoro_split_pattern_raw = env("KOKORO_SPLIT_PATTERN", r"\n+").strip()
    kokoro_split_pattern = None if kokoro_split_pattern_raw.lower() == "none" else kokoro_split_pattern_raw
    tts_simple_playback = env_bool("TTS_SIMPLE_PLAYBACK", True)
    tts_max_chars = max(0, int(env("TTS_MAX_CHARS", "0")))
    replay_all_captured_audio = env_bool("REPLAY_ALL_CAPTURED_AUDIO", False)
    replay_all_captured_audio_max_sec = max(0.0, float(env("REPLAY_ALL_CAPTURED_AUDIO_MAX_SEC", "0")))
    replay_on_stt_fail = env_bool("REPLAY_ON_STT_FAIL", True)
    replay_on_stt_fail_max_sec = max(0.0, float(env("REPLAY_ON_STT_FAIL_MAX_SEC", "6.0")))
    lm_max_tokens_raw = env("LM_MAX_TOKENS", "0").strip().lower()
    lm_max_tokens = None if lm_max_tokens_raw in {"", "0", "none", "null", "auto"} else int(lm_max_tokens_raw)
    system_prompt = env("SYSTEM_PROMPT", "").strip()
    if not system_prompt:
        raise ValueError("SYSTEM_PROMPT is empty. Set SYSTEM_PROMPT in .env.")
    system_prompt_source = "SYSTEM_PROMPT"
    lm_base_url = normalize_lm_openai_base_url(env("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"))
    return Settings(
        input_mode=input_mode,
        lm_base_url=lm_base_url,
        lm_model=env("LM_STUDIO_MODEL", "zai-org/glm-4.6v-flash"),
        lm_temperature=float(env("LM_TEMPERATURE", "0.0")),
        lm_max_tokens=lm_max_tokens,
        whisper_model=env("WHISPER_MODEL", "large-v3"),
        whisper_device=env("WHISPER_DEVICE", "auto"),
        whisper_compute_type=env("WHISPER_COMPUTE_TYPE", "default"),
        whisper_language=whisper_language,
        whisper_initial_prompt=whisper_initial_prompt,
        stt_preprocess=stt_preprocess,
        input_device_name=input_device_name,
        output_device_name=output_device_name,
        input_sample_rate=input_sample_rate,
        push_to_talk=push_to_talk,
        push_to_talk_key=push_to_talk_key,
        kokoro_repo_id=env("KOKORO_REPO_ID", "hexgrad/Kokoro-82M"),
        kokoro_lang_code=env("KOKORO_LANG_CODE", "a"),
        # Kokoro accepts comma-separated blend voices in one pass.
        kokoro_voice=env("KOKORO_VOICE", "af_bella,af_nicole"),
        kokoro_speed=float(env("KOKORO_SPEED", "1.0")),
        kokoro_split_pattern=kokoro_split_pattern,
        tts_simple_playback=tts_simple_playback,
        tts_max_chars=tts_max_chars,
        replay_all_captured_audio=replay_all_captured_audio,
        replay_all_captured_audio_max_sec=replay_all_captured_audio_max_sec,
        replay_on_stt_fail=replay_on_stt_fail,
        replay_on_stt_fail_max_sec=replay_on_stt_fail_max_sec,
        assistant_prompt=system_prompt,
        assistant_prompt_source=system_prompt_source,
        no_tts=no_tts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime local voice assistant")
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech playback",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available input and output devices and exit",
    )
    parser.add_argument(
        "--always-on",
        action="store_true",
        help="Disable push-to-talk and use voice activity detection mode",
    )
    parser.add_argument(
        "--ptt-key",
        default=None,
        help="Override push-to-talk key (examples: F8, RCTRL, SPACE)",
    )
    parser.add_argument(
        "--input-mode",
        choices=["voice", "text", "auto"],
        default=None,
        help="Input mode: 'voice' (microphone), 'text' (terminal typing), or 'auto' (both)",
    )
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    dotenv_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=dotenv_path, override=True)
    args = parse_args()
    if args.list_devices:
        list_audio_devices()
        return 0

    settings = build_settings(
        no_tts=args.no_tts,
        force_always_on=args.always_on,
        cli_ptt_key=args.ptt_key,
        cli_input_mode=args.input_mode,
    )
    print(
        f"System prompt loaded ({len(settings.assistant_prompt)} chars) from "
        f"{settings.assistant_prompt_source}"
    )
    print(f"LM Studio model: {settings.lm_model}")
    print(f"LM Studio endpoint: {settings.lm_base_url} (OpenAI-compatible)")
    output_device = resolve_output_device(settings.output_device_name)
    print(f"Input mode: {settings.input_mode}")
    print(f"Output source: {describe_audio_device(output_device, 'output')}")
    input_device: str | int | None = None
    capture_sample_rate = 0
    detector: MicrophoneTurnDetector | None = None
    stt: FasterWhisperSTT | None = None
    ptt_vk_code: int | None = None
    if settings.input_mode in {"voice", "auto"}:
        input_device = resolve_input_device(settings.input_device_name)
        print(f"Input source: {describe_audio_device(input_device, 'input')}")
        capture_sample_rate = resolve_input_sample_rate(
            input_device=input_device,
            requested_rate=settings.input_sample_rate,
            for_vad=not settings.push_to_talk,
        )
        capture_channels = resolve_input_channels(input_device=input_device)
        print(f"Input sample rate: {capture_sample_rate} Hz")
        print(f"Input channels captured: {capture_channels}")

        print(
            f"STT backend: faster-whisper model={settings.whisper_model} "
            f"device={settings.whisper_device}"
        )
        stt = FasterWhisperSTT(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            language=settings.whisper_language,
            initial_prompt=settings.whisper_initial_prompt,
            preprocess=settings.stt_preprocess,
        )
        if settings.stt_preprocess.enabled:
            print(
                "STT preprocess: "
                f"denoise={settings.stt_preprocess.denoise_strength:.2f} "
                f"highpass={settings.stt_preprocess.highpass_hz:.0f}Hz "
                f"normalize={settings.stt_preprocess.normalize_audio} "
                f"target_rms={settings.stt_preprocess.target_rms:.3f}"
            )
        else:
            print("STT preprocess: disabled")
        print(
            f"Captured-audio replay: enabled={settings.replay_all_captured_audio} "
            f"max_sec={settings.replay_all_captured_audio_max_sec:.1f} (0=full)"
        )
        print(
            f"STT fail replay: enabled={settings.replay_on_stt_fail} "
            f"max_sec={settings.replay_on_stt_fail_max_sec:.1f}"
        )
        detector = MicrophoneTurnDetector(
            input_device=input_device,
            input_channels=capture_channels,
            sample_rate=capture_sample_rate,
        )
        ptt_vk_code = resolve_vk_code(settings.push_to_talk_key) if settings.push_to_talk else None
    if settings.input_mode in {"text", "auto"}:
        print("Terminal text input enabled.")
    llm = LMStudioChat(
        settings.lm_base_url,
        settings.lm_model,
        temperature=settings.lm_temperature,
        max_tokens=settings.lm_max_tokens,
    )

    tts: KokoroTTS | None = None
    if not settings.no_tts:
        tts = KokoroTTS(
            repo_id=settings.kokoro_repo_id,
            lang_code=settings.kokoro_lang_code,
            voice=settings.kokoro_voice,
            speed=settings.kokoro_speed,
            split_pattern=settings.kokoro_split_pattern,
            device=resolve_whisper_device(settings.whisper_device),
            output_device=output_device,
            simple_playback=settings.tts_simple_playback,
        )
        print(
            f"TTS enabled: Kokoro repo={settings.kokoro_repo_id} voice={settings.kokoro_voice} "
            f"speed={settings.kokoro_speed}"
        )
        print(
            f"LLM max tokens: "
            f"{settings.lm_max_tokens if settings.lm_max_tokens is not None else 'unlimited'}"
        )
        print(f"TTS max chars: {settings.tts_max_chars}")
        print(
            f"TTS playback sample rate: {tts.output_sample_rate} Hz "
            f"(pipeline {tts.pipeline_sample_rate} Hz)"
        )
        print(
            "TTS playback mode: simple (matches tts_serial_test/kokoro_us_uk_voices)"
            if settings.tts_simple_playback
            else "TTS playback mode: safe (device-aware playback)"
        )
    else:
        print("TTS disabled (--no-tts).")

    messages: list[dict[str, str]] = [{"role": "system", "content": settings.assistant_prompt}]
    turn_index = 0
    if settings.input_mode == "voice" and settings.push_to_talk:
        print(f"Assistant ready. Hold {settings.push_to_talk_key} to talk. Say 'exit' or 'quit' to stop.")
    elif settings.input_mode == "voice":
        print("Assistant ready. Speak naturally. Say 'exit' or 'quit' to stop.")
    elif settings.input_mode == "auto":
        print(
            f"Assistant ready in auto mode. "
            f"Use voice ({settings.push_to_talk_key} hold-to-talk) or type in terminal. "
            "Say/type 'exit' or 'quit' to stop."
        )
    else:
        print("Assistant ready. Type your message and press Enter. Type 'exit' or 'quit' to stop.")

    stop_event = threading.Event()
    turn_queue: queue.Queue[tuple[str, str]] = queue.Queue()

    def capture_voice_turn() -> str | None:
        if detector is None or stt is None:
            raise RuntimeError("Voice components are not initialized.")
        if settings.push_to_talk:
            print(f"\nHold {settings.push_to_talk_key} to talk...")
            audio = detector.listen_for_turn_push_to_talk(ptt_vk_code)
        else:
            print("\nListening...")
            audio = detector.listen_for_turn()
        audio_sec = float(audio.shape[0]) / float(capture_sample_rate) if audio.size else 0.0
        if settings.replay_all_captured_audio and audio.size > 0:
            try:
                replay_limit_text = (
                    "full capture"
                    if settings.replay_all_captured_audio_max_sec <= 0
                    else f"max {settings.replay_all_captured_audio_max_sec:.1f}s"
                )
                print(f"[AUDIO] Replaying captured turn ({replay_limit_text})...")
                replay_audio_for_debug(
                    audio=audio,
                    sample_rate=capture_sample_rate,
                    output_device=output_device,
                    max_seconds=settings.replay_all_captured_audio_max_sec,
                )
                print("[AUDIO] Replay done.")
            except Exception as replay_exc:
                print(f"[AUDIO replay warning] {replay_exc}")
        stt_result = stt.transcribe_with_reason(audio, sample_rate=capture_sample_rate)
        user_text_local = stt_result.text
        if not user_text_local:
            print(
                f"[STT] No speech recognized from captured audio ({audio_sec:.2f}s). "
                f"reason={stt_result.reason}"
            )
            if settings.replay_on_stt_fail and (not settings.replay_all_captured_audio) and audio.size > 0:
                try:
                    print(
                        f"[STT] Replaying captured audio for debug "
                        f"(max {settings.replay_on_stt_fail_max_sec:.1f}s)..."
                    )
                    replay_audio_for_debug(
                        audio=audio,
                        sample_rate=capture_sample_rate,
                        output_device=output_device,
                        max_seconds=settings.replay_on_stt_fail_max_sec,
                    )
                    print("[STT] Replay done.")
                except Exception as replay_exc:
                    print(f"[STT replay warning] {replay_exc}")
            return None
        return user_text_local

    def text_input_worker() -> None:
        while not stop_event.is_set():
            try:
                typed = input("\nYou (text): ")
            except EOFError:
                turn_queue.put(("CTRL", "EOF"))
                return
            text_value = normalize_text(typed)
            if not text_value:
                print("[TEXT] Empty input. Type a message or 'exit'.")
                continue
            turn_queue.put(("TEXT", text_value))

    def voice_input_worker() -> None:
        while not stop_event.is_set():
            try:
                user_text_local = capture_voice_turn()
            except Exception as exc:
                print(f"[Voice worker warning] {exc}")
                time.sleep(0.5)
                continue
            if user_text_local:
                turn_queue.put(("STT", user_text_local))

    tts_busy = threading.Event()

    def speak_with_barge_in(spoken_text: str, current_turn: int) -> tuple[str, str] | None:
        if tts is None:
            return None
        if settings.input_mode != "auto":
            tts.speak(spoken_text)
            return None
        if tts_busy.is_set():
            print(
                f"[TURN {current_turn}] [TTS warning] Previous TTS task is still running; "
                "skipping speech for this turn."
            )
            return None

        cancel_tts = threading.Event()
        tts_done = threading.Event()
        tts_errors: list[Exception] = []

        def _tts_worker() -> None:
            try:
                tts.speak(spoken_text, cancel_event=cancel_tts)
            except Exception as exc:
                tts_errors.append(exc)
            finally:
                tts_done.set()
                tts_busy.clear()

        tts_busy.set()
        threading.Thread(target=_tts_worker, daemon=True).start()
        barged_turn: tuple[str, str] | None = None
        while not tts_done.is_set():
            try:
                queued_source, queued_text = turn_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if queued_source == "CTRL" and queued_text == "EOF":
                print("\nInput stream closed.")
                stop_event.set()
                cancel_tts.set()
                try:
                    sd.stop()
                except Exception:
                    pass
                barged_turn = (queued_source, queued_text)
                break

            barged_turn = (queued_source, queued_text)
            print(
                f"[TURN {current_turn}] [TTS] Interrupted by incoming "
                f"{queued_source} input."
            )
            cancel_tts.set()
            try:
                sd.stop()
            except Exception:
                pass
            break

        if not tts_done.wait(5.0):
            print(
                f"[TURN {current_turn}] [TTS warning] Interrupt requested but TTS is still "
                "shutting down in background."
            )
        if tts_errors:
            raise tts_errors[0]
        return barged_turn

    if settings.input_mode == "auto":
        if detector is None or stt is None:
            raise RuntimeError("Auto mode requires voice components.")
        threading.Thread(target=text_input_worker, daemon=True).start()
        threading.Thread(target=voice_input_worker, daemon=True).start()

    pending_turn: tuple[str, str] | None = None
    while True:
        try:
            source_label = "TEXT"
            source_verb = "typed"
            if settings.input_mode == "auto":
                if pending_turn is not None:
                    source_label, user_text = pending_turn
                    pending_turn = None
                else:
                    source_label, user_text = turn_queue.get()
                if source_label == "CTRL" and user_text == "EOF":
                    print("\nInput stream closed.")
                    stop_event.set()
                    return 0
                source_verb = "said" if source_label == "STT" else "typed"
            elif settings.input_mode == "voice":
                user_text = capture_voice_turn()
                if not user_text:
                    continue
                source_label = "STT"
                source_verb = "said"
            else:
                typed = input("\nYou (text): ")
                user_text = normalize_text(typed)
                if not user_text:
                    print("[TEXT] Empty input. Type a message or 'exit'.")
                    continue

            turn_index += 1
            print(f"[TURN {turn_index}] [{source_label}] You {source_verb}: {user_text}")
            if user_text.lower().strip() in {"exit", "quit", "stop"}:
                print("Stopping.")
                stop_event.set()
                return 0

            messages.append({"role": "user", "content": user_text})
            messages = trim_history(messages)
            user_count = sum(1 for msg in messages if msg.get("role") == "user")
            assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
            print(
                f"[TURN {turn_index}] [CTX] after user append: "
                f"users={user_count} assistants={assistant_count} total={len(messages)}"
            )
            print(f"[TURN {turn_index}] [LLM] Requesting completion...")
            reply = llm.complete(messages)
            if not reply:
                print(f"[TURN {turn_index}] Assistant: [No response]")
                continue

            print(f"[TURN {turn_index}] Assistant: {reply}")
            messages.append({"role": "assistant", "content": reply})
            messages = trim_history(messages)
            user_count = sum(1 for msg in messages if msg.get("role") == "user")
            assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
            print(
                f"[TURN {turn_index}] [CTX] after assistant append: "
                f"users={user_count} assistants={assistant_count} total={len(messages)}"
            )

            if tts is not None:
                try:
                    tts_start = time.perf_counter()
                    spoken_text = shrink_text_for_tts(reply, settings.tts_max_chars)
                    spoken_text = normalize_text_for_tts(spoken_text)
                    if spoken_text != reply:
                        print(
                            f"[TURN {turn_index}] [TTS] Using shortened speech text "
                            f"({len(reply)} -> {len(spoken_text)} chars)."
                        )
                    print(f"[TURN {turn_index}] [TTS] Synthesizing response...")
                    barged_turn = speak_with_barge_in(spoken_text, turn_index)
                    tts_elapsed = time.perf_counter() - tts_start
                    print(f"[TURN {turn_index}] [TTS] Done in {tts_elapsed:.2f}s")
                    if barged_turn is not None:
                        if barged_turn[0] == "CTRL" and barged_turn[1] == "EOF":
                            return 0
                        pending_turn = barged_turn
                        print(
                            f"[TURN {turn_index}] [TTS] Barge-in accepted; "
                            "processing new input now."
                        )
                except Exception as exc:
                    print(f"[TURN {turn_index}] [TTS warning] {exc}")
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            stop_event.set()
            return 0
        except EOFError:
            print("\nInput stream closed.")
            stop_event.set()
            return 0
        except Exception as exc:
            print(f"[Loop warning] {exc}")
            time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
