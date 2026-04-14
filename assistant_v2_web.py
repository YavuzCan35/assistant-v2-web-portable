import argparse
import base64
import io
import json
import os
import socket
import ssl
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from dotenv import load_dotenv
from faster_whisper.audio import decode_audio
from PIL import Image, ImageOps

from assistant import (
    FasterWhisperSTT,
    KokoroTTS,
    LMStudioChat,
    build_settings,
    normalize_text,
    normalize_text_for_tts,
    resolve_output_device,
    resolve_whisper_device,
    shrink_text_for_tts,
    trim_history,
)

PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Voice Assistant v2</title>
  <style>
    :root {
      --bg: #0e1318;
      --card: #18212b;
      --text: #e9f0f7;
      --muted: #97a7b8;
      --ok: #2ecc71;
      --danger: #e74c3c;
      --btn: #3a88ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(1200px 700px at 20% -20%, #223447, var(--bg));
      color: var(--text);
      font: 16px/1.4 "Segoe UI", system-ui, sans-serif;
      min-height: 100vh;
      display: flex;
      justify-content: center;
      padding: 24px;
    }
    main { width: min(820px, 100%); }
    h1 { margin: 0 0 8px 0; font-size: 1.35rem; }
    p { margin: 0 0 16px 0; color: var(--muted); }
    .panel {
      background: color-mix(in oklab, var(--card) 92%, #000 8%);
      border: 1px solid #2a3642;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,.3);
    }
    #ptt {
      width: 100%;
      min-height: 76px;
      border: 0;
      border-radius: 12px;
      font-size: 1.1rem;
      font-weight: 700;
      color: white;
      background: linear-gradient(180deg, #5ea2ff, var(--btn));
      cursor: pointer;
      touch-action: none;
      user-select: none;
    }
    #ptt.active {
      background: linear-gradient(180deg, #ff7467, var(--danger));
    }
    .toolbar {
      margin-top: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .icon-btn {
      width: 42px;
      height: 42px;
      border-radius: 10px;
      border: 1px solid #2a3642;
      background: #121920;
      color: #dbe9f6;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      padding: 0;
    }
    .icon-btn svg {
      width: 22px;
      height: 22px;
      fill: currentColor;
    }
    .icon-btn:hover {
      border-color: #4d6782;
      background: #17212b;
    }
    .action-btn {
      min-height: 38px;
      border-radius: 10px;
      border: 1px solid #2a3642;
      background: #121920;
      color: #dbe9f6;
      cursor: pointer;
      padding: 0 12px;
      font-weight: 600;
    }
    .action-btn:hover {
      border-color: #4d6782;
      background: #17212b;
    }
    #imageStatus {
      color: var(--muted);
      font-size: 0.9rem;
      min-height: 1.2em;
    }
    #timeoutWrap {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 0.9rem;
      user-select: none;
    }
    #timeoutEnable {
      transform: scale(1.05);
    }
    #langWrap {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 0.9rem;
      user-select: none;
    }
    #langSelect {
      background: #121920;
      color: #dbe9f6;
      border: 1px solid #2a3642;
      border-radius: 8px;
      padding: 4px 8px;
    }
    #status {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.95rem;
      min-height: 1.4em;
    }
    #events {
      margin-top: 16px;
      display: grid;
      gap: 10px;
    }
    .event {
      border-radius: 10px;
      padding: 12px;
      background: #121920;
      border: 1px solid #2a3642;
      animation: fadeIn .12s ease-out;
    }
    .event .head { color: var(--muted); margin-bottom: 6px; font-size: 0.85rem; }
    .event .stt { color: #d7ebff; margin-bottom: 6px; }
    .event .ans { color: #e9ffe9; }
    .event.error { border-color: #5a2a2a; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px);} to { opacity: 1; transform: translateY(0);} }
  </style>
</head>
<body>
  <main>
    <h1>Voice Assistant v2</h1>
    <p>Hold the button to talk. Release to send.</p>
    <div class="panel">
      <button id="ptt">Hold To Talk</button>
      <div class="toolbar">
        <button id="btnCamera" class="icon-btn" title="Take Photo" aria-label="Take Photo">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M7 5h2l1.3-2h3.4L15 5h2a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V8a3 3 0 0 1 3-3Zm5 12a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9Zm0-1.8a2.7 2.7 0 1 1 0-5.4 2.7 2.7 0 0 1 0 5.4Z"/>
          </svg>
        </button>
        <button id="btnGallery" class="icon-btn" title="Choose From Gallery" aria-label="Choose From Gallery">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 4h16a2 2 0 0 1 2 2v12.5a2.5 2.5 0 0 1-2.5 2.5h-15A2.5 2.5 0 0 1 2 18.5V6a2 2 0 0 1 2-2Zm0 2v10.4l4.3-4.3a1 1 0 0 1 1.4 0l3.3 3.3 2.3-2.3a1 1 0 0 1 1.4 0l3.3 3.3V6H4Zm11 3.2a1.8 1.8 0 1 0 0-3.6 1.8 1.8 0 0 0 0 3.6Z"/>
          </svg>
        </button>
        <button id="btnNewRound" class="action-btn" title="Clear context and start fresh">
          New Round
        </button>
        <div id="imageStatus">No image queued</div>
        <label id="langWrap" title="Speech language for both Whisper STT and Kokoro TTS">
          Speech Language
          <select id="langSelect">
            <option value="en:a" selected>English</option>
            <option value="tr:tr">Turkish</option>
            <option value="de:de">German</option>
          </select>
        </label>
        <label id="timeoutWrap" title="Abort slow LLM turn after 20 seconds">
          <input id="timeoutEnable" type="checkbox" checked />
          20s timeout
        </label>
      </div>
      <input id="cameraInput" type="file" accept="image/*" capture="environment" hidden />
      <input id="galleryInput" type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" hidden />
      <div id="status">Idle</div>
      <div id="events"></div>
    </div>
  </main>
  <script>
    const ptt = document.getElementById("ptt");
    const btnCamera = document.getElementById("btnCamera");
    const btnGallery = document.getElementById("btnGallery");
    const btnNewRound = document.getElementById("btnNewRound");
    const cameraInput = document.getElementById("cameraInput");
    const galleryInput = document.getElementById("galleryInput");
    const imageStatusEl = document.getElementById("imageStatus");
    const langSelectEl = document.getElementById("langSelect");
    const timeoutEnableEl = document.getElementById("timeoutEnable");
    const statusEl = document.getElementById("status");
    const eventsEl = document.getElementById("events");
    let mediaRecorder = null;
    let stream = null;
    let chunks = [];
    let isRecording = false;
    let pointerActive = false;
    let audioCtx = null;
    let pendingImage = null;
    let queuedImageForTurn = null;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function setImageStatus(text) {
      imageStatusEl.textContent = text;
    }

    function addEvent(stt, ans, isError = false, meta = "") {
      const card = document.createElement("div");
      card.className = "event" + (isError ? " error" : "");
      const now = new Date().toLocaleTimeString();
      const metaHtml = meta ? `<div class="head">${escapeHtml(meta)}</div>` : "";
      const sttText = stt && String(stt).trim() ? String(stt) : "(none)";
      const ansText = ans && String(ans).trim() ? String(ans) : "(none)";
      const sttHtml = `<div class="stt"><b>Transcribed:</b> ${escapeHtml(sttText)}</div>`;
      const ansHtml = `<div class="ans"><b>Answer:</b> ${escapeHtml(ansText)}</div>`;
      card.innerHTML = `<div class="head">${now}</div>${metaHtml}${sttHtml}${ansHtml}`;
      eventsEl.prepend(card);
    }

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, (m) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m]));
    }

    function clearQueuedImage() {
      pendingImage = null;
      cameraInput.value = "";
      galleryInput.value = "";
      setImageStatus("No image queued");
    }

    async function blobToBase64(blob) {
      const arrayBuffer = await blob.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);
      let binary = "";
      const chunkSize = 0x8000;
      for (let i = 0; i < bytes.length; i += chunkSize) {
        const chunk = bytes.subarray(i, i + chunkSize);
        binary += String.fromCharCode(...chunk);
      }
      return btoa(binary);
    }

    async function normalizeImageBlob(file) {
      if (!file || !file.type || !file.type.startsWith("image/")) {
        throw new Error("Please select an image file.");
      }
      const allowed = new Set(["image/jpeg", "image/png"]);
      if (allowed.has(file.type)) {
        return {
          blob: file,
          mime: file.type,
          name: file.name || "image",
        };
      }

      const objectUrl = URL.createObjectURL(file);
      try {
        const img = await new Promise((resolve, reject) => {
          const element = new Image();
          element.onload = () => resolve(element);
          element.onerror = () => reject(new Error("Unsupported image format for conversion."));
          element.src = objectUrl;
        });
        const width = img.naturalWidth || img.width;
        const height = img.naturalHeight || img.height;
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          throw new Error("Image conversion context unavailable.");
        }
        ctx.drawImage(img, 0, 0);
        const convertedBlob = await new Promise((resolve) => {
          canvas.toBlob(resolve, "image/jpeg", 0.92);
        });
        if (!convertedBlob) {
          throw new Error("Image conversion failed.");
        }
        const baseName = (file.name || "image").replace(/\\.[^.]+$/, "");
        return {
          blob: convertedBlob,
          mime: "image/jpeg",
          name: `${baseName}.jpg`,
        };
      } finally {
        URL.revokeObjectURL(objectUrl);
      }
    }

    async function queueImageFromFile(file, sourceLabel) {
      if (!file) return;
      try {
        const normalized = await normalizeImageBlob(file);
        pendingImage = normalized;
        setImageStatus(`Queued image (${sourceLabel}): ${normalized.name} [${normalized.mime}]`);
      } catch (err) {
        addEvent("", "Image selection failed: " + (err?.message || err), true, "image");
      }
    }

    async function ensureMic() {
      if (stream) return stream;
      if (!window.isSecureContext) {
        throw new Error("Microphone access on phones requires HTTPS (secure context).");
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("getUserMedia unavailable (likely non-HTTPS page or unsupported browser).");
      }
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      return stream;
    }

    function stopTracks() {
      if (!stream) return;
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }

    async function ensurePlaybackContext() {
      const Ctor = window.AudioContext || window.webkitAudioContext;
      if (!Ctor) return null;
      if (!audioCtx) audioCtx = new Ctor();
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
      return audioCtx;
    }

    function base64ToArrayBuffer(base64Text) {
      const binary = atob(base64Text);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      return bytes.buffer;
    }

    async function playReturnedAudio(audioB64, mimeType) {
      if (!audioB64) return false;
      const audioBuffer = base64ToArrayBuffer(audioB64);
      try {
        const ctx = await ensurePlaybackContext();
        if (ctx) {
          const decoded = await ctx.decodeAudioData(audioBuffer.slice(0));
          const source = ctx.createBufferSource();
          source.buffer = decoded;
          source.connect(ctx.destination);
          source.start();
          return true;
        }
      } catch (err) {
        console.warn("WebAudio playback failed, trying HTMLAudio fallback.", err);
      }

      try {
        const blob = new Blob([audioBuffer], { type: mimeType || "audio/wav" });
        const url = URL.createObjectURL(blob);
        const player = new Audio(url);
        player.autoplay = true;
        player.playsInline = true;
        player.onended = () => URL.revokeObjectURL(url);
        await player.play();
        return true;
      } catch (err) {
        console.warn("HTMLAudio playback failed.", err);
        return false;
      }
    }

    async function beginRecord() {
      if (isRecording) return;
      try {
        const mic = await ensureMic();
        const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm";
        chunks = [];
        mediaRecorder = new MediaRecorder(mic, { mimeType: mime });
        mediaRecorder.ondataavailable = (ev) => {
          if (ev.data && ev.data.size > 0) chunks.push(ev.data);
        };
        await ensurePlaybackContext();
        mediaRecorder.start();
        isRecording = true;
        ptt.classList.add("active");
        setStatus("Recording...");
      } catch (err) {
        setStatus("Mic access failed.");
        addEvent("", "Microphone permission failed: " + (err?.message || err), true);
      }
    }

    async function endRecord() {
      if (!isRecording || !mediaRecorder) return;
      await new Promise((resolve) => {
        mediaRecorder.onstop = resolve;
        mediaRecorder.stop();
      });
      isRecording = false;
      ptt.classList.remove("active");
      setStatus("Sending...");
      const attachedImage = queuedImageForTurn;
      queuedImageForTurn = null;

      if (!chunks.length) {
        setStatus("No audio captured.");
        return;
      }
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
      chunks = [];
      const started = Date.now();
      setStatus("Sending... 0s");
      const tick = setInterval(() => {
        const sec = Math.max(0, Math.floor((Date.now() - started) / 1000));
        setStatus(`Sending... ${sec}s`);
      }, 1000);
      const llmTimeoutEnabled = Boolean(timeoutEnableEl && timeoutEnableEl.checked);
      const llmTimeoutSec = llmTimeoutEnabled ? 20 : 0;
      const requestTimeoutMs = llmTimeoutEnabled ? 65000 : 180000;
      const langPair = String((langSelectEl && langSelectEl.value) || "en:a");
      const langParts = langPair.split(":");
      const whisperLang = (langParts[0] || "en").trim().toLowerCase();
      const kokoroLang = (
        langParts[1] || (whisperLang === "tr" ? "tr" : (whisperLang === "de" ? "de" : "a"))
      ).trim().toLowerCase();
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), requestTimeoutMs);
      try {
        const payload = {
          audio_b64: await blobToBase64(blob),
          audio_mime: blob.type || "audio/webm",
          image_b64: "",
          image_mime: "",
          image_name: "",
          llm_timeout_enabled: llmTimeoutEnabled,
          llm_timeout_sec: llmTimeoutSec,
          whisper_language: whisperLang,
          kokoro_lang_code: kokoroLang,
        };
        if (attachedImage) {
          payload.image_b64 = await blobToBase64(attachedImage.blob);
          payload.image_mime = attachedImage.mime;
          payload.image_name = attachedImage.name;
        }
        const res = await fetch("/api/turn", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal
        });
        const data = await res.json();
        const total = Math.max(0, ((Date.now() - started) / 1000)).toFixed(1);
        const meta = data.timings ? `STT ${data.timings.stt_sec}s, LLM ${data.timings.llm_sec}s, total ${total}s` : `total ${total}s`;
        if (!res.ok) {
          setStatus("Server error.");
          const errorText = data.error || data.reason || "Unknown error";
          addEvent(data.transcript || "", errorText, true, meta);
        } else {
          if (attachedImage) {
            clearQueuedImage();
          }
          setStatus("Idle");
          addEvent(data.transcript || "", data.answer || "", false, meta);
          if (data.audio_b64) {
            const played = await playReturnedAudio(data.audio_b64, data.audio_mime || "audio/wav");
            if (!played) {
              addEvent(data.transcript || "", "Audio autoplay blocked by browser policy.", true, "audio");
            }
          }
        }
      } catch (err) {
        setStatus("Network error.");
        addEvent("", "Request failed: " + (err?.message || err), true);
      } finally {
        clearInterval(tick);
        clearTimeout(timeout);
      }
    }

    async function resetRound() {
      try {
        const res = await fetch("/api/reset", { method: "POST" });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "reset_failed");
        }
        queuedImageForTurn = null;
        clearQueuedImage();
        eventsEl.innerHTML = "";
        setStatus("Idle");
      } catch (err) {
        addEvent("", "New round failed: " + (err?.message || err), true, "reset");
      }
    }

    function pressStart(ev) {
      ev.preventDefault();
      if (pointerActive) return;
      pointerActive = true;
      queuedImageForTurn = pendingImage;
      eventsEl.innerHTML = "";
      beginRecord();
    }

    function pressEnd(ev) {
      ev.preventDefault();
      if (!pointerActive) return;
      pointerActive = false;
      endRecord();
    }

    ptt.addEventListener("pointerdown", pressStart);
    ptt.addEventListener("pointerup", pressEnd);
    ptt.addEventListener("pointercancel", pressEnd);
    ptt.addEventListener("lostpointercapture", pressEnd);
    ptt.addEventListener("contextmenu", (ev) => ev.preventDefault());
    btnCamera.addEventListener("click", () => cameraInput.click());
    btnGallery.addEventListener("click", () => galleryInput.click());
    btnNewRound.addEventListener("click", (ev) => {
      ev.preventDefault();
      resetRound();
    });
    cameraInput.addEventListener("change", (ev) => queueImageFromFile(ev.target.files?.[0], "camera"));
    galleryInput.addEventListener("change", (ev) => queueImageFromFile(ev.target.files?.[0], "gallery"));
    window.addEventListener("beforeunload", stopTracks);

    if (!window.isSecureContext) {
      setStatus("HTTPS required for microphone on mobile.");
      addEvent("", "This page is not secure. Open via HTTPS to use microphone.", true);
    }
  </script>
</body>
</html>
"""


@dataclass
class ClientConversationState:
    messages: list[dict[str, object]]
    turn_index: int = 0


def guess_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


class WebAssistantRuntime:
    def __init__(self, no_tts: bool) -> None:
        dotenv_path = Path(__file__).with_name(".env")
        load_dotenv(dotenv_path=dotenv_path, override=True)
        self.settings = build_settings(
            no_tts=no_tts,
            force_always_on=False,
            cli_ptt_key=None,
            cli_input_mode="voice",
        )
        # Keep v2 web STT pinned to large-v3; LLM model still comes from LM_STUDIO_MODEL in .env.
        self.settings.whisper_model = "large-v3"

        print(
            f"Web STT backend: faster-whisper model={self.settings.whisper_model} "
            f"device={self.settings.whisper_device}"
        )
        print(f"Web LLM model: {self.settings.lm_model}")
        print(f"LM Studio endpoint: {self.settings.lm_base_url}")

        self.stt = FasterWhisperSTT(
            model_name=self.settings.whisper_model,
            device=self.settings.whisper_device,
            compute_type=self.settings.whisper_compute_type,
            language=self.settings.whisper_language,
            initial_prompt=self.settings.whisper_initial_prompt,
            preprocess=self.settings.stt_preprocess,
        )
        self.llm = LMStudioChat(
            self.settings.lm_base_url,
            self.settings.lm_model,
            temperature=self.settings.lm_temperature,
            max_tokens=self.settings.lm_max_tokens,
        )
        self.output_device: str | int | None = None
        self.tts: KokoroTTS | None = None
        if not self.settings.no_tts:
            output_device = resolve_output_device(self.settings.output_device_name)
            self.output_device = output_device
            self.tts = KokoroTTS(
                repo_id=self.settings.kokoro_repo_id,
                lang_code=self.settings.kokoro_lang_code,
                voice=self.settings.kokoro_voice,
                speed=self.settings.kokoro_speed,
                split_pattern=self.settings.kokoro_split_pattern,
                device=resolve_whisper_device(self.settings.whisper_device),
                output_device=output_device,
                simple_playback=self.settings.tts_simple_playback,
            )
            print(
                f"Web TTS enabled: voice={self.settings.kokoro_voice} "
                f"mode={'simple' if self.settings.tts_simple_playback else 'safe'}"
            )
        else:
            print("Web TTS disabled (--no-tts).")

        self.lock = threading.Lock()
        self.client_states: dict[str, ClientConversationState] = {}
        self.active_whisper_language = ""
        self.active_kokoro_lang_code = ""
        # Web default profile: Whisper English + Kokoro American English.
        self._apply_language_profile("en", "a")

    def _new_client_state(self) -> ClientConversationState:
        return ClientConversationState(
            messages=[{"role": "system", "content": self.settings.assistant_prompt}],
            turn_index=0,
        )

    def _get_or_create_client_state(self, client_id: str) -> ClientConversationState:
        state = self.client_states.get(client_id)
        if state is None:
            state = self._new_client_state()
            self.client_states[client_id] = state
        return state

    def reset_conversation(self, client_id: str | None = None) -> None:
        with self.lock:
            if client_id:
                self.client_states[client_id] = self._new_client_state()
                print(f"[WEB] Conversation reset for client={client_id}.")
                return
            self.client_states.clear()
        print("[WEB] Conversation reset for all clients.")

    def _normalize_language_profile(
        self,
        whisper_language: str | None,
        kokoro_lang_code: str | None,
    ) -> tuple[str, str]:
        whisper_raw = normalize_text((whisper_language or "").strip().lower())
        kokoro_raw = normalize_text((kokoro_lang_code or "").strip().lower())

        whisper_norm = ""
        if whisper_raw in {"tr", "tr-tr", "turkish"}:
            whisper_norm = "tr"
        elif whisper_raw in {"de", "de-de", "german"}:
            whisper_norm = "de"
        elif whisper_raw in {"en", "en-us", "english", "english-us", "english_us"}:
            whisper_norm = "en"

        kokoro_norm = ""
        if kokoro_raw in {"tr", "turkish", "t"}:
            kokoro_norm = "tr"
        elif kokoro_raw in {"de", "de-de", "german", "g"}:
            kokoro_norm = "de"
        elif kokoro_raw in {"a", "en", "english", "american", "us"}:
            kokoro_norm = "a"

        if not whisper_norm and not kokoro_norm:
            return "en", "a"
        if not whisper_norm:
            if kokoro_norm == "tr":
                whisper_norm = "tr"
            elif kokoro_norm == "de":
                whisper_norm = "de"
            else:
                whisper_norm = "en"
        if not kokoro_norm:
            if whisper_norm == "tr":
                kokoro_norm = "tr"
            elif whisper_norm == "de":
                kokoro_norm = "de"
            else:
                kokoro_norm = "a"
        return whisper_norm, kokoro_norm

    def _apply_language_profile(
        self,
        whisper_language: str | None,
        kokoro_lang_code: str | None,
    ) -> tuple[str, str]:
        normalized_whisper, normalized_kokoro = self._normalize_language_profile(
            whisper_language,
            kokoro_lang_code,
        )

        if self.stt.language != normalized_whisper:
            self.stt.language = normalized_whisper
        self.settings.whisper_language = normalized_whisper
        self.settings.kokoro_lang_code = normalized_kokoro

        if self.tts is not None and normalized_kokoro != self.active_kokoro_lang_code:
            self.tts = KokoroTTS(
                repo_id=self.settings.kokoro_repo_id,
                lang_code=normalized_kokoro,
                voice=self.settings.kokoro_voice,
                speed=self.settings.kokoro_speed,
                split_pattern=self.settings.kokoro_split_pattern,
                device=resolve_whisper_device(self.settings.whisper_device),
                output_device=self.output_device,
                simple_playback=self.settings.tts_simple_playback,
            )
        self.active_whisper_language = normalized_whisper
        self.active_kokoro_lang_code = normalized_kokoro
        return normalized_whisper, normalized_kokoro

    def _transcribe_uploaded_audio(self, audio_bytes: bytes) -> tuple[str, str]:
        if not audio_bytes:
            return "", "empty_upload"
        if len(audio_bytes) < 1200:
            return "", f"upload_too_small ({len(audio_bytes)} bytes)"

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            # Decode uploaded webm/opus bytes to mono float32 waveform, then reuse
            # the exact STT pipeline used by assistant.py (preprocess + reason codes).
            audio = decode_audio(temp_path, sampling_rate=16000)
            stt_result = self.stt.transcribe_with_reason(audio, sample_rate=16000)
            return stt_result.text, stt_result.reason
        except Exception as exc:
            return "", f"transcribe_error: {normalize_text(str(exc))}"
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _synthesize_for_web(self, text: str) -> dict[str, object]:
        if self.tts is None or not text:
            return {}
        spoken_text = normalize_text_for_tts(shrink_text_for_tts(text, self.settings.tts_max_chars))
        if not spoken_text:
            return {}

        synth_start = time.perf_counter()
        chunks: list[np.ndarray] = []
        for result in self.tts.pipeline(
            text=spoken_text,
            voice=self.tts.voice,
            speed=self.tts.speed,
            split_pattern=self.tts.split_pattern,
        ):
            chunk_audio = result.audio
            if chunk_audio is None:
                continue
            if hasattr(chunk_audio, "detach"):
                chunk_np = chunk_audio.detach().cpu().numpy().astype(np.float32)
            else:
                chunk_np = np.asarray(chunk_audio, dtype=np.float32)
            chunks.append(chunk_np)

        if not chunks:
            return {}

        audio = np.concatenate(chunks)
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        sample_rate = int(self.tts.pipeline_sample_rate)

        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm16.tobytes())
            wav_bytes = wav_buffer.getvalue()

        synth_sec = time.perf_counter() - synth_start
        duration_sec = float(audio.shape[0]) / float(sample_rate) if sample_rate > 0 else 0.0
        print(
            f"[WEB TTS] Synthesis done in {synth_sec:.2f}s "
            f"(duration={duration_sec:.2f}s bytes={len(wav_bytes)})"
        )
        return {
            "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "audio_mime": "audio/wav",
            "audio_sample_rate": sample_rate,
            "audio_duration_sec": round(duration_sec, 3),
            "tts_sec": round(synth_sec, 3),
        }

    def _prepare_image_for_llm(
        self,
        image_bytes: bytes,
        image_mime: str | None,
    ) -> tuple[str, bytes]:
        if not image_bytes:
            raise ValueError("empty_image")

        normalized_image_mime = normalize_text((image_mime or "").lower())
        if normalized_image_mime == "image/jpg":
            normalized_image_mime = "image/jpeg"
        if normalized_image_mime and normalized_image_mime not in {"image/jpeg", "image/png"}:
            raise ValueError(f"unsupported_image_mime: {normalized_image_mime}")

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                max_side = 1024
                width, height = image.size
                if max(width, height) > max_side:
                    scale = max_side / float(max(width, height))
                    new_size = (
                        max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))),
                    )
                    if hasattr(Image, "Resampling"):
                        resample_filter = Image.Resampling.LANCZOS
                    else:
                        resample_filter = Image.LANCZOS
                    image = image.resize(new_size, resample_filter)

                output = io.BytesIO()
                image.save(output, format="JPEG", quality=100, optimize=True)
                safe_bytes = output.getvalue()
        except Exception as exc:
            raise ValueError(f"image_decode_failed: {normalize_text(str(exc))}") from exc

        if not safe_bytes:
            raise ValueError("image_encode_failed")
        return "image/jpeg", safe_bytes

    def handle_turn(
        self,
        client_id: str,
        audio_bytes: bytes,
        image_bytes: bytes | None = None,
        image_mime: str | None = None,
        llm_timeout_sec: float | None = None,
        whisper_language: str | None = None,
        kokoro_lang_code: str | None = None,
    ) -> dict[str, object]:
        with self.lock:
            state = self._get_or_create_client_state(client_id)
            started = time.perf_counter()
            state.turn_index += 1
            turn = state.turn_index
            image_attached = bool(image_bytes)
            active_whisper, active_kokoro = self._apply_language_profile(
                whisper_language,
                kokoro_lang_code,
            )
            timeout_text = (
                f"{llm_timeout_sec:.1f}s"
                if llm_timeout_sec is not None and llm_timeout_sec > 0
                else "off"
            )
            print(
                f"[WEB TURN {turn}] client={client_id} Received audio bytes={len(audio_bytes)} "
                f"image={'yes' if image_attached else 'no'} "
                f"lang=whisper:{active_whisper}/kokoro:{active_kokoro} "
                f"llm_timeout={timeout_text}"
            )
            stt_start = time.perf_counter()
            transcript, reason = self._transcribe_uploaded_audio(audio_bytes)
            stt_sec = time.perf_counter() - stt_start
            if not transcript:
                total_sec = time.perf_counter() - started
                print(f"[WEB TURN {turn}] STT empty. reason={reason} stt={stt_sec:.2f}s total={total_sec:.2f}s")
                return {
                    "transcript": "",
                    "answer": "",
                    "reason": reason,
                    "error": f"No speech recognized ({reason}).",
                    "whisper_language": active_whisper,
                    "kokoro_lang_code": active_kokoro,
                    "timings": {"stt_sec": round(stt_sec, 3), "llm_sec": 0.0, "total_sec": round(total_sec, 3)},
                }

            print(f"[WEB TURN {turn}] STT: {transcript}")
            llm_messages: list[dict[str, object]]
            if image_attached:
                try:
                    normalized_image_mime, normalized_image_bytes = self._prepare_image_for_llm(
                        image_bytes or b"",
                        image_mime=image_mime,
                    )
                except ValueError as exc:
                    return {
                        "transcript": transcript,
                        "answer": "",
                        "reason": "invalid_image",
                        "error": normalize_text(str(exc)),
                        "whisper_language": active_whisper,
                        "kokoro_lang_code": active_kokoro,
                        "timings": {"stt_sec": round(stt_sec, 3), "llm_sec": 0.0, "total_sec": 0.0},
                    }
                image_data_uri = (
                    f"data:{normalized_image_mime};base64,"
                    f"{base64.b64encode(normalized_image_bytes).decode('ascii')}"
                )
                user_content: object = [
                    {"type": "text", "text": transcript},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ]
            else:
                user_content = transcript

            llm_messages = trim_history([*state.messages, {"role": "user", "content": user_content}])
            llm_start = time.perf_counter()
            answer = ""
            last_exc: Exception | None = None
            try:
                answer = self.llm.complete(
                    llm_messages,
                    request_timeout_sec=llm_timeout_sec,
                )
            except Exception as exc:
                last_exc = exc
            llm_sec = time.perf_counter() - llm_start
            if last_exc is not None:
                total_sec = time.perf_counter() - started
                error_text = normalize_text(str(last_exc))
                reason = "llm_timeout" if "timeout" in error_text.lower() else "llm_error"
                return {
                    "transcript": transcript,
                    "answer": "",
                    "reason": reason,
                    "error": error_text,
                    "image_used": image_attached,
                    "whisper_language": active_whisper,
                    "kokoro_lang_code": active_kokoro,
                    "timings": {
                        "stt_sec": round(stt_sec, 3),
                        "llm_sec": round(llm_sec, 3),
                        "total_sec": round(total_sec, 3),
                    },
                }

            # Keep textual context only. Do not retain image data URIs in history.
            state.messages.append({"role": "user", "content": transcript})
            state.messages = trim_history(state.messages)
            audio_payload: dict[str, object] = {}
            if answer:
                state.messages.append({"role": "assistant", "content": answer})
                state.messages = trim_history(state.messages)
                if self.tts is not None:
                    audio_payload = self._synthesize_for_web(answer)
            else:
                answer = ""
            total_sec = time.perf_counter() - started
            print(
                f"[WEB TURN {turn}] ANSWER: {answer if answer else '[No response]'} "
                f"(stt={stt_sec:.2f}s llm={llm_sec:.2f}s total={total_sec:.2f}s)"
            )
            return {
                "transcript": transcript,
                "answer": answer,
                "reason": "ok",
                "error": "",
                "image_used": image_attached,
                "whisper_language": active_whisper,
                "kokoro_lang_code": active_kokoro,
                "timings": {
                    "stt_sec": round(stt_sec, 3),
                    "llm_sec": round(llm_sec, 3),
                    "total_sec": round(total_sec, 3),
                },
                **audio_payload,
            }


class AssistantHTTPRequestHandler(BaseHTTPRequestHandler):
    runtime: WebAssistantRuntime | None = None

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTTPStatus.OK, PAGE_HTML)
            return
        if parsed.path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "ts": time.time()})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        client_ip = normalize_text(self.client_address[0]) or str(self.client_address[0])
        if parsed.path == "/api/reset":
            if self.runtime is None:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "runtime_not_ready"})
                return
            self.runtime.reset_conversation(client_ip)
            self._send_json(HTTPStatus.OK, {"ok": True})
            return

        if parsed.path != "/api/turn":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if self.runtime is None:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "runtime_not_ready"})
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "empty_request"})
            return
        if content_length > 28 * 1024 * 1024:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large"})
            return

        content_type = normalize_text(self.headers.get("Content-Type", "")).lower()
        if content_type.startswith("application/json"):
            try:
                payload_raw = self.rfile.read(content_length)
                payload = json.loads(payload_raw.decode("utf-8"))
            except Exception as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"invalid_json: {normalize_text(str(exc))}"},
                )
                return

            audio_b64 = str(payload.get("audio_b64", "") or "").strip()
            audio_mime = normalize_text(str(payload.get("audio_mime", "")))
            if not audio_b64:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing_audio_b64"})
                return
            try:
                audio_bytes = base64.b64decode(audio_b64, validate=True)
            except Exception:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_audio_b64"})
                return
            if not audio_bytes:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "empty_audio"})
                return
            if len(audio_bytes) > 12 * 1024 * 1024:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "audio_too_large"})
                return

            image_bytes: bytes | None = None
            image_mime = ""
            image_b64 = str(payload.get("image_b64", "") or "").strip()
            if image_b64:
                image_mime = normalize_text(str(payload.get("image_mime", ""))).lower()
                try:
                    image_bytes = base64.b64decode(image_b64, validate=True)
                except Exception:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_image_b64"})
                    return
                if not image_bytes:
                    image_bytes = None
                elif len(image_bytes) > 8 * 1024 * 1024:
                    self._send_json(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        {"error": "image_too_large"},
                    )
                    return

            timeout_enabled_raw = payload.get("llm_timeout_enabled", True)
            if isinstance(timeout_enabled_raw, bool):
                timeout_enabled = timeout_enabled_raw
            elif isinstance(timeout_enabled_raw, (int, float)):
                timeout_enabled = bool(timeout_enabled_raw)
            else:
                timeout_enabled = normalize_text(str(timeout_enabled_raw)).lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            try:
                timeout_sec_raw = float(payload.get("llm_timeout_sec", 20))
            except Exception:
                timeout_sec_raw = 20.0
            timeout_sec = max(5.0, min(120.0, timeout_sec_raw))
            llm_timeout_sec = timeout_sec if timeout_enabled else None
            whisper_language = normalize_text(str(payload.get("whisper_language", ""))).lower()
            kokoro_lang_code = normalize_text(str(payload.get("kokoro_lang_code", ""))).lower()

            print(
                f"[web] /api/turn client={client_ip} payload audio_mime={audio_mime or 'unknown'} "
                f"audio_bytes={len(audio_bytes)} image_bytes={len(image_bytes) if image_bytes else 0} "
                f"lang={whisper_language or 'en'}/{kokoro_lang_code or 'a'} "
                f"llm_timeout={'off' if llm_timeout_sec is None else f'{llm_timeout_sec:.1f}s'}"
            )
            result = self.runtime.handle_turn(
                client_ip,
                audio_bytes,
                image_bytes=image_bytes,
                image_mime=image_mime,
                llm_timeout_sec=llm_timeout_sec,
                whisper_language=whisper_language,
                kokoro_lang_code=kokoro_lang_code,
            )
        else:
            audio_bytes = self.rfile.read(content_length)
            result = self.runtime.handle_turn(
                client_ip,
                audio_bytes,
                whisper_language="en",
                kokoro_lang_code="a",
            )
        if result.get("error"):
            self._send_json(HTTPStatus.BAD_REQUEST, result)
            return
        self._send_json(HTTPStatus.OK, result)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice Assistant v2 web server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--no-tts", action="store_true", help="Disable local Kokoro playback")
    parser.add_argument(
        "--cert-file",
        default="",
        help="TLS certificate PEM path. If set, --key-file is required and server runs on HTTPS.",
    )
    parser.add_argument(
        "--key-file",
        default="",
        help="TLS private key PEM path. If set, --cert-file is required and server runs on HTTPS.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cert_file = args.cert_file.strip()
    key_file = args.key_file.strip()

    script_dir = Path(__file__).resolve().parent
    default_cert = script_dir / "certs" / "cert.pem"
    default_key = script_dir / "certs" / "key.pem"
    if not cert_file and not key_file and default_cert.exists() and default_key.exists():
        cert_file = str(default_cert)
        key_file = str(default_key)
        print(f"Auto HTTPS: using {default_cert} and {default_key}")

    if bool(cert_file) != bool(key_file):
        print("Both --cert-file and --key-file must be set together for HTTPS.")
        return 2

    runtime = WebAssistantRuntime(no_tts=args.no_tts)
    AssistantHTTPRequestHandler.runtime = runtime

    server = ThreadingHTTPServer((args.host, args.port), AssistantHTTPRequestHandler)
    scheme = "http"
    if cert_file and key_file:
        if not os.path.exists(cert_file):
            print(f"TLS cert file not found: {cert_file}")
            return 2
        if not os.path.exists(key_file):
            print(f"TLS key file not found: {key_file}")
            return 2
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"

    lan_ip = guess_lan_ip()
    print(f"Web UI: {scheme}://127.0.0.1:{args.port}")
    print(f"LAN UI: {scheme}://{lan_ip}:{args.port}")
    if scheme == "http":
        print("Phone microphones usually require HTTPS. Start with --cert-file and --key-file.")
    print("Open the LAN UI URL on your phone (same network).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
