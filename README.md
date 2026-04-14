# Assistant V2 Web Portable Bundle

This folder is a self-contained copy of the files needed to run `assistant_v2_web.py`
on another Windows PC.

Included:
- `assistant_v2_web.py`
- `assistant.py`
- `.env.example`
- `requirements.txt`
- `certs/openssl.cnf`

Not included:
- local virtual environments
- model caches
- generated TLS certificate/key files
- machine-specific audio device names

Notes:
- `assistant_v2_web.py` loads `.env` from this folder.
- `setup.ps1` creates local `.env` from `.env.example` if `.env` is missing.
- It also looks for `certs/cert.pem` and `certs/key.pem` in this folder for auto HTTPS.
- Those cert files are intentionally not copied here because they should be generated per machine.
- LM Studio is still an external dependency. It must be installed separately, running, and serving the model named in `.env`.

## First Use On A New PC

1. Install LM Studio on that PC.
2. Download or load the chat model named in `.env`:
   - default: `qwen/qwen3-vl-4b`
3. Make sure LM Studio is running and its OpenAI-compatible server is enabled at:
   - `http://localhost:1234/v1`
4. Open PowerShell in this folder.
5. Run setup:

```powershell
.\setup.ps1
```

6. Start the web assistant:

```powershell
.\start.ps1
```

7. Open the printed HTTPS LAN URL on the phone or another device on the same network.
8. If microphone permission is blocked, trust the generated local certificate on that device/browser.

## Optional Configuration

- Change the model name in `.env` if the new PC uses a different LM Studio model:
  - `LM_STUDIO_MODEL=...`
- If the new PC has no working CUDA setup, keep:
  - `WHISPER_DEVICE=auto`
- If you want different Kokoro voices, change:
  - `KOKORO_VOICE=...`
- If you want to skip preloading large model downloads during setup:

```powershell
.\setup.ps1 -SkipModelPreload
```

Setup:

```powershell
.\setup.ps1
```

Start:

```powershell
.\start.ps1
```

What setup does:
- creates `.venv` if missing
- installs packages from `requirements.txt`
- generates local `certs\cert.pem` and `certs\key.pem` if missing
- optionally preloads Whisper and Kokoro downloads

Phone/browser HTTPS note:
- mobile microphone permission usually requires HTTPS
- the generated cert is local to that PC and is not meant to be committed to Git
- you may need to trust that cert on the phone/browser before microphone access works
- Android usually accepts this after installing/trusting the cert
- iPhone/iPad may require enabling full trust after installing the cert
