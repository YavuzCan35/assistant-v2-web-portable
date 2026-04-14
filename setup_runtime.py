import argparse
import ipaddress
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from dotenv import load_dotenv


def collect_hostnames() -> list[str]:
    names: set[str] = {"localhost"}
    for value in {socket.gethostname(), socket.getfqdn()}:
        if not value:
            continue
        text = str(value).strip()
        if text:
            names.add(text)
    return sorted(names)


def collect_ips() -> list[str]:
    candidates: set[str] = {"127.0.0.1"}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidates.add(sock.getsockname()[0])
    except OSError:
        pass

    host = socket.gethostname()
    try:
        for family, _type, _proto, _canon, sockaddr in socket.getaddrinfo(host, None):
            if family == socket.AF_INET and sockaddr and sockaddr[0]:
                candidates.add(sockaddr[0])
    except OSError:
        pass

    normalized: list[str] = []
    for value in sorted(candidates):
        try:
            ipaddress.ip_address(value)
        except ValueError:
            continue
        normalized.append(value)
    return normalized


def generate_local_cert(bundle_dir: Path, force: bool = False) -> None:
    cert_dir = bundle_dir / "certs"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    if cert_path.exists() and key_path.exists() and not force:
        print(f"[setup] Reusing existing cert: {cert_path}")
        print(f"[setup] Reusing existing key:  {key_path}")
        return

    hostnames = collect_hostnames()
    ips = collect_ips()

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, hostnames[0] if hostnames else "localhost")]
    )

    san_entries: list[x509.GeneralName] = []
    for hostname in hostnames:
        san_entries.append(x509.DNSName(hostname))
    for raw_ip in ips:
        san_entries.append(x509.IPAddress(ipaddress.ip_address(raw_ip)))

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(private_key=key, algorithm=hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    print(f"[setup] Generated cert: {cert_path}")
    print(f"[setup] Generated key:  {key_path}")
    print(f"[setup] Certificate SAN hostnames: {', '.join(hostnames)}")
    print(f"[setup] Certificate SAN IPs: {', '.join(ips)}")


def preload_models(bundle_dir: Path) -> None:
    from assistant import KokoroTTS, FasterWhisperSTT, build_settings, resolve_whisper_device

    dotenv_path = bundle_dir / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=True)
    settings = build_settings(
        no_tts=False,
        force_always_on=False,
        cli_ptt_key=None,
        cli_input_mode="voice",
    )

    print(
        f"[setup] Preloading Whisper model={settings.whisper_model} device={settings.whisper_device}"
    )
    _stt = FasterWhisperSTT(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
        initial_prompt=settings.whisper_initial_prompt,
        preprocess=settings.stt_preprocess,
    )
    print("[setup] Whisper ready.")

    print(
        f"[setup] Preloading Kokoro repo={settings.kokoro_repo_id} voice={settings.kokoro_voice}"
    )
    _tts = KokoroTTS(
        repo_id=settings.kokoro_repo_id,
        lang_code=settings.kokoro_lang_code,
        voice=settings.kokoro_voice,
        speed=settings.kokoro_speed,
        split_pattern=settings.kokoro_split_pattern,
        device=resolve_whisper_device(settings.whisper_device),
        output_device=None,
        simple_playback=True,
    )
    print("[setup] Kokoro ready.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Assistant V2 Web portable runtime")
    parser.add_argument("--generate-certs", action="store_true", help="Create local HTTPS cert.pem/key.pem")
    parser.add_argument("--preload-models", action="store_true", help="Download and initialize Whisper/Kokoro assets")
    parser.add_argument("--force-cert", action="store_true", help="Regenerate local cert files even if they already exist")
    args = parser.parse_args()

    bundle_dir = Path(__file__).resolve().parent
    if args.generate_certs:
        generate_local_cert(bundle_dir=bundle_dir, force=args.force_cert)
    if args.preload_models:
        preload_models(bundle_dir=bundle_dir)
    if not args.generate_certs and not args.preload_models:
        print("[setup] Nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
