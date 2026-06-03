"""scribe CLI — transcribe any audio/video file to text using faster-whisper.

Design goals:
  * Drop a file, get `<same name>.txt` next to it.
  * Works on audio AND video (audio is extracted with ffmpeg first).
  * GPU when available, automatic CPU fallback.
  * No environment juggling: the bundled NVIDIA CUDA libraries are preloaded
    in-process, so the user never has to touch LD_LIBRARY_PATH.
  * Models are cached locally (offline after the first run).
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__

# Extensions we treat as "media we can extract audio from". Anything not listed
# is still attempted (ffmpeg is very permissive); this is only used for messages.
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".wmv", ".mpg", ".mpeg"}


# --------------------------------------------------------------------------- #
# CUDA libraries                                                              #
# --------------------------------------------------------------------------- #
def preload_cuda_libs() -> bool:
    """Preload the pip-installed NVIDIA cuBLAS/cuDNN shared libraries.

    faster-whisper's backend (ctranslate2) dlopen's these at runtime and would
    otherwise need them on LD_LIBRARY_PATH. We load them with RTLD_GLOBAL so the
    symbols are visible process-wide. Returns True if anything was loaded.
    """
    so_files: list[Path] = []
    # cublas must come before cudnn (cudnn depends on it).
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        libdir = Path(list(spec.submodule_search_locations)[0]) / "lib"
        so_files.extend(sorted(libdir.glob("*.so*")))

    if not so_files:
        return False

    # Retry loop resolves inter-library load-order dependencies.
    remaining = list(so_files)
    loaded_any = False
    for _ in range(len(so_files)):
        progress = False
        still: list[Path] = []
        for so in remaining:
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                loaded_any = True
                progress = True
            except OSError:
                still.append(so)
        remaining = still
        if not remaining or not progress:
            break
    return loaded_any


# --------------------------------------------------------------------------- #
# Audio extraction                                                            #
# --------------------------------------------------------------------------- #
def extract_audio(src: Path, dst: Path) -> None:
    """Extract/normalize audio to 16 kHz mono PCM WAV via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("error: ffmpeg introuvable. Installe-le (apt install ffmpeg).")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"error: ffmpeg a échoué sur {src.name} (code {exc.returncode}).")


# --------------------------------------------------------------------------- #
# Formatting                                                                  #
# --------------------------------------------------------------------------- #
def _hms(t: float, sep: str = ":", ms: bool = False) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    if ms:
        millis = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"
    return f"{h:02d}{sep}{m:02d}{sep}{s:02d}"


def write_txt(segments, path: Path, header: str, plain: bool) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        if not plain:
            f.write(header + "\n" + "=" * 70 + "\n\n")
        for seg in segments:
            text = seg.text.strip()
            if plain:
                f.write(text + "\n")
            else:
                f.write(f"[{_hms(seg.start)}] {text}\n")
            n += 1
    return n


def write_srt(segments, path: Path) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{_hms(seg.start, ms=True)} --> {_hms(seg.end, ms=True)}\n")
            f.write(seg.text.strip() + "\n\n")
            n += 1
    return n


def write_vtt(segments, path: Path) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = _hms(seg.start, ms=True).replace(",", ".")
            end = _hms(seg.end, ms=True).replace(",", ".")
            f.write(f"{start} --> {end}\n{seg.text.strip()}\n\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Core                                                                        #
# --------------------------------------------------------------------------- #
def pick_device(requested: str):
    """Return (device, compute_type)."""
    if requested == "cpu":
        return "cpu", "int8"
    # auto or cuda
    cuda_ok = preload_cuda_libs()
    if requested == "cuda" and not cuda_ok:
        print("warn: libs CUDA introuvables, bascule sur CPU.", file=sys.stderr)
        return "cpu", "int8"
    if requested == "cuda":
        return "cuda", "int8_float16"
    # auto: try cuda, the model load will fail-fast if unusable
    if cuda_ok:
        return "cuda", "int8_float16"
    return "cpu", "int8"


def transcribe_file(model, src: Path, args) -> Path:
    out = src.with_suffix("." + args.format)
    if out.exists() and not args.overwrite:
        print(f"skip: {out.name} existe déjà (utilise --overwrite).")
        return out

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        print(f"[{src.name}] extraction audio…", flush=True)
        extract_audio(src, wav)

        print(f"[{src.name}] transcription…", flush=True)
        segments, info = model.transcribe(
            str(wav),
            language=args.language,           # None = auto-détection
            multilingual=args.multilingual,   # gère le code-switching
            vad_filter=not args.no_vad,
            beam_size=args.beam_size,
        )
        segments = list(segments)  # consume the generator once

        header = (
            f"Transcription : {src.name}\n"
            f"Modèle : {args.model} | langue : "
            f"{args.language or 'auto (' + info.language + ')'}"
        )
        if args.format == "srt":
            n = write_srt(segments, out)
        elif args.format == "vtt":
            n = write_vtt(segments, out)
        else:
            n = write_txt(segments, out, header, args.plain)

    print(f"[{src.name}] ✓ {n} segments -> {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scribe",
        description="Transcris un fichier audio/vidéo en texte (faster-whisper, GPU auto).",
    )
    p.add_argument("inputs", nargs="+", help="Fichier(s) audio ou vidéo à transcrire.")
    p.add_argument("-m", "--model", default="large-v3",
                   help="Modèle Whisper (tiny, base, small, medium, large-v3…). Défaut: large-v3.")
    p.add_argument("-l", "--language", default=None,
                   help="Force une langue (ex: fr, en). Défaut: auto-détection.")
    p.add_argument("--multilingual", action="store_true", default=True,
                   help="Détection de langue par segment (code-switching). Activé par défaut.")
    p.add_argument("--monolingual", dest="multilingual", action="store_false",
                   help="Désactive la détection multilingue par segment.")
    p.add_argument("-d", "--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Backend de calcul. Défaut: auto.")
    p.add_argument("-f", "--format", choices=["txt", "srt", "vtt"], default="txt",
                   help="Format de sortie. Défaut: txt.")
    p.add_argument("--plain", action="store_true",
                   help="txt sans horodatage ni en-tête (texte brut).")
    p.add_argument("--beam-size", type=int, default=5, help="Beam size. Défaut: 5.")
    p.add_argument("--no-vad", action="store_true",
                   help="Désactive le filtre VAD (silences).")
    p.add_argument("--overwrite", action="store_true",
                   help="Écrase un .txt existant.")
    p.add_argument("-V", "--version", action="version", version=f"scribe {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    files = [Path(x).expanduser() for x in args.inputs]
    missing = [f for f in files if not f.is_file()]
    if missing:
        for f in missing:
            print(f"error: introuvable: {f}", file=sys.stderr)
        return 1

    device, compute_type = pick_device(args.device)
    print(f"Chargement du modèle '{args.model}' sur {device} ({compute_type})…", flush=True)

    # Imported here so --help / arg errors don't pay the import cost.
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001
        if device == "cuda":
            print(f"warn: chargement GPU impossible ({exc}); bascule CPU.", file=sys.stderr)
            model = WhisperModel(args.model, device="cpu", compute_type="int8")
        else:
            raise

    rc = 0
    for f in files:
        try:
            transcribe_file(model, f, args)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"error: échec sur {f.name}: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
