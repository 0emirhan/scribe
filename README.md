# scribe

Drop an audio or video file, get a transcript `.txt` **of the same name** next to it.
Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — runs on your
GPU when available, falls back to CPU automatically, and works fully offline once the
model is cached.

```
$ scribe conference.m4a
Chargement du modèle 'large-v3' sur cuda (int8_float16)…
[conference.m4a] extraction audio…
[conference.m4a] transcription…
[conference.m4a] ✓ 412 segments -> conference.txt
```

## Features

- **Same-name output** — `talk.mp4` → `talk.txt`, right beside the source.
- **Audio *and* video** — audio is extracted with `ffmpeg` first (mp4, mkv, mov, mp3, m4a, wav, …).
- **Multilingual / code-switching** — per-segment language detection (e.g. a FR+EN recording) on by default.
- **GPU auto** — bundled NVIDIA CUDA libs via the `[gpu]` extra, no `LD_LIBRARY_PATH` to set. CPU fallback built in.
- **Formats** — `txt` (timestamped or `--plain`), plus `srt` and `vtt` subtitles.
- **Local models** — downloaded once, cached under `~/.cache/huggingface`, then offline.

## Requirements

- Python ≥ 3.9
- [`ffmpeg`](https://ffmpeg.org/) on your `PATH` (`sudo apt install ffmpeg`)
- For GPU: an NVIDIA card + recent driver (no system CUDA toolkit needed — the libs come with the `[gpu]` extra)

## Install

Recommended — global command via [pipx](https://pipx.pypa.io):

```bash
pipx install "scribe-cli[gpu]"      # GPU (NVIDIA)
pipx install scribe-cli             # CPU-only
```

From source:

```bash
git clone https://github.com/0emirhan/scribe
cd scribe
pipx install ".[gpu]"
```

## Usage

```bash
scribe recording.m4a                 # -> recording.txt (timestamped)
scribe video.mp4 --plain             # plain text, no timestamps
scribe talk.wav -f srt               # subtitles -> talk.srt
scribe a.mp3 b.mp4 c.mkv             # batch
scribe interview.m4a -l fr           # force French (skip auto-detect)
scribe lecture.mp4 -m medium         # smaller/faster model
scribe podcast.mp3 -d cpu            # force CPU
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | `large-v3` | Whisper model (`tiny`…`large-v3`). |
| `-l, --language` | auto | Force a language code (`fr`, `en`, …). |
| `--monolingual` | off | Disable per-segment language detection. |
| `-d, --device` | `auto` | `auto`, `cuda`, or `cpu`. |
| `-f, --format` | `txt` | `txt`, `srt`, or `vtt`. |
| `--plain` | off | `txt` with no timestamps/header. |
| `--beam-size` | `5` | Decoding beam size. |
| `--no-vad` | off | Disable silence (VAD) filtering. |
| `--overwrite` | off | Overwrite an existing output file. |

## Model sizes (rough)

| Model | VRAM (int8_float16) | Speed | Quality |
|-------|--------------------|-------|---------|
| `tiny` / `base` | < 1 GB | fastest | low |
| `small` / `medium` | 1–2 GB | fast | good |
| `large-v3` | ~3 GB | slower | best (recommended) |

## License

MIT — see [LICENSE](LICENSE).
