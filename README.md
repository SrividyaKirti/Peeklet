# Peeklet

Turn a screen recording + its transcript into an annotated transcript with the screenshots (and short descriptions) a downstream LLM needs.

LLMs handle images well and video poorly. Peeklet finds the distinct screens shown in a recording, lets an LLM judge which ones the spoken words actually depend on, and writes a transcript where each stretch of speech points at the screenshot that was showing. Every screenshot comes with a text description, so text-only consumers (RAG, embeddings, search, cheaper models) get the visual context too.

```
demo.mp4 ──┐
           ├──► peeklet ──► out/transcript.json
demo.vtt ──┘                out/frame_134.jpg, frame_147.jpg, ...
```

## Output

`transcript.json` is an array with one entry per block: a run of consecutive transcript lines spoken while the same screen was showing (or while no usable screen was showing).

```json
[
  {
    "timestamp": "00:02:14",
    "transcript": "Alice: If you look at the left-hand side, you'll see the policy that was triggered.\nAlice: This policy blocks the agent from creating an S3 bucket unless the team tag is present.",
    "image": "frame_134.jpg",
    "visual_context": "Policy detail page. Left panel lists the triggered policy 'require-team-tag'; right panel shows its rule: deny s3:CreateBucket unless tag 'team' is present."
  },
  {
    "timestamp": "00:02:27",
    "transcript": "Alice: And here you can see the remediation message we send back to Claude.",
    "image": "frame_147.jpg",
    "visual_context": "Claude Code terminal showing a PAWS remediation message asking for a 'team' tag before retrying.",
    "action_item": "Add team-tag check to S3 policy"
  },
  {
    "timestamp": "00:05:02",
    "transcript": "Bob: Can everyone hear me okay?",
    "image": null
  },
  {
    "timestamp": "00:08:20",
    "transcript": "Alice: Going back to the policy page…",
    "image": "frame_134.jpg",
    "visual_context": "Policy detail page. Left panel lists the triggered policy 'require-team-tag'; ..."
  }
]
```

- `timestamp` is the start of the block's first line (`HH:MM:SS`).
- `image` is a filename in the output directory, or `null`. Files are named `frame_<seconds>.jpg`. A screen that returns later repeats the same filename, and the file exists once.
- `visual_context` appears only when an image is present and the LLM step ran.
- `action_item` appears only on blocks containing a Fathom action-item checkpoint.
- With `--debug`, `debug.json` is also written (screens, signals, scores, LLM verdicts, checkpoints, warnings). It is for tuning only.

`Entries.to_claude_content()` renders the same entries as Anthropic content blocks:

```
[00:02:14]
Alice: If you look at the left-hand side, you'll see the policy that was triggered.
Alice: This policy blocks the agent from creating an S3 bucket unless the team tag is present.

[SCREENSHOT @ 00:02:14]
<image block>
Visual context: Policy detail page. ...
```

Consecutive text is merged into one block. An action item renders as `[ACTION ITEM @ HH:MM:SS] <text>` before the screenshot. A screen that appeared earlier renders as `[SCREENSHOT @ 00:08:20 — same screen as 00:02:14]` with no image block and no repeated description. Images are shrunk so the long edge is at most 1568px.

## Install

```bash
pip install peeklet
```

Peeklet uses OCR, so the Tesseract binary is a system requirement:

```bash
brew install tesseract          # macOS
sudo apt install tesseract-ocr  # Debian/Ubuntu
```

## Usage

```bash
peeklet demo.mp4 --transcript demo.vtt --out ./out \
    [--max-images 20] [--no-llm] [--llm-provider anthropic] [--llm-model claude-haiku-4-5] \
    [--config peeklet.yaml] [--debug]
```

| Flag | Meaning |
|---|---|
| `--transcript` | Required. SRT, VTT or Fathom markdown transcript. |
| `--out` | Required. Output directory for `transcript.json` and frame images. |
| `--max-images` | Maximum screenshots (default 20; 0 gives text-only output). |
| `--no-llm` | Heuristic selection only; no descriptions. |
| `--llm-provider` | `anthropic` (default), `openai` or `openrouter`. |
| `--llm-model` | Model id (default `claude-haiku-4-5`). |
| `--config` | YAML or JSON config file (see Configuration). |
| `--debug` | Also write `debug.json`. |

From Python:

```python
from peeklet import annotate

entries = annotate("demo.mp4", "demo.vtt", out_dir="./out", max_images=20)
blocks = entries.to_claude_content()   # Anthropic text/image content blocks
print(entries.stats)                   # lines, screens_found, shortlisted, kept, warnings
```

Supported transcripts are SRT, VTT and Fathom markdown (including its action items). Transcripts often chunk speech coarsely, so any line longer than `max_line_seconds` (default 8) is split at sentence boundaries; consecutive pieces on the same screen merge back into one block in the output.

## How it works

```
transcript ──► parse ──► verbal-cue checkpoints ─────────┐
transcript ──► Fathom action-item checkpoints ───────────┤
video audio ──► speech-onset checkpoints ────────────────┤
                                                         ▼
video ──► 1 fps samples ──► change detection ──► candidates (+ best frame near each checkpoint)
                                                         │
                       reject low-info ─► OCR + fingerprint ─► group into screens
                                                         │
                                     align lines to screens
                                                         │
                         heuristic score ─► shortlist (2×N) ─► LLM judge + describe
                                                         │
                                 select ≤ N ─► blocks ─► transcript.json + frames
```

**Checkpoints.** A checkpoint is a timestamp where Peeklet makes sure it has a good capture and boosts the screen on display. There are three kinds: Fathom action items, speech onsets (speech starting after at least 1.5s of silence), and verbal cues (lines that mention UI nouns such as "dashboard" or "settings", product words, or deictic phrases; e.g. "look at this dashboard"). Without Fathom markers there are no action items; without a decodable audio track a warning is recorded and there are no onsets.

**Screens.** The video is sampled at 1 fps and streamed through adaptive masking (which absorbs flickering webcam tiles), pHash and SSIM to find changes. Low-information frames (gallery views, blank loaders) are rejected. The rest are OCR'd and fingerprinted (URL, heading, sidebar text plus a header-strip pHash) and grouped into screens across the whole video, so a screen shown at 1:40 and again at 8:20 is one screen with one image and one description. Each screen keeps its best frame, the one with the most OCR words.

**Alignment.** Each transcript line links to at most one screen: the one whose occurrences overlap `[start, end + lead_seconds]` the most. If none overlap, the line has no screen. A wrong image is worse than no image.

**Scoring.** Each screen gets a heuristic score from six signals scaled to [0, 1]: references (lines linked), text overlap between those lines and the screen's OCR text, speech onsets, verbal cues, visual change, and time on screen. Screens showing at an action item get a bonus. Weights are configurable.

**LLM judge.** The top `2 × max_images` screens form a shortlist. For each one, the LLM sees the screenshot, its OCR text and up to 8 linked transcript lines, decides whether those lines need the image, and describes what is on screen. It never guesses timestamps. Results are cached on disk, so re-runs are free and identical. Cost is roughly $0.15 per video on Haiku 4.5 for a 40-screen shortlist. Use `--no-llm` (or run without an API key) for heuristic-only selection with no descriptions. If a call fails, that screen is kept with no description; if every call fails, the run proceeds as if `--no-llm` were set, with a warning.

**Budget.** At most `max_images` screens are kept: with the LLM, those judged `include`, highest score first (action-item screens are kept regardless); with `--no-llm`, the top scorers. Lines linked to screens that were not kept get `image: null`.

## Configuration

Every tunable lives in one YAML (or JSON) file passed with `--config`. Unknown keys are rejected. For example:

```yaml
# peeklet.yaml
max_images: 30
llm_model: claude-sonnet-5-5
score_weights:
  references: 0.35
```

All fields and defaults are in [`src/peeklet/config.py`](src/peeklet/config.py). CLI flags override the config file.

## LLM providers

| Provider | Environment |
|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` |
| `openai` | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` |
| `openrouter` | `OPENROUTER_API_KEY` |

If the chosen provider's credentials are missing, Peeklet warns once and continues as if `--no-llm` were set. Judgments are cached in `~/.cache/peeklet/judgments/` (`llm_cache_dir` in the config).

## Evaluation

Evaluation scripts live in `eval/` and compare Peeklet's choices against two public datasets:

```bash
uv run --with yt-dlp python -m eval.lpm_eval --limit 10     # LPM dataset
uv run python -m eval.guide_eval --limit 5 [--llm]          # GUIDE dataset
```

GUIDE is gated: accept the dataset terms on Hugging Face and provide a token via `HF_TOKEN` (or `~/.cache/huggingface/token`).

Dataset licenses: LPM is CC BY-NC-SA 4.0; GUIDE is CC BY 4.0 and gated (request access on Hugging Face first). Downloaded data and results go to `eval/data/` and `eval/results/`, which are git-ignored. Dataset content is never committed.

## Development

```bash
uv sync --dev
uv run pytest tests/unit --cov=src/peeklet
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src/peeklet/
```

Integration tests (`uv run pytest tests/integration`) need the `tesseract` binary and are skipped without it.

## License

MIT
