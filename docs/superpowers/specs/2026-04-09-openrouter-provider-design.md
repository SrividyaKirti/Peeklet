# OpenRouter as a first-class CLI provider

**Date:** 2026-04-09
**Status:** Approved, ready for implementation plan

## Goal

Add `openrouter` as a first-class choice for `--llm-provider` in the Peeklet
CLI, alongside the existing `anthropic` and `openai` options. Users today can
already route through OpenRouter by setting `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
and reusing `OPENAI_API_KEY`, but that path is undiscoverable. This makes the
affordance explicit.

## Non-goals

- No new top-level dependency. OpenRouter is OpenAI-wire-compatible, so the
  existing `openai` SDK (already in the `[demo]` extra) is sufficient.
- No streaming, no per-request cost tracking, no model auto-selection, no
  OpenRouter-specific routing parameters (`provider`, `transforms`, etc.).
- No fallback from `OPENROUTER_API_KEY` to `OPENAI_API_KEY`. Keys stay
  distinct so the env var contract is unambiguous.
- No `--llm-base-url` generic flag. We are enumerating named providers, not
  generalizing to arbitrary OpenAI-compatible hosts.

## Architecture

OpenRouter speaks the OpenAI chat completions wire protocol, so the concrete
client is a thin variant of `OpenAIClient` that constructs the SDK with a
fixed `base_url` and a different env var. The `LLMClient` Protocol in
`src/peeklet/core/llm.py` is unchanged.

The retry + parse loop currently lives inside `OpenAIClient.pick_moments`
(see `src/peeklet/core/llm_openai.py:49-70`). With two near-identical OpenAI-
shaped clients in the tree, that loop is extracted into a single helper used
by both, so the only thing each client owns is constructing its SDK instance.

```
                  ┌────────────────────────┐
                  │  build_llm_client      │   src/peeklet/core/llm.py
                  │  (provider dispatch)   │
                  └─────┬──────┬────────┬──┘
                        │      │        │
              anthropic │  openai       │ openrouter
                        │      │        │
                  ┌─────▼──┐ ┌─▼─────┐ ┌▼──────────────┐
                  │ Anthr. │ │ OpenAI│ │ OpenRouter    │
                  │ Client │ │ Client│ │ Client        │
                  └────────┘ └───┬───┘ └──┬────────────┘
                                 │        │
                                 └────┬───┘
                                      │
                              ┌───────▼─────────┐
                              │ _call_openai_   │  shared helper:
                              │ chat_with_retry │  retry-once + parse
                              └─────────────────┘
```

## Files changed

### `src/peeklet/core/llm_openrouter.py` — NEW

- `class OpenRouterClient` implementing the same `pick_moments` signature as
  `OpenAIClient`.
- Lazy-imports `openai` exactly like `OpenAIClient` does, raising the same
  "install with: pip install peeklet[demo]" `RuntimeError` if absent.
- Constructor:
  ```python
  self._client = openai.OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=os.environ["OPENROUTER_API_KEY"],
      default_headers={
          "HTTP-Referer": "https://github.com/SrividyaKirti/Peeklet",
          "X-Title": "Peeklet",
      },
  )
  ```
- `pick_moments` delegates to the shared retry helper.

### `src/peeklet/core/llm_openai.py` — MODIFIED

- Extract the retry+parse loop into a module-level helper
  `_call_openai_chat_with_retry(client, model, system_prompt, user_message,
  video_duration, *, provider_label)` that:
  - Calls `client.chat.completions.create(...)` once.
  - Parses with `_parse_moments_json`.
  - On `LLMResponseError`, logs `"%s returned unparseable JSON, retrying once"`
    using `provider_label`, retries once, then re-raises.
- `OpenAIClient.pick_moments` becomes a thin wrapper that calls the helper
  with `provider_label="OpenAI"`.
- `OpenRouterClient.pick_moments` calls the same helper with
  `provider_label="OpenRouter"`.

The helper lives in `llm_openai.py` (not `llm.py`) so `llm.py` stays free of
`openai` SDK references and continues to lazy-import the concrete clients.
`llm_openrouter.py` imports the helper from `llm_openai.py`.

### `src/peeklet/core/llm.py` — MODIFIED

- Add `"openrouter": "OPENROUTER_API_KEY"` to `_PROVIDER_KEYS` (currently at
  line 158).
- Add an `if provider == "openrouter":` branch in `build_llm_client` that
  lazy-imports `OpenRouterClient` and returns it.
- The existing missing-env-var check at lines 173-177 already produces a
  provider-specific error message because it reads from `_PROVIDER_KEYS` —
  no change needed there.

### `src/peeklet/cli.py` — MODIFIED

- Add `"openrouter"` to the `click.Choice` at line 93.
- Update the `--llm-provider` help text from
  `"LLM provider for --demo-mode (anthropic or openai). ..."` to list all
  three providers.

### `src/peeklet/config.py` — MODIFIED

- Extend `Literal["anthropic", "openai"]` at line 93 to
  `Literal["anthropic", "openai", "openrouter"]`.
- Default stays `"anthropic"` — no behavior change for existing users.

### `pyproject.toml` — UNCHANGED

The `[demo]` extra already pulls `openai>=1.0`, which is all OpenRouter
needs. No new dependency. No new optional extra.

### `README.md` — MODIFIED

- Brief addition to the demo-mode section showing:
  ```
  export OPENROUTER_API_KEY=sk-or-...
  peeklet video.mp4 --transcript t.srt --demo-mode \
      --llm-provider openrouter \
      --llm-model anthropic/claude-3.5-sonnet
  ```
- One sentence noting that OpenRouter model ids are namespaced
  (`<vendor>/<model>`).

## Env var contract

| Provider     | Required env var       | Notes                                  |
|--------------|------------------------|----------------------------------------|
| `anthropic`  | `ANTHROPIC_API_KEY`    | unchanged                              |
| `openai`     | `OPENAI_API_KEY`       | still honors `OPENAI_BASE_URL` override |
| `openrouter` | `OPENROUTER_API_KEY`   | base URL hardcoded; no override        |

The `openai` provider deliberately keeps the `OPENAI_BASE_URL` escape hatch
so users with non-OpenRouter compatible hosts (Ollama, Groq, vLLM, etc.) are
not regressed.

## Model ids

OpenRouter models are namespaced (`anthropic/claude-3.5-sonnet`,
`meta-llama/llama-3.1-70b-instruct`, etc.). Users pass them through
`--llm-model` unchanged. Peeklet does not validate the string; OpenRouter
returns a clear error if the model is unknown, and that error surfaces
through the existing exception path.

## Testing

### `tests/unit/test_llm_openrouter.py` — NEW

Mirrors the structure of `tests/unit/test_llm_openai.py`. Specifically:

- A test that constructing `OpenRouterClient` without `openai` installed
  raises the expected `RuntimeError` (patch `openai` to `None` at the module
  level).
- A test that `OpenRouterClient.__init__` calls `openai.OpenAI` with
  `base_url="https://openrouter.ai/api/v1"`, the expected
  `default_headers`, and the value of `OPENROUTER_API_KEY` from the env.
- A `pick_moments` happy-path test using a mocked client that returns a
  fixture JSON array; assert moments are parsed and sorted.
- A `pick_moments` retry test: first call returns invalid JSON, second
  returns valid JSON; assert one warning is logged and the second response
  is returned.
- A `pick_moments` exhausted-retry test: both calls return invalid JSON;
  assert `LLMResponseError` is raised.

### `tests/unit/test_llm.py` (or equivalent) — MODIFIED

- Add a case to whatever test exercises `build_llm_client` provider
  dispatch: `build_llm_client("openrouter", "anthropic/claude-3.5-sonnet")`
  with `OPENROUTER_API_KEY` set returns an `OpenRouterClient`.
- Add a case asserting that calling `build_llm_client("openrouter", ...)`
  with `OPENROUTER_API_KEY` unset raises `RuntimeError` with a message that
  mentions `OPENROUTER_API_KEY`.

### `tests/unit/test_llm_openai.py` — MODIFIED

- The existing tests still pass after the retry-loop extraction, but the
  retry test should be re-pointed at the new helper (or kept as-is if it
  exercises `OpenAIClient.pick_moments` end-to-end). Either is fine —
  end-to-end is preferred.

### CLI smoke test — MODIFIED OR NEW

Wherever the CLI's `--llm-provider` option is exercised, add a case showing
that `--llm-provider openrouter` is accepted by Click without error. If no
such test exists today, add a one-line `click.testing.CliRunner` invocation.

## Error handling

All error paths reuse existing infrastructure:

- Missing env var → `RuntimeError` from `build_llm_client` (centralized).
- `[demo]` extra not installed → `RuntimeError` from `OpenRouterClient.__init__`.
- Unparseable LLM JSON → `LLMResponseError` from the shared retry helper,
  with the OpenRouter raw output included for debugging.
- Network/SDK errors → propagate from the `openai` SDK unchanged.

No new exception types. No new logging streams.

## Risk

Low. The change is additive: a new provider branch, a new module, a Literal
extension, and a Choice extension. No existing code path changes behavior.
The only refactor (retry-loop extraction) is covered by existing
`OpenAIClient` tests.

## Open questions

None — both clarifying questions resolved during brainstorming:
- Attribution headers (`HTTP-Referer`, `X-Title`): **yes, include them.**
- Shared retry helper: **yes, factor it out.**
