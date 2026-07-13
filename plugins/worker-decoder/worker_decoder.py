"""worker-decoder — Hermes plugin for post-hoc epistemic review of worker output.

After every worker turn (the model that produced the response — by default
``minimax-m3``), this plugin calls a *decoder* model (by default ``glm-5.2:cloud``
served by Ollama at ``http://127.0.0.1:11434/v1``) and asks it to issue one
of three verdicts about the worker's final response text:

    {"verdict": "OK",      "text": "..."}   # pass through unchanged
    {"verdict": "CORRECT", "text": "..."}   # replace worker text with decoder's text
    {"verdict": "REJECT",  "text": "..."}   # replace worker text with reject notice

The hook returns ``None`` (the contract for "pass through unchanged" — see
``website/docs/user-guide/features/hooks.md`` § ``transform_llm_output``) when
the decoder says OK or the plugin is disabled, and returns a non-empty string
to replace the worker text otherwise.

The hook is intentionally narrow:

* The decoder receives only the worker's final response text — *no* tool calls,
  no conversation history, no system prompt. This keeps the cache prefix intact
  on the worker side and the decoder token cost proportional to one turn's
  answer, not the whole session.
* Every call is logged to ``/tmp/worker-decoder.log`` (timestamp, session_id,
  model, verdict, input_tokens, output_tokens) for later token-tracking.
* All knobs are environment variables so the plugin can be flipped off or
  retargeted at a different decoder without code changes.

Configuration (env vars):

    WORKER_DECODER_ENABLED   - "true"/"1"/"yes"/"on" to enable (default: true)
    WORKER_DECODER_MODEL     - decoder model name (default: "glm-5.2:cloud")
    WORKER_DECODER_URL       - decoder OpenAI-compatible endpoint
                                (default: "http://127.0.0.1:11434/v1")
    WORKER_DECODER_TIMEOUT   - HTTP timeout in seconds (default: 30)
    WORKER_DECODER_MAX_CHARS - hard cap on worker text sent to decoder
                                (default: 32000; oversized turns pass through)
    WORKER_DECODER_ALLOW_MODELS - comma-separated worker-model whitelist
                                (default: empty = all models reviewed).
                                The hook is a no-op for models not in the list.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "glm-5.2:cloud"
_DEFAULT_URL = "http://127.0.0.1:11434"
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_CHARS = 32_000
_LOG_PATH = "/tmp/worker-decoder.log"

# Worker models whose final output we want to put under decoder review.
# Empty = all worker models. Restricting the list keeps the decoder out of
# turns produced by tools like the decoder itself (when running headless) or
# by background gateway routes that shouldn't be second-guessed.
_DEFAULT_ALLOW_MODELS = ""


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or default


def _enabled() -> bool:
    return _env_bool("WORKER_DECODER_ENABLED", default=True)


def _model() -> str:
    return _env_str("WORKER_DECODER_MODEL", _DEFAULT_MODEL)


def _url() -> str:
    """Return the Ollama base URL (root, not ``/v1``).

    The plugin calls Ollama's native ``/api/chat`` endpoint rather than the
    OpenAI-compatible ``/v1/chat/completions`` because only the native endpoint
    honours the ``think: false`` option. The latter silently falls back to
    the model's default (always-on for glm-5.2), which consumes the entire
    token budget on reasoning and leaves ``content`` empty.

    The endpoint is overridable through ``WORKER_DECODER_URL``; the default
    assumes a local Ollama daemon reachable at the canonical address.
    """
    return _env_str("WORKER_DECODER_URL", _DEFAULT_URL).rstrip("/")


def _timeout() -> float:
    try:
        return float(os.environ.get("WORKER_DECODER_TIMEOUT", _DEFAULT_TIMEOUT))
    except ValueError:
        return _DEFAULT_TIMEOUT


def _max_chars() -> int:
    try:
        return int(os.environ.get("WORKER_DECODER_MAX_CHARS", _DEFAULT_MAX_CHARS))
    except ValueError:
        return _DEFAULT_MAX_CHARS


def _allow_models() -> Tuple[str, ...]:
    raw = os.environ.get("WORKER_DECODER_ALLOW_MODELS", _DEFAULT_ALLOW_MODELS).strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


# ---------------------------------------------------------------------------
# Decoder prompt
# ---------------------------------------------------------------------------

# Static system prompt. Stays small so the per-call token overhead is dominated
# by the worker output we paste in, not by instruction tokens we re-send every
# turn. The decoder is told explicitly that the only signal it gets is the
# worker text — it must not invent missing context.
_SYSTEM_PROMPT = (
    "Du bist ein epistemischer Reviewer. Du bekommst ausschließlich den "
    "finalen Antwort-Text eines Worker-LLM und sollst ihn auf sachliche "
    "Richtigkeit, logische Konsistenz und unbegründete Behauptungen prüfen.\n\n"
    "Antworte AUSSCHLIESSLICH mit genau einem JSON-Objekt in dieser Form:\n"
    '{"verdict": "OK" | "CORRECT" | "REJECT", "text": "..."}\n\n'
    "Bedeutung der Verdict-Werte:\n"
    '- "OK":      Der Worker-Output ist korrekt. Setze "text" auf den '
    "unveränderten Worker-Output.\n"
    '- "CORRECT": Der Worker-Output ist inhaltlich falsch oder irreführend. '
    "Setze \"text\" auf deine korrigierte Fassung (vollständig, nicht nur "
    "die reparierte Stelle — der User sieht nur diesen Text).\n"
    '- "REJECT":  Der Worker-Output ist so fehlerhaft, unsicher oder '
    "schädlich, dass er dem User nicht gezeigt werden sollte. Setze \"text\" "
    "auf eine kurze, ehrliche Notiz an den User, dass die Antwort verworfen "
    "wurde und warum (max. drei Sätze).\n\n"
    "Gib nichts anderes als dieses JSON aus — keine Einleitung, keine "
    "Markdown-Codeblöcke, keine Erklärung."
)

# Matches a single top-level JSON object even if the model wraps it in prose
# or ```json fences. Non-greedy so a trailing sentence after the closing brace
# does not swallow the whole response.
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _build_user_message(worker_output: str) -> str:
    return (
        "Worker-Output (final response text, kein weiterer Kontext verfügbar):\n\n"
        "<<<WORKER_OUTPUT>>>\n"
        f"{worker_output}\n"
        "<<<END_WORKER_OUTPUT>>>\n\n"
        "Prüfe den obigen Text und antworte mit dem geforderten JSON-Objekt."
    )


# ---------------------------------------------------------------------------
# Decoder HTTP call
# ---------------------------------------------------------------------------


def _call_decoder(worker_output: str) -> Dict[str, Any]:
    """POST to Ollama's native ``/api/chat`` endpoint.

    Returns a normalized dict with keys:
        verdict        - "OK" | "CORRECT" | "REJECT" (empty until parsed)
        text           - the replacement text (== worker_output on OK)
        input_tokens   - reported prompt token count, 0 if unavailable
        output_tokens  - reported completion token count, 0 if unavailable
        raw            - the decoder's raw text reply, for debugging

    Raises on transport / HTTP errors so the caller can decide whether to
    pass through or replace the worker text.

    We use the Ollama-native API (not OpenAI-compatible ``/v1/chat/completions``)
    because only the native endpoint honours ``think: false``. With the
    OpenAI-compatible endpoint, glm-5.2 always emits reasoning tokens and
    routinely consumes the entire token budget on them, leaving the visible
    ``content`` field empty — which makes the session non-replayable.
    """
    endpoint = _url() + "/api/chat"
    payload = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(worker_output)},
        ],
        # Disable the model's internal reasoning pass. We don't want any
        # "thinking" content in either the visible reply or the sidecar log —
        # the user-visible session must be replayable from the captured
        # worker text without needing the model's private scratchpad.
        "think": False,
        # Keep the answer small but generous enough that JSON fits even
        # with terse models. The worker text we paste in stays below
        # WORKER_DECODER_MAX_CHARS, so prompt_tokens are bounded too.
        "options": {
            "temperature": 0.0,
            "num_predict": 600,
        },
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=_timeout()) as response:
        raw_bytes = response.read()
    reply = json.loads(raw_bytes.decode("utf-8", errors="replace"))

    message = reply.get("message") or {}
    text = message.get("content", "") or ""
    # Ollama-native API returns token counts on the top-level, not nested
    # under ``usage`` (the OpenAI-compatible endpoint does the opposite).
    # Read from either location so the plugin works against both.
    usage_block = reply.get("usage") or {}
    prompt_tokens = (
        usage_block.get("prompt_tokens")
        or usage_block.get("prompt_eval_count")
        or reply.get("prompt_eval_count")
        or 0
    )
    completion_tokens = (
        usage_block.get("completion_tokens")
        or usage_block.get("eval_count")
        or reply.get("eval_count")
        or 0
    )
    return {
        "verdict": "",
        "text": "",
        "input_tokens": int(prompt_tokens or 0),
        "output_tokens": int(completion_tokens or 0),
        "raw": text,
    }


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


def _parse_verdict(reply_text: str, worker_output: str) -> Dict[str, str]:
    """Pull {verdict, text} out of the decoder's reply.

    Accepts either a clean JSON object or one wrapped in markdown fences /
    surrounded by prose — the regex fallback handles the latter. Unknown
    verdicts degrade to OK (pass-through) so the worker never silently loses
    its output to a confused decoder.
    """
    candidate: Optional[str] = None
    stripped = (reply_text or "").strip()
    if stripped.startswith("{"):
        candidate = stripped
    else:
        match = _JSON_OBJECT_RE.search(stripped)
        if match:
            candidate = match.group(0)
    if not candidate:
        return {"verdict": "OK", "text": worker_output}

    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return {"verdict": "OK", "text": worker_output}

    if not isinstance(parsed, dict):
        return {"verdict": "OK", "text": worker_output}

    verdict = str(parsed.get("verdict", "")).strip().upper()
    text = parsed.get("text", "")
    if not isinstance(text, str):
        text = worker_output

    if verdict not in {"OK", "CORRECT", "REJECT"}:
        verdict = "OK"

    # On OK or CORRECT, an empty text means the decoder forgot to echo the
    # worker output — fall back to the worker text so we never silently
    # blank the user-visible reply.
    if verdict == "OK" and not text:
        text = worker_output
    if verdict == "CORRECT" and not text.strip():
        verdict = "OK"
        text = worker_output
    if verdict == "REJECT" and not text.strip():
        text = "[Worker-Output vom Decoder verworfen.]"

    return {"verdict": verdict, "text": text}


# ---------------------------------------------------------------------------
# Sidecar logging
# ---------------------------------------------------------------------------


def _log_call(
    *,
    session_id: str,
    model: str,
    verdict: str,
    input_tokens: int,
    output_tokens: int,
    error: str = "",
) -> None:
    """Append one CSV-ish line to the sidecar log.

    Format: timestamp,session_id,model,verdict,input_tokens,output_tokens[,error]

    Failures are swallowed — the log is observability, not a critical path.
    """
    try:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        # Strip commas from free-text fields so a CSV-style tail-grep stays sane.
        safe_session = (session_id or "").replace(",", "_")
        safe_model = (model or "").replace(",", "_")
        safe_error = error.replace(",", "_") if error else ""
        line = (
            f"{timestamp},{safe_session},{safe_model},{verdict},"
            f"{input_tokens},{output_tokens}"
        )
        if safe_error:
            line += f",{safe_error}"
        with open(_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception as exc:  # pragma: no cover - logging never breaks the agent
        logger.debug("worker-decoder: sidecar log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Hook callback
# ---------------------------------------------------------------------------


def worker_decoder_callback(
    response_text: str,
    session_id: str,
    model: str,
    platform: str,
    **kwargs: Any,
) -> Optional[str]:
    """``transform_llm_output`` hook entry point.

    Signature mirrors ``website/docs/user-guide/features/hooks.md`` lines
    1180–1188. Returning ``None`` or "" leaves the worker text unchanged;
    returning a non-empty string replaces it. First non-empty string wins
    across plugins, so a CORRECT/REJECT verdict from us will pre-empt any
    downstream style-transform plugins (e.g. pirate-speak).
    """
    del platform  # currently unused — kept for signature parity / future routing
    del kwargs    # ``**kwargs`` swallows turn_id, api_request_id, etc.

    if not _enabled():
        return None

    if not response_text or not response_text.strip():
        # Empty / whitespace-only worker output: nothing to review.
        return None

    allow = _allow_models()
    if allow and model not in allow:
        # Worker model not on the review list (e.g. a tool-internal LLM).
        return None

    cap = _max_chars()
    if len(response_text) > cap:
        logger.info(
            "worker-decoder: worker output %d chars exceeds cap %d; passing through",
            len(response_text), cap,
        )
        return None

    try:
        decoder_reply = _call_decoder(response_text)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        # Network problems: log and pass through. Never block the agent on a
        # decoder that's offline — the worker text reaches the user unchanged.
        logger.warning("worker-decoder: decoder HTTP call failed: %s", exc)
        _log_call(
            session_id=session_id,
            model=model,
            verdict="ERROR",
            input_tokens=0,
            output_tokens=0,
            error=str(exc),
        )
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("worker-decoder: unexpected decoder error: %s", exc)
        _log_call(
            session_id=session_id,
            model=model,
            verdict="ERROR",
            input_tokens=0,
            output_tokens=0,
            error=repr(exc),
        )
        return None

    parsed = _parse_verdict(decoder_reply.get("raw", ""), response_text)
    _log_call(
        session_id=session_id,
        model=model,
        verdict=parsed["verdict"],
        input_tokens=decoder_reply.get("input_tokens", 0),
        output_tokens=decoder_reply.get("output_tokens", 0),
    )

    if parsed["verdict"] == "OK":
        return None

    # CORRECT or REJECT: replace the worker text with the decoder's version.
    return parsed["text"]