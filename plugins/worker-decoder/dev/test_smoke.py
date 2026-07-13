"""Smoke test for the worker-decoder plugin — exercises the decoder call path
without any Hermes plugin loader / agent context.

What it does:

1. Imports :func:`worker_decoder.worker_decoder_callback` directly.
2. Mocks out the actual HTTP call to Ollama by monkey-patching
   :func:`worker_decoder._call_decoder`, so the test runs without a live
   decoder model and is fully deterministic.
3. Runs the hook three times — OK, CORRECT, REJECT — and asserts:

   * OK      → returns ``None`` (worker text unchanged)
   * CORRECT → returns the decoder's corrected text
   * REJECT  → returns the decoder's reject notice
   * Errors  → return ``None`` (fail-open, never block the agent)
   * Empty / oversized worker text → short-circuit, no decoder call
   * Disabled env var → no decoder call

Run it from the repo root::

    python -m plugins.worker-decoder.dev.test_smoke

Or directly::

    python plugins/worker-decoder/dev/test_smoke.py

Exit code is 0 on success, 1 on the first assertion failure.
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def _import_plugin_module():
    """Import the worker_decoder module with the plugin dir on sys.path.

    The plugin package's own ``__init__.py`` calls ``ctx.register_hook``
    only when Hermes invokes ``register(ctx)`` — importing the submodule
    directly bypasses that and lets us poke at private helpers.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.dirname(here)            # .../plugins/worker-decoder
    plugins_root = os.path.dirname(plugin_dir)    # .../plugins
    for path in (plugins_root, plugin_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    import worker_decoder  # type: ignore[import-not-found]
    return worker_decoder


# ---------------------------------------------------------------------------
# Mock decoder backend
# ---------------------------------------------------------------------------


class _MockDecoder:
    """Replaces ``worker_decoder._call_decoder`` for the duration of a test.

    Configure per-call replies with :meth:`set_reply`. Records every call so
    tests can assert that ``_call_decoder`` was (or wasn't) invoked.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._next_reply: Optional[dict] = None
        self._next_error: Optional[BaseException] = None
        self._fail_always = False

    def set_reply(
        self,
        raw: str = "",
        *,
        verdict: str = "",
        text: str = "",
        input_tokens: int = 7,
        output_tokens: int = 5,
    ) -> None:
        self._next_reply = {
            "verdict": verdict,
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "raw": raw,
        }
        self._next_error = None
        self._fail_always = False

    def set_error(self, exc: BaseException, *, fail_always: bool = True) -> None:
        self._next_error = exc
        self._fail_always = fail_always
        self._next_reply = None

    def __call__(self, worker_output: str):  # noqa: D401 — replaces _call_decoder
        self.calls.append({"worker_output": worker_output})
        if self._fail_always and self._next_error is not None:
            raise self._next_error
        if self._next_error is not None and not self._fail_always:
            err, self._next_error = self._next_error, None
            raise err
        assert self._next_reply is not None, "MockDecoder has no reply configured"
        reply, self._next_reply = self._next_reply, None
        return reply


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


_failures: list[str] = []


def _assert_equal(label: str, expected, actual) -> None:
    if expected != actual:
        _failures.append(
            f"{label}: expected {expected!r}, got {actual!r}"
        )


def _assert_is_none(label: str, value) -> None:
    if value is not None:
        _failures.append(f"{label}: expected None, got {value!r}")


def _assert_true(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        _failures.append(f"{label}: condition failed ({detail})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ok_verdict_returns_none(worker_decoder, mock, monkeypatch=None) -> None:
    mock.set_reply(
        raw='{"verdict": "OK", "text": "unchanged"}',
        verdict="OK",
        text="unchanged",
    )
    result = worker_decoder.worker_decoder_callback(
        response_text="Worker says 2+2=4.",
        session_id="sess-ok",
        model="minimax-m3",
        platform="cli",
    )
    _assert_is_none("OK verdict -> None", result)
    _assert_equal("OK verdict -> decoder call count", 1, len(mock.calls))


def test_correct_verdict_returns_decoder_text(worker_decoder, mock, monkeypatch=None) -> None:
    mock.set_reply(
        raw='{"verdict": "CORRECT", "text": "2+2=4 (corrected to 4)."}',
        verdict="CORRECT",
        text="2+2=4 (corrected to 4).",
    )
    result = worker_decoder.worker_decoder_callback(
        response_text="Worker says 2+2=5.",
        session_id="sess-correct",
        model="minimax-m3",
        platform="cli",
    )
    _assert_equal(
        "CORRECT verdict -> decoder text",
        "2+2=4 (corrected to 4).",
        result,
    )


def test_reject_verdict_returns_notice(worker_decoder, mock, monkeypatch=None) -> None:
    mock.set_reply(
        raw='{"verdict": "REJECT", "text": "Antwort verworfen: keine Grundlage."}',
        verdict="REJECT",
        text="Antwort verworfen: keine Grundlage.",
    )
    result = worker_decoder.worker_decoder_callback(
        response_text="Worker fabricates a citation.",
        session_id="sess-reject",
        model="minimax-m3",
        platform="cli",
    )
    _assert_equal(
        "REJECT verdict -> decoder notice",
        "Antwort verworfen: keine Grundlage.",
        result,
    )


def test_decoder_json_in_prose_still_parses(worker_decoder, mock, monkeypatch=None) -> None:
    """Some decoders wrap JSON in markdown fences or preamble. Make sure
    the regex fallback path picks the JSON object out anyway."""
    raw = (
        "Hier ist mein Urteil:\n"
        "```json\n"
        '{"verdict": "CORRECT", "text": "gekorrigiert."}\n'
        "```\n"
        "Auf Wiedersehen."
    )
    mock.set_reply(raw=raw, verdict="CORRECT", text="gekorrigiert.")
    result = worker_decoder.worker_decoder_callback(
        response_text="falsch",
        session_id="sess-prose",
        model="minimax-m3",
        platform="cli",
    )
    _assert_equal("JSON-in-prose fallback", "gekorrigiert.", result)


def test_decoder_unparseable_falls_back_to_ok(worker_decoder, mock, monkeypatch=None) -> None:
    mock.set_reply(raw="not even close to JSON", verdict="OK", text="")
    result = worker_decoder.worker_decoder_callback(
        response_text="Worker says hi.",
        session_id="sess-garbage",
        model="minimax-m3",
        platform="cli",
    )
    _assert_is_none("Unparseable decoder reply -> pass through", result)


def test_decoder_unknown_verdict_falls_back_to_ok(worker_decoder, mock, monkeypatch=None) -> None:
    mock.set_reply(
        raw='{"verdict": "MAYBE", "text": "unclear"}',
        verdict="OK",
        text="Worker says hi.",
    )
    result = worker_decoder.worker_decoder_callback(
        response_text="Worker says hi.",
        session_id="sess-unknown",
        model="minimax-m3",
        platform="cli",
    )
    _assert_is_none("Unknown verdict -> pass through", result)


def test_decoder_correct_with_empty_text_falls_back_to_ok(worker_decoder, mock, monkeypatch=None) -> None:
    """A CORRECT verdict with empty text would blank the user-visible reply —
    we degrade to OK instead."""
    mock.set_reply(
        raw='{"verdict": "CORRECT", "text": ""}',
        verdict="CORRECT",
        text="",
    )
    result = worker_decoder.worker_decoder_callback(
        response_text="Worker says hi.",
        session_id="sess-correct-empty",
        model="minimax-m3",
        platform="cli",
    )
    _assert_is_none("CORRECT + empty text -> pass through", result)


def test_decoder_http_error_passes_through(worker_decoder, mock, monkeypatch=None) -> None:
    """Network errors must never block the agent — fail-open."""
    import urllib.error

    mock.set_error(urllib.error.URLError("connection refused"), fail_always=False)
    result = worker_decoder.worker_decoder_callback(
        response_text="Worker says hi.",
        session_id="sess-neterr",
        model="minimax-m3",
        platform="cli",
    )
    _assert_is_none("Decoder HTTP error -> pass through", result)


def test_empty_worker_text_skips_decoder(worker_decoder, mock, monkeypatch=None) -> None:
    """No content, nothing to review."""
    result = worker_decoder.worker_decoder_callback(
        response_text="",
        session_id="sess-empty",
        model="minimax-m3",
        platform="cli",
    )
    _assert_is_none("Empty worker text -> pass through", result)
    _assert_equal("Empty worker text -> no decoder call", 0, len(mock.calls))


def test_whitespace_worker_text_skips_decoder(worker_decoder, mock, monkeypatch=None) -> None:
    result = worker_decoder.worker_decoder_callback(
        response_text="   \n\n\t  ",
        session_id="sess-ws",
        model="minimax-m3",
        platform="cli",
    )
    _assert_is_none("Whitespace worker text -> pass through", result)
    _assert_equal("Whitespace worker text -> no decoder call", 0, len(mock.calls))


def test_oversized_worker_text_skips_decoder(worker_decoder, mock, monkeypatch) -> None:
    """Above the cap we skip rather than paying decoder tokens on a megabyte
    of pasted log spam."""
    # Force a tiny cap for the duration of the test.
    monkeypatch.setattr(worker_decoder, "_max_chars", lambda: 16)
    result = worker_decoder.worker_decoder_callback(
        response_text="x" * 64,
        session_id="sess-huge",
        model="minimax-m3",
        platform="cli",
    )
    _assert_is_none("Oversized worker text -> pass through", result)
    _assert_equal("Oversized worker text -> no decoder call", 0, len(mock.calls))


def test_disabled_env_skips_decoder(worker_decoder, mock, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_DECODER_ENABLED", "false")
    try:
        result = worker_decoder.worker_decoder_callback(
            response_text="Worker says hi.",
            session_id="sess-disabled",
            model="minimax-m3",
            platform="cli",
        )
        _assert_is_none("Disabled -> pass through", result)
        _assert_equal("Disabled -> no decoder call", 0, len(mock.calls))
    finally:
        monkeypatch.delenv("WORKER_DECODER_ENABLED", raising=False)


def test_allowlist_filters_worker_models(worker_decoder, mock, monkeypatch) -> None:
    monkeypatch.setenv("WORKER_DECODER_ALLOW_MODELS", "minimax-m3,qwen-7b")
    try:
        # Allowed model -> decoder IS called.
        mock.set_reply(raw='{"verdict":"OK","text":"x"}', verdict="OK", text="x")
        result = worker_decoder.worker_decoder_callback(
            response_text="Worker says hi.",
            session_id="sess-allow-yes",
            model="minimax-m3",
            platform="cli",
        )
        _assert_is_none("Allowlisted model -> OK pass-through", result)
        _assert_equal("Allowlisted model -> decoder called", 1, len(mock.calls))

        # Disallowed model -> decoder is skipped.
        result = worker_decoder.worker_decoder_callback(
            response_text="Worker says hi.",
            session_id="sess-allow-no",
            model="glm-5.2:cloud",
            platform="cli",
        )
        _assert_is_none("Non-allowlisted model -> pass through", result)
        _assert_equal(
            "Non-allowlisted model -> decoder still only called once",
            1, len(mock.calls),
        )
    finally:
        monkeypatch.delenv("WORKER_DECODER_ALLOW_MODELS", raising=False)


def test_extra_kwargs_are_tolerated(worker_decoder, mock, monkeypatch=None) -> None:
    """The hook contract uses **kwargs — callers may add turn_id,
    api_request_id, etc. The callback must not crash on unknown kwargs."""
    mock.set_reply(raw='{"verdict":"OK","text":"x"}', verdict="OK", text="x")
    result = worker_decoder.worker_decoder_callback(
        response_text="Worker says hi.",
        session_id="sess-kwargs",
        model="minimax-m3",
        platform="cli",
        turn_id="turn-42",
        api_request_id="req-42",
        whatever_else=True,
    )
    _assert_is_none("Extra **kwargs -> OK pass-through", result)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


class _MonkeyPatch:
    """Tiny replacement for pytest's monkeypatch — handles setattr / setenv /
    delenv without pulling in pytest as a dependency.
    """

    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, target, name: str, value) -> None:
        previous = getattr(target, name)
        self._undo.append(("setattr", target, name, previous))
        setattr(target, name, value)

    def setenv(self, name: str, value: str) -> None:
        previous = os.environ.get(name)
        self._undo.append(("setenv", name, previous))
        os.environ[name] = value

    def delenv(self, name: str, *, raising: bool = True) -> None:
        previous = os.environ.get(name)
        if previous is None and not raising:
            return
        self._undo.append(("delenv", name, previous))
        del os.environ[name]

    def undo(self) -> None:
        while self._undo:
            entry = self._undo.pop()
            kind = entry[0]
            if kind == "setattr":
                _, target, name, previous = entry
                setattr(target, name, previous)
            elif kind == "setenv":
                _, name, previous = entry
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
            elif kind == "delenv":
                _, name, previous = entry
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous


def _run_all() -> int:
    worker_decoder = _import_plugin_module()
    mock = _MockDecoder()

    # Swap the real HTTP call for the mock.
    original_call_decoder = worker_decoder._call_decoder
    worker_decoder._call_decoder = mock  # type: ignore[assignment]
    # Silence the plugin's logger during the test — the HTTP-error path
    # deliberately triggers a logger.warning, which would otherwise show up
    # on stderr and confuse the "all green" output.
    import logging as _logging
    _logging.getLogger(worker_decoder.__name__).setLevel(_logging.CRITICAL + 1)

    monkeypatch = _MonkeyPatch()
    tests = [
        test_ok_verdict_returns_none,
        test_correct_verdict_returns_decoder_text,
        test_reject_verdict_returns_notice,
        test_decoder_json_in_prose_still_parses,
        test_decoder_unparseable_falls_back_to_ok,
        test_decoder_unknown_verdict_falls_back_to_ok,
        test_decoder_correct_with_empty_text_falls_back_to_ok,
        test_decoder_http_error_passes_through,
        test_empty_worker_text_skips_decoder,
        test_whitespace_worker_text_skips_decoder,
        test_oversized_worker_text_skips_decoder,
        test_disabled_env_skips_decoder,
        test_allowlist_filters_worker_models,
        test_extra_kwargs_are_tolerated,
    ]

    try:
        for test in tests:
            mock.calls = []
            try:
                test(worker_decoder, mock, monkeypatch)
            except Exception:
                _failures.append(
                    f"{test.__name__}: raised "
                    f"{traceback.format_exc(limit=2)}"
                )
    finally:
        worker_decoder._call_decoder = original_call_decoder  # type: ignore[assignment]
        monkeypatch.undo()

    print(f"worker-decoder smoke test: {len(tests)} cases run")
    if _failures:
        print("FAILURES:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("All assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())