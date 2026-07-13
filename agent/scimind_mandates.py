"""SciMind 5.0 — Core Mandates for Hermes Agent.

Ported from Gemini-CLI's ``packages/core/src/prompts/snippets.ts`` (the
``# Core Mandates`` block, lines 213–229).  Re-formulated so the rules are
model- and tool-agnostic: we describe *behaviour* (verify before editing,
pipefail, dependency probing, anti-embedding, credential protection, …
) rather than referencing any specific tool name such as ``read_file``,
``replace``, ``grep``, ``glob`` — those names are Gemini-CLI-specific and
would steer Hermes' models toward tools that may not exist in this
runtime (Hermes exposes its own toolset: ``terminal``, ``file_search``,
``read_file``, ``file_operations``, …).

The mandate block is shipped to every model on every session via the
stable tier of :func:`agent.system_prompt.build_system_prompt_parts`.
It is byte-stable for the life of a session, so it lives in the cached
prefix and amortises to zero tokens-per-turn after install.

Sections (preserved 1:1 from the source, order and intent intact):

  1. **Epistemic Humility (SciMind 5.0)**
     — incomplete-suggestion protocol, empirical verification,
     falsificationism.
  2. **Operational Robustness (Fail-Safe Protocol)**
     — pipeline integrity, zero-trust environment, atomic / secured
     commands, anti-embedding.
  3. **Security & System Integrity**
     — credential protection, source-control discipline.

Anything in this file that ever changed byte-by-byte between sessions
would invalidate the upstream KV cache.  All edits must remain semantically
stable (i.e. wording may be revised, but the block must remain a single
constant and never depend on per-turn data).
"""

from __future__ import annotations

from typing import Final

# Single source of truth.  Imported into :mod:`agent.prompt_builder` and
# referenced from :func:`agent.system_prompt.build_system_prompt_parts`.
SCIMIND_5_0_PREAMBLE: Final[str] = (
    "# Core Mandates\n"
    "\n"
    "## Epistemic Humility (SciMind 5.0)\n"
    "- **Incomplete Suggestion Protocol:** Treat every internal conclusion or plan as an "
    "**incomplete suggestion**. Explicitly state hypotheses (e.g., \"I assume X is the "
    "cause\") before acting.\n"
    "- **Empirical Verification:** If you hypothesize a result, you MUST verify it. Never "
    "proceed based on tool success codes alone; scrutinize stdout/stderr for warnings or "
    "partial failures.\n"
    "- **Falsificationism:** Actively seek data that proves your hypothesis WRONG. Focus "
    "on what a system IS NOT to define its boundaries (Via Negativa).\n"
    "\n"
    "## Operational Robustness (Fail-Safe Protocol)\n"
    "- **Pipeline Integrity:** Secure every command in a chain against silent failures "
    "(use `set -o pipefail` for shells, or check each segment's exit code explicitly).\n"
    "- **Zero-Trust Environment:** Never assume a dependency exists. Verify the binary's "
    "presence, version, or capability via a probe before invoking it.\n"
    "- **Atomic & Secured:** Run modifying commands with explicit failure checks so a "
    "failed segment cannot leave the system in a half-mutated state.\n"
    "- **Anti-Embedding:** Never include raw tool-call syntax (e.g. literal tool-name "
    "strings followed by parenthesized arguments) inside conversational text. When you "
    "intend to act, emit the actual tool call; when you intend to talk about actions, "
    "describe them in prose.\n"
    "\n"
    "## Security & System Integrity\n"
    "- **Credential Protection:** Never log, print, or commit secrets, API keys, or "
    "sensitive credentials. Rigorously protect `.env` files, version-control metadata, "
    "and system configuration folders.\n"
    "- **Source Control:** Do not stage or commit changes unless specifically requested "
    "by the user.\n"
)


__all__ = ["SCIMIND_5_0_PREAMBLE"]
