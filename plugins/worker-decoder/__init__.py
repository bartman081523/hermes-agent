"""worker-decoder plugin — public entry point for the Hermes plugin loader.

The hook logic lives in :mod:`worker_decoder` so it can also be exercised by
the smoke test in :mod:`dev.test_smoke` without going through Hermes. This
file is the thin ``register(ctx)`` shim the loader expects.
"""
from __future__ import annotations

import logging

from .worker_decoder import worker_decoder_callback

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Wire the worker-decoder callback into the ``transform_llm_output`` hook.

    See ``website/docs/user-guide/features/hooks.md`` § ``transform_llm_output``
    for the contract: returning ``None``/"" leaves the worker text unchanged;
    returning a non-empty string replaces it.
    """
    ctx.register_hook("transform_llm_output", worker_decoder_callback)
    logger.debug("worker-decoder: registered transform_llm_output hook")