"""Limits and logging around the LLM learn path, for a publicly reachable demo.

- :func:`admit` is called once per LLM learn, right before the model would be
  prompted (cached library hits never get here). It enforces a per-session quota
  (``config.LLM_SESSION_LIMIT``; afterwards the user is pointed at self-hosting /
  ``config.CONTACT``) and a global sliding window shared by all users
  (``config.LLM_RATE_LIMIT`` prompts per ``config.LLM_RATE_WINDOW`` seconds).
  Sessions opened with an invite code (``config.INVITE_CODES``) skip both.
- :func:`install` wraps chebILP's model call (``structured_completion``, which the
  pipeline's ``generate_one`` looks up in ``auxiliary_generation``) so every call is
  size-checked against ``config.LLM_MAX_PROMPT_CHARS`` and appended to the JSONL
  prompt log ``config.LLM_PROMPT_LOG`` — prompt, response, cost and the session /
  client it came from (set per learn thread with :func:`context`).
- In ``config.LLM_AUTH == "api_key"`` mode it also stops chebILP from stripping
  ``ANTHROPIC_API_KEY`` from the environment, so the ``claude`` CLI bills the key.
"""

import collections
import contextlib
import datetime
import hmac
import json
import math
import os
import re
import threading
import time

from chebILP.predicate_generation import auxiliary_generation, llm_client

from . import config


class PromptTooLarge(RuntimeError):
    pass


class SlidingWindowLimiter:
    """At most ``limit`` acquisitions per ``window`` seconds (thread-safe)."""

    def __init__(self, limit: int, window: float):
        self.limit = limit
        self.window = window
        self._times: collections.deque[float] = collections.deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> float | None:
        """Take a slot and return ``None``, or return the seconds until one frees up."""
        now = time.monotonic()
        with self._lock:
            while self._times and now - self._times[0] >= self.window:
                self._times.popleft()
            if len(self._times) >= self.limit:
                return self.window - (now - self._times[0])
            self._times.append(now)
            return None


GLOBAL_LIMITER = SlidingWindowLimiter(config.LLM_RATE_LIMIT, config.LLM_RATE_WINDOW)


def session_limit_message() -> str:
    contact = f" or get in touch ({config.CONTACT})" if config.CONTACT else \
        " or get in touch"
    return (f"You have hit your session limit of {config.LLM_SESSION_LIMIT} LLM prompts per session. "
            "Thanks for using ICaRuS! If you need more, try hosting ICaRuS locally with your own LLM"
            f"{contact}. Popper, Aleph and cached LLM predictions remain available.")


def is_invite(code: str | None) -> bool:
    """Whether ``code`` is one of ``config.INVITE_CODES`` (constant-time compare)."""
    if not code:
        return False
    return any(hmac.compare_digest(code.encode(), c.encode()) for c in config.INVITE_CODES)


def admit(s) -> str | None:
    """Count one LLM prompt for session ``s``; return a user-facing refusal or ``None``.
    Invited sessions are never refused and don't use up the global window."""
    with s.lock:
        if s.invited:
            s.llm_prompts += 1
            return None
        if config.LLM_SESSION_LIMIT and s.llm_prompts >= config.LLM_SESSION_LIMIT:
            return session_limit_message()
        wait = GLOBAL_LIMITER.try_acquire()
        if wait is not None:
            return (f"You have hit a global rate limit: at most {config.LLM_RATE_LIMIT} prompts "
                    f"per {config.LLM_RATE_WINDOW // 60} minutes (across all users). Please try "
                    f"again in about {math.ceil(wait)} s (Popper and Aleph are not limited).")
        s.llm_prompts += 1
        return None


# ── prompt log ─────────────────────────────────────────────────────────────
_ctx = threading.local()
_log_lock = threading.Lock()


@contextlib.contextmanager
def context(**fields):
    """Tag model calls made by this thread (session id, client address, …) in the log."""
    _ctx.fields = fields
    try:
        yield
    finally:
        _ctx.fields = {}


@contextlib.contextmanager
def recording(exchanges: list):
    """Append each model call this thread makes (its log record) to ``exchanges``,
    so the session can offer the conversation for download."""
    _ctx.exchanges = exchanges
    try:
        yield
    finally:
        _ctx.exchanges = None


def _log(record: dict) -> None:
    exchanges = getattr(_ctx, "exchanges", None)
    if exchanges is not None:
        exchanges.append(record)
    os.makedirs(os.path.dirname(config.LLM_PROMPT_LOG), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _log_lock, open(config.LLM_PROMPT_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


class LLMTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def deadline(seconds: float):
    """Bound the model calls this thread makes to ``seconds`` of wall-clock in total."""
    _ctx.deadline = time.monotonic() + seconds
    try:
        yield
    finally:
        _ctx.deadline = None


HEARTBEAT_SECONDS = 20


@contextlib.contextmanager
def progress(on_line, concept: str | None = None, n_pos: int = 0, n_neg: int = 0):
    """Report this thread's model calls to ``on_line`` (the live learn log): what each
    call is for, a periodic "still waiting" line, and how long it took."""
    _ctx.progress = {"on_line": on_line, "concept": concept, "n_pos": n_pos,
                     "n_neg": n_neg, "calls": 0}
    try:
        yield
    finally:
        _ctx.progress = None


def _describe_call(p: dict, size: int) -> str:
    """The first call writes the predicates and the rule; the pipeline's later calls
    are its feedback round, revising what failed validation against the examples."""
    kb = f"{size / 1000:.0f}k-character prompt"
    if p["calls"] == 1:
        target = f" for “{p['concept']}”" if p["concept"] else ""
        return (f"Asking the model to propose auxiliary predicates and a rule{target} "
                f"from {p['n_pos']} positive and {p['n_neg']} negative examples "
                f"({kb})…")
    return (f"Asking the model to revise its predicates after checking them against "
            f"your examples (step {p['calls']}, {kb})…")


def expired() -> bool:
    """Whether this thread's :func:`deadline` has passed."""
    end = getattr(_ctx, "deadline", None)
    return end is not None and time.monotonic() >= end


def _call_with_deadline(model, system, prompt, schema, **kwargs):
    """Run the model call, giving up when this thread's :func:`deadline` passes.

    The call runs in a helper thread, since the CLI call itself can't be interrupted
    from outside; on timeout that thread is abandoned (daemon) and finishes on its own.
    """
    end = getattr(_ctx, "deadline", None)
    if end is None:
        return _original_completion(model, system, prompt, schema, **kwargs)
    remaining = end - time.monotonic()
    if remaining <= 0:
        raise LLMTimeout("the LLM time limit was reached")
    p = getattr(_ctx, "progress", None)
    started = time.monotonic()
    if p:
        p["calls"] += 1
        p["on_line"](_describe_call(p, len(system) + len(prompt)))
    box = {}

    def run():
        try:
            box["result"] = _original_completion(model, system, prompt, schema, **kwargs)
        except BaseException as e:  # re-raised in the caller's thread
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    while t.is_alive():
        left = end - time.monotonic()
        if left <= 0:
            if p:
                p["on_line"](f"No answer after {time.monotonic() - started:.0f} s — "
                             "the time limit was reached, giving up.")
            raise LLMTimeout("the LLM time limit was reached")
        t.join(min(HEARTBEAT_SECONDS, left))
        if t.is_alive() and p:
            p["on_line"](f"  …still waiting for the model "
                         f"({time.monotonic() - started:.0f} s, "
                         f"{max(end - time.monotonic(), 0):.0f} s left in the time limit)")
    if p:
        p["on_line"](f"Model answered after {time.monotonic() - started:.0f} s"
                     + ("." if "result" in box else " (with an error)."))
    if "error" in box:
        raise box["error"]
    if p:
        reasoning = (getattr(box["result"][0], "reasoning", "") or "").strip()
        if reasoning:
            p["on_line"](f"Model reasoning: {' '.join(reasoning.split())}")
    return box["result"]


def _guarded_completion(model, system, prompt, schema, **kwargs):
    size = len(system) + len(prompt)
    record = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
              **getattr(_ctx, "fields", {}), "model": model, "auth": config.LLM_AUTH,
              "prompt_chars": size, "system": system, "prompt": prompt}
    if size > config.LLM_MAX_PROMPT_CHARS:
        record["error"] = "prompt too large (not sent)"
        _log(record)
        raise PromptTooLarge(
            f"the prompt is too large ({size} characters, limit "
            f"{config.LLM_MAX_PROMPT_CHARS}); shorten the concept definition or use "
            "smaller example molecules")
    try:
        parsed, raw, attempts = _call_with_deadline(model, system, prompt, schema, **kwargs)
    except Exception as e:
        record.update(error=repr(e), attempts=getattr(e, "_chebilp_attempts", None))
        _log(record)
        raise
    record.update(response=raw, attempts=attempts,
                  cost_usd=sum(a.get("cost") or 0 for a in attempts))
    _log(record)
    return parsed, raw, attempts


_original_completion = auxiliary_generation.structured_completion


def install() -> None:
    """Patch chebILP's model call (idempotent)."""
    auxiliary_generation.structured_completion = _guarded_completion
    if config.LLM_AUTH == "api_key":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("llm_guard: ICARUS_LLM_AUTH=api_key but ANTHROPIC_API_KEY is not set")
        llm_client._drop_api_key_auth = lambda: None


# ── conversation export ────────────────────────────────────────────────────
def _fenced(text: str, lang: str = "") -> str:
    """``text`` in a code fence longer than any backtick run inside it."""
    longest = max((len(r) for r in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text.rstrip()}\n{fence}\n"


def _response_markdown(raw) -> str:
    """The model's structured answer, one section per field, so multi-line strings
    (reasoning, Prolog) read as text rather than escaped JSON."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return _fenced(str(raw))
    if not isinstance(data, dict):
        return _fenced(json.dumps(data, ensure_ascii=False, indent=2), "json")
    parts = []
    for key, value in data.items():
        parts.append(f"#### `{key}`\n\n")
        if isinstance(value, str):
            parts.append(_fenced(value))
        else:
            parts.append(_fenced(json.dumps(value, ensure_ascii=False, indent=2), "json"))
        parts.append("\n")
    return "".join(parts)


def conversation_markdown(runs: list[dict]) -> str:
    """A readable transcript of a session's LLM learn runs (see ``app._learn_worker``):
    per model call the system prompt, the prompt and the model's answer."""
    out = ["# ICaRuS — LLM conversation\n\n"]
    for i, run in enumerate(runs, 1):
        concept = run.get("concept") or "unnamed concept"
        out.append(f"## Run {i}: {concept}\n\n- Started: {run['started']}\n"
                   f"- Model: {run.get('model') or 'unknown'}\n"
                   f"- Examples: {run['n_pos']} positive, {run['n_neg']} negative\n")
        if run.get("outcome"):
            out.append(f"- Outcome: {run['outcome']}\n")
        out.append("\n")
        for j, ex in enumerate(run["exchanges"], 1):
            cost = f", ${ex['cost_usd']:.4f}" if ex.get("cost_usd") else ""
            out.append(f"### Call {j} ({ex.get('time', '')}{cost})\n\n")
            out.append("#### System prompt\n\n" + _fenced(ex.get("system") or "") + "\n")
            out.append("#### Prompt\n\n" + _fenced(ex.get("prompt") or "") + "\n")
            if ex.get("error"):
                out.append(f"**Error:** {ex['error']}\n\n")
            if ex.get("response") is not None:
                out.append("### Response\n\n" + _response_markdown(ex["response"]))
        if run.get("rule"):
            out.append("### Resulting rule\n\n" + _fenced(run["rule"], "prolog") + "\n")
    return "".join(out)
