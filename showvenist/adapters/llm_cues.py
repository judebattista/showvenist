"""LLM cue enrichment at ingest (spec §10.3): per-episode names, terms and memorable lines.

Uses the Batches API (half price). `submit_cues`, `poll_cues` and `collect_cues` are separate so
the CLI can wait or resume; `generate_cues` chains them. Results are stored into
`Episode.llm_cues`; this module never writes files (the caller saves the bundle). `anthropic` is
never imported here: the caller passes the client. COSTS MONEY when given a real client.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from showvenist.bundle.schema import Bundle, Episode, LlmCues
from showvenist.scoring.llm import cost_usd, usage_dict

MAX_TOKENS = 2000
BATCH_DISCOUNT = 0.5
MAX_NAMES, MAX_TERMS, MAX_LINES = 20, 20, 3

SYSTEM = """\
You help identify TV episodes from their dialogue. For one episode of a show you list cues likely to be spoken in its dialogue.
- names: character and place names that appear in this episode.
- terms: distinctive terms, objects or phrases specific to this episode.
- lines: up to 3 memorable lines of dialogue, verbatim, only if you are sure of them.
Use the synopsis and guest cast given plus what you know of the show. Omit anything you are not confident about: an empty list is better than a guess. Never invent lines."""  # noqa: E501


class CuesOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    names: list[str]
    terms: list[str]
    lines: list[str]


@dataclass
class CueReport:
    stored: list[str] = field(default_factory=list)  # episode ids
    failed: dict[str, str] = field(default_factory=dict)  # episode id -> reason
    cost_usd: float = 0.0  # batch-discounted, from usage


def custom_id(ep: Episode) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", ep.id)[:64]


def _prompt(bundle: Bundle, ep: Episode) -> str:
    show = bundle.show
    parts = [f"Show: {show.title}" + (f" ({show.year})" if show.year else ""),
             f"Episode: {ep.title}"]
    if ep.air_date:
        parts.append(f"Air date: {ep.air_date}")
    syn = " ".join(s.text for s in ep.synopses if s.text)
    parts.append(f"Synopsis: {syn or '(none)'}")
    cast = [f"{g.character} ({g.actor})" if g.character else g.actor for g in ep.guest_cast]
    parts.append(f"Guest cast: {', '.join(cast) if cast else '(none)'}")
    return "\n".join(parts)


def build_requests(
    bundle: Bundle, model: str, effort: str = "low", overwrite: bool = False
) -> list[dict]:
    reqs = []
    for ep in bundle.episodes:
        if ep.excluded or (ep.llm_cues is not None and not overwrite):
            continue
        reqs.append(
            {
                "custom_id": custom_id(ep),
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "output_config": {
                        "effort": effort,
                        "format": {"type": "json_schema", "schema": CuesOut.model_json_schema()},
                    },
                    "system": SYSTEM,
                    "messages": [{"role": "user", "content": _prompt(bundle, ep)}],
                },
            }
        )
    return reqs


def estimate_cost(bundle: Bundle, model: str, overwrite: bool = False) -> float:
    """Dry run: rough batch-discounted USD from text length. No API."""
    total = 0.0
    for r in build_requests(bundle, model, overwrite=overwrite):
        text = SYSTEM + r["params"]["messages"][0]["content"]
        usage = {"input_tokens": int(len(text) / 4) + 1, "output_tokens": 600}
        total += cost_usd(usage, model) * BATCH_DISCOUNT
    return total


def submit_cues(
    bundle: Bundle, client: Any, model: str, effort: str = "low", overwrite: bool = False
) -> str | None:
    """Create the batch; returns its id (None if there is nothing to ask)."""
    reqs = build_requests(bundle, model, effort, overwrite)
    if not reqs:
        return None
    return client.messages.batches.create(requests=reqs).id


def poll_cues(client: Any, batch_id: str) -> str:
    """processing_status: 'in_progress' | 'canceling' | 'ended'."""
    return client.messages.batches.retrieve(batch_id).processing_status


def collect_cues(
    bundle: Bundle,
    client: Any,
    batch_id: str,
    model: str,
    now: Callable[[], datetime] | None = None,
) -> CueReport:
    """Read an ended batch and store cues into the bundle's episodes (results key by custom_id)."""
    stamp = (now or (lambda: datetime.now(UTC)))().strftime("%Y-%m-%dT%H:%M:%SZ")
    by_cid = {custom_id(e): e for e in bundle.episodes}
    rep = CueReport()
    for item in client.messages.batches.results(batch_id):
        ep = by_cid.get(item.custom_id)
        if ep is None:
            continue
        res = item.result
        if res.type != "succeeded":
            rep.failed[ep.id] = res.type
            continue
        msg = res.message
        rep.cost_usd += cost_usd(usage_dict(msg.usage), model) * BATCH_DISCOUNT
        if getattr(msg, "stop_reason", None) == "refusal":
            rep.failed[ep.id] = "refusal"
            continue
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        try:
            out = CuesOut.model_validate(json.loads(text))
        except ValueError:
            rep.failed[ep.id] = "invalid"
            continue
        ep.llm_cues = LlmCues(
            names=_clean(out.names, MAX_NAMES),
            terms=_clean(out.terms, MAX_TERMS),
            lines=_clean(out.lines, MAX_LINES),
            model=model,
            generated_at=stamp,
        )
        rep.stored.append(ep.id)
    return rep


def _clean(items: list[str], limit: int) -> list[str]:
    seen, out = set(), []
    for s in items:
        s = " ".join(s.split())
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:limit]


def generate_cues(
    bundle: Bundle,
    client: Any,
    model: str,
    effort: str = "low",
    overwrite: bool = False,
    poll_interval: float = 30.0,
    timeout: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_submitted: Callable[[str], None] | None = None,
) -> CueReport:
    """Submit, wait and collect. `on_submitted(batch_id)` lets the caller persist the id so a
    killed run can resume with `poll_cues` / `collect_cues`."""
    bid = submit_cues(bundle, client, model, effort, overwrite)
    if bid is None:
        return CueReport()
    if on_submitted:
        on_submitted(bid)
    waited = 0.0
    while poll_cues(client, bid) != "ended":
        if timeout is not None and waited >= timeout:
            raise TimeoutError(f"batch {bid} still running after {waited:.0f}s; use collect_cues")
        sleep(poll_interval)
        waited += poll_interval
    return collect_cues(bundle, client, bid, model)
