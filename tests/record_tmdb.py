# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.25.0"]
# ///
"""Record trimmed real TMDB responses for tests/fixtures/tmdb (not run by pytest).

Usage: uv run tests/record_tmdb.py [tmdb_id]   (default 1437, Firefly)
Key from SHOWVENIST_TMDB_API_KEY or ~/.config/tmdb/api.token / api.key. Never printed or stored.
"""

import json
import os
import re
import sys
from pathlib import Path

import httpx

BASE = "https://api.themoviedb.org/3"
OUT = Path(__file__).parent / "fixtures" / "tmdb"


def credential() -> str:
    if k := os.environ.get("SHOWVENIST_TMDB_API_KEY"):
        return k.strip()
    for n in ("api.token", "api.key"):
        p = Path.home() / ".config" / "tmdb" / n
        if p.exists():
            return p.read_text().strip()
    sys.exit("no TMDB credentials")


def trim_ep(ep: dict) -> dict:
    ep = dict(ep)
    ep["overview"] = (ep.get("overview") or "")[:160]
    ep["guest_stars"] = [
        {"name": g["name"], "character": g.get("character", "")}
        for g in (ep.get("guest_stars") or [])[:3]
    ]
    ep["crew"] = [
        {"name": c["name"], "job": c["job"]}
        for c in ep.get("crew") or []
        if c.get("job") in ("Director", "Writer", "Screenplay", "Teleplay", "Story")
    ]
    keep = (
        "id name overview air_date runtime production_code season_number episode_number "
        "order guest_stars crew"
    ).split()
    return {k: ep[k] for k in keep if k in ep}


def main() -> None:
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1437
    key = credential()
    v3 = re.fullmatch(r"[0-9a-fA-F]{32}", key) is not None
    headers = {} if v3 else {"Authorization": f"Bearer {key}"}

    def get(path: str, **params):
        if v3:
            params["api_key"] = key
        r = httpx.get(BASE + path, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    OUT.mkdir(parents=True, exist_ok=True)
    show = get(f"/tv/{tid}", append_to_response="external_ids")
    nums = [s["season_number"] for s in show["seasons"]]
    full = get(f"/tv/{tid}", append_to_response=",".join(f"season/{n}" for n in nums))
    out = {
        k: full[k]
        for k in (
            "id name first_air_date original_language in_production seasons external_ids".split()
        )
        if k in full
    }
    out["external_ids"] = {"imdb_id": (full.get("external_ids") or {}).get("imdb_id")}
    out["seasons"] = [{"season_number": s["season_number"]} for s in full["seasons"]]
    for n in nums:
        s = full[f"season/{n}"]
        out[f"season/{n}"] = {
            "season_number": n,
            "episodes": [trim_ep(e) for e in s["episodes"]],
        }
    (OUT / "show.json").write_text(json.dumps(out, indent=1))
    groups = get(f"/tv/{tid}/episode_groups")
    groups["results"] = [
        {k: g[k] for k in ("id", "name", "type") if k in g} for g in groups["results"]
    ]
    (OUT / "episode_groups.json").write_text(json.dumps(groups, indent=1))
    for g in groups["results"]:
        d = get(f"/tv/episode_group/{g['id']}")
        det = {
            "id": d["id"],
            "name": d["name"],
            "type": d.get("type"),
            "groups": [
                {
                    "name": sub["name"],
                    "order": sub["order"],
                    "episodes": [trim_ep(e) for e in sub["episodes"]],
                }
                for sub in d["groups"]
            ],
        }
        (OUT / f"group_{g['id']}.json").write_text(json.dumps(det, indent=1))
    print("recorded", tid)


main()
