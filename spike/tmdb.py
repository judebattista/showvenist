#!/usr/bin/env python3
"""
Spike: Fetch episode and cast data from TMDB for four TV seasons.
PEP 723 script: uv run spike/tmdb.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.25.0",
# ]
# ///

import json
import os
import sys
import time
from pathlib import Path
import httpx

SHOWS = [
    ("Supernatural", 2, "supernatural-s02"),
    ("The Incredible Hulk", 2, "the-incredible-hulk-s02"),
    ("The West Wing", 1, "the-west-wing-s01"),
    ("The Invisible Man", 1, "the-invisible-man-s01"),
]

OUT_DIR = Path(__file__).parent / "out" / "tmdb"


def get_tmdb_auth() -> tuple[str, str]:
    """Get TMDB authentication. Returns (base_url, auth_header_value)."""
    # Check for env override first
    if key := os.environ.get("SHOWVENIST_TMDB_API_KEY"):
        return "https://api.themoviedb.org/3", f"Bearer {key}"

    # Try v4 Bearer token
    v4_path = Path.home() / ".config" / "tmdb" / "api.token"
    if v4_path.exists():
        token = v4_path.read_text().strip()
        return "https://api.themoviedb.org/3", f"Bearer {token}"

    # Fall back to v3 API key
    v3_path = Path.home() / ".config" / "tmdb" / "api.key"
    if v3_path.exists():
        key = v3_path.read_text().strip()
        return "https://api.themoviedb.org/3", key

    raise RuntimeError("No TMDB credentials found. Set SHOWVENIST_TMDB_API_KEY or create ~/.config/tmdb/api.token or api.key")


def search_tmdb(client: httpx.Client, base_url: str, show_name: str) -> dict:
    """Search TMDB for a show by name. Return the best match by exact title + year."""
    # Use query parameter for v3 or header for v4
    resp = client.get(f"{base_url}/search/tv", params={"query": show_name})
    resp.raise_for_status()
    results = resp.json().get("results", [])

    if not results:
        raise ValueError(f"No results for '{show_name}'")

    # Find best match: exact title match preferred
    best = None
    for result in results:
        if result.get("name", "").lower() == show_name.lower():
            if best is None or result.get("popularity", 0) > best.get("popularity", 0):
                best = result

    if not best:
        best = results[0]  # Fallback to first result

    return best


def fetch_season_with_credits(client: httpx.Client, base_url: str, tmdb_id: int, season_num: int) -> dict:
    """Fetch season data with credits (guest_stars, crew)."""
    resp = client.get(
        f"{base_url}/tv/{tmdb_id}/season/{season_num}",
        params={"append_to_response": "credits"}
    )
    resp.raise_for_status()
    return resp.json()


def fetch_series_credits(client: httpx.Client, base_url: str, tmdb_id: int) -> list:
    """Fetch series aggregate_credits to get cast names."""
    resp = client.get(f"{base_url}/tv/{tmdb_id}/aggregate_credits")
    resp.raise_for_status()
    data = resp.json()
    cast = data.get("cast", [])
    return [{"actor": c.get("name", ""), "character": c.get("roles", [{}])[0].get("character", "")} for c in cast]


def process_season(client: httpx.Client, base_url: str, show_name: str, season_num: int, key_name: str) -> dict:
    """Fetch and format season data for a show."""
    # Search for the show
    print(f"Searching TMDB for '{show_name}'...", file=sys.stderr)
    search_result = search_tmdb(client, base_url, show_name)
    tmdb_id = search_result["id"]
    title = search_result.get("name", "")
    year = search_result.get("first_air_date", "")[:4] if search_result.get("first_air_date") else ""
    print(f"  Found: '{title}' ({year}) - TMDB ID {tmdb_id}", file=sys.stderr)

    # Fetch season data
    print(f"Fetching season {season_num}...", file=sys.stderr)
    season_data = fetch_season_with_credits(client, base_url, tmdb_id, season_num)

    # Fetch series cast
    print(f"Fetching series cast...", file=sys.stderr)
    series_cast = fetch_series_credits(client, base_url, tmdb_id)

    # Process episodes
    episodes = []
    for ep in season_data.get("episodes", []):
        # Extract guest_stars from credits
        guest_stars = []
        if "credits" in ep:
            guest_stars = [
                {"actor": c.get("name", ""), "character": c.get("character", "")}
                for c in ep["credits"].get("guest_stars", [])
            ]

        # Extract writers and directors from credits
        writers = []
        directors = []
        if "credits" in ep:
            for crew_member in ep["credits"].get("crew", []):
                job = crew_member.get("job", "")
                name = crew_member.get("name", "")
                if job == "Writer":
                    writers.append(name)
                elif job == "Director":
                    directors.append(name)

        episodes.append({
            "id": ep.get("id"),
            "season": ep.get("season_number"),
            "episode": ep.get("episode_number"),
            "title": ep.get("name", ""),
            "overview": ep.get("overview", ""),
            "guest_stars": guest_stars,
            "writers": writers,
            "directors": directors,
        })

    return {
        "tmdb_id": tmdb_id,
        "show_title": title,
        "year": year,
        "episodes": episodes,
        "series_cast": series_cast,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_url, auth_header = get_tmdb_auth()

    # Create client with auth
    headers = {"Authorization": auth_header} if "Bearer" in auth_header else {}

    with httpx.Client(headers=headers, timeout=10.0) as client:
        if "Bearer" not in auth_header:
            # For v3 API key, it goes in params, handled by search/fetch functions
            pass

        for show_name, season_num, key_name in SHOWS:
            try:
                # Add rate limiting: be polite
                time.sleep(0.25)

                data = process_season(client, base_url, show_name, season_num, key_name)

                # Write output
                output_file = OUT_DIR / f"{key_name}.json"
                output_file.write_text(json.dumps(data, indent=2))
                print(f"Wrote {output_file.name}: {len(data['episodes'])} episodes, {len(data['series_cast'])} cast", file=sys.stderr)

            except Exception as e:
                print(f"Error processing {show_name} S{season_num:02d}: {e}", file=sys.stderr)
                sys.exit(1)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
