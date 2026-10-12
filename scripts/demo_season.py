"""Build a small synthetic season to try `showvenist identify` (and the review TUI) on.

Everything is generated (tiny silent clips with text subtitles); no real media is involved.
Usage:  .venv/bin/python scripts/demo_season.py DIR
Then follow the commands it prints.
"""

import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showvenist.bundle.store import bundle_path, save_bundle  # noqa: E402
from tests.test_session import (  # noqa: E402
    dialogue_for,
    make_episode_mkv,
    synth_bundle,
    write_refs,
)

SLUG = "synth-show-2000"


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    bundles = out / "bundles"
    d = bundle_path(SLUG, bundles)
    d.mkdir(parents=True, exist_ok=True)
    save_bundle(write_refs(synth_bundle(), d), bundles)
    season = out / "media" / "Synth Show" / "Season 1"
    runtimes = [8, 11, 14, 17]
    layout = {  # disc -> [(file, episode index or None for no dialogue, minutes)]
        "Disc 1": [("t00.mkv", 0), ("t01.mkv", 1)],
        "Disc 2": [("t00.mkv", 2), ("t01.mkv", 3)],
    }
    for disc, files in layout.items():
        for name, k in files:
            # the last episode's subtitles are in another language: the tool cannot read them,
            # so it needs your pin
            lang = "ger" if k == 3 else "eng"
            make_episode_mkv(season / disc / name, runtimes[k], dialogue_for(k), lang=lang)
    shutil.copyfile(season / "Disc 1" / "t00.mkv", season / "Disc 2" / "t02.mkv")  # a copy
    make_episode_mkv(season / "Disc 2" / "t03.mkv", 2, dialogue_for(0)[:20])  # a short extra
    then = time.time() - 3600  # readiness: look like finished rips
    for p in season.rglob("*.mkv"):
        os.utime(p, (then, then))
    run = out / "run.sh"
    run.write_text(
        "#!/bin/sh\n# showvenist with its state, cache and config kept inside this demo folder\n"
        f'export XDG_STATE_HOME="{out}/state" XDG_CACHE_HOME="{out}/cache"\n'
        f'export XDG_DATA_HOME="{out}/data" XDG_CONFIG_HOME="{out}/config"\n'
        f'exec "{Path(sys.executable).parent / "showvenist"}" "$@"\n')
    run.chmod(0o755)
    print(f"Built {out}\n\nIn a real terminal (so the TUI opens):\n"
          f"  {run} identify '{out}/media/Synth Show' --show {SLUG} -o {out}/bundles\n\n"
          f"Add --report for the table only, or --plan for the moves. After an apply:\n"
          f"  {run} undo --last")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
