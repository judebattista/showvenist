# /// script
# requires-python = ">=3.11"
# dependencies = ["av>=14", "numpy", "Pillow", "pytesseract", "wordfreq"]
# ///
"""M0 spike, stages 1-2: probe, subtitle track selection, sampled-window OCR.

Spec §8.2-§8.4; output contract in spike/README.md.

    uv run spike/extract.py --probe-only FILE_OR_DIR...
    uv run spike/extract.py --show supernatural --season 2 --skip-dupes \
        /mnt/storage/tv/Supernatural/Season02 [--limit K]

Reads media only (seek + demux); never writes next to it. Output goes to
spike/out/dialogue/<show>-s<NN>/<basename>.json.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pytesseract
from PIL import Image
from wordfreq import zipf_frequency

OUT_ROOT = Path(__file__).resolve().parent / "out" / "dialogue"
LANGS = {"eng", "en"}
MIN_EVENTS = 50  # tracks with fewer packets are treated as forced/partial
WINDOW_FRACS = (0.2, 0.5, 0.8)
WINDOW_MAX_S = 300.0
WINDOW_MIN_LINES = 60
BITMAP_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle"}
TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "eia_608"}
OCR_THREADS = 8
DEBUG_DIR: Path | None = None  # --debug-images: save rendered OCR inputs here
TESS_CONFIG = "--psm 6"
DUPE_RE = re.compile(r"\d[A-Za-z]\.mkv$")
# PyAV reports decoder names; map to ffprobe's codec names.
CODEC_ALIASES = {"pgssub": "hdmv_pgs_subtitle", "dvdsub": "dvd_subtitle", "cc_dec": "eia_608"}


# --------------------------------------------------------------------------- probe

def _tag(stream, key):
    md = stream.metadata or {}
    for k, v in md.items():
        if k.lower() == key.lower():
            return v
    return None


def _plausible(st: dict) -> bool:
    return (st["type"] == "subtitle" and (st["language"] or "").lower() in LANGS
            and not st["forced"] and (st["codec"] in BITMAP_CODECS or st["codec"] in TEXT_CODECS))


def probe(path: Path, always_count: bool = False) -> dict:
    """Header-level probe plus subtitle packet counts (from mkvmerge statistics
    tags when present, else counted by demuxing the subtitle streams).

    Counting without tags is a full read of the file, so unless always_count
    it is skipped when there is at most one plausible track (events stays None
    and the few-events check doesn't apply)."""
    with av.open(str(path)) as c:
        duration = c.duration / av.time_base if c.duration else None
        streams = []
        for s in c.streams:
            disp = s.disposition
            forced = bool(disp & av.stream.Disposition.forced) if disp is not None else False
            title = _tag(s, "title")
            if title and "forced" in title.lower():
                forced = True
            nf = _tag(s, "NUMBER_OF_FRAMES")
            streams.append({
                "index": s.index,
                "type": s.type,
                "codec": (CODEC_ALIASES.get(s.codec_context.name, s.codec_context.name)
                          if s.codec_context else None),
                "language": _tag(s, "language"),
                "title": title,
                "forced": forced,
                "default": bool(disp & av.stream.Disposition.default) if disp is not None else False,
                "events": int(nf) if (s.type == "subtitle" and nf and nf.isdigit()) else None,
                "events_source": "tag" if (s.type == "subtitle" and nf) else None,
            })
        need = [st for st in streams if st["type"] == "subtitle" and st["events"] is None]
        if not always_count and sum(map(_plausible, streams)) <= 1:
            need = []
        if need:
            idx = {st["index"] for st in need}
            counts = dict.fromkeys(idx, 0)
            for pkt in c.demux([c.streams[i] for i in idx]):
                if pkt.size:
                    counts[pkt.stream.index] += 1
            for st in need:
                st["events"] = counts[st["index"]]
                st["events_source"] = "packets"
    return {"path": str(path), "size": path.stat().st_size, "duration": duration, "streams": streams}


def select_track(streams: list[dict]) -> dict | None:
    """§8.2: language match, drop forced and few-event tracks, most events wins."""
    cands = [s for s in streams if _plausible(s)
             and (s["events"] is None or s["events"] >= MIN_EVENTS)]
    if not cands:
        return None
    # Text beats bitmap only on ties; otherwise most events.
    return max(cands, key=lambda s: (s["events"] or 0, s["codec"] in TEXT_CODECS))


# --------------------------------------------------------------------------- palettes

class PgsPalette:
    """Tracks PGS palettes from PDS segments. ffmpeg's decoded rect keeps raw
    palette indices but PyAV doesn't expose the palette, so parse it ourselves."""

    def __init__(self):
        self.pals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.current = 0

    def feed(self, data: bytes) -> None:
        i = 0
        n = len(data)
        while i + 3 <= n:
            if data[i:i + 2] == b"PG":  # raw .sup framing (not used by MKV)
                i += 10
                continue
            seg, size = data[i], int.from_bytes(data[i + 1:i + 3], "big")
            body = data[i + 3:i + 3 + size]
            if seg == 0x14 and len(body) >= 2:  # PDS
                pid = body[0]
                y, a = self.pals.get(pid, (np.zeros(256, np.float32), np.zeros(256, np.float32)))
                y, a = y.copy(), a.copy()
                for j in range(2, len(body) - 4, 5):
                    eid, yy, _cr, _cb, aa = body[j:j + 5]
                    y[eid] = yy
                    a[eid] = aa
                self.pals[pid] = (y, a)
            elif seg == 0x16 and len(body) >= 10:  # PCS
                self.current = body[9]
            i += 3 + size

    def lut(self) -> np.ndarray:
        """Index -> luminance composited over black (0..255)."""
        y, a = self.pals.get(self.current) or next(iter(self.pals.values()), (None, None))
        if y is None:
            return np.arange(256, dtype=np.float32)
        yf = np.clip((y - 16) * 255 / 219, 0, 255)
        return yf * a / 255


class VobPalette:
    """VobSub: global 16-colour palette from the idx extradata, per-packet
    colour/alpha assignment from SET_COLOR/SET_CONTR control commands."""

    def __init__(self, extradata: bytes | None):
        self.luma = None
        if extradata:
            m = re.search(rb"palette:\s*([0-9a-fA-F, ]+)", extradata)
            if m:
                cols = [int(x, 16) for x in m.group(1).decode().replace(" ", "").split(",") if x]
                self.luma = np.array([0.299 * (c >> 16 & 255) + 0.587 * (c >> 8 & 255) + 0.114 * (c & 255)
                                      for c in cols[:16]] + [0.0] * (16 - len(cols[:16])), np.float32)
        self.colormap = [0, 1, 2, 3]
        self.alpha = [0, 15, 15, 15]

    def feed(self, data: bytes) -> None:
        if len(data) < 4:
            return
        ctrl = int.from_bytes(data[2:4], "big")
        seen = set()
        while ctrl + 4 <= len(data) and ctrl not in seen:
            seen.add(ctrl)
            nxt = int.from_bytes(data[ctrl + 2:ctrl + 4], "big")
            p = ctrl + 4
            while p < len(data):
                cmd = data[p]
                p += 1
                if cmd in (0x00, 0x01, 0x02):
                    continue
                if cmd == 0x03 and p + 2 <= len(data):
                    b0, b1 = data[p], data[p + 1]
                    self.colormap = [b1 & 15, b1 >> 4, b0 & 15, b0 >> 4]
                    p += 2
                elif cmd == 0x04 and p + 2 <= len(data):
                    b0, b1 = data[p], data[p + 1]
                    self.alpha = [b1 & 15, b1 >> 4, b0 & 15, b0 >> 4]
                    p += 2
                elif cmd == 0x05:
                    p += 6
                elif cmd == 0x06:
                    p += 4
                else:  # 0xff end or unknown/extended
                    break
            if nxt == ctrl:
                break
            ctrl = nxt

    def lut(self) -> np.ndarray:
        out = np.zeros(256, np.float32)
        if self.luma is None:  # no palette: assume code 1 is the fill (most common DVD layout)
            out[:4] = [0, 255, 0, 128]
            return out
        for code in range(4):
            out[code] = self.luma[self.colormap[code]] * self.alpha[code] / 15
        return out


# --------------------------------------------------------------------------- image + OCR

def render(rects: list, lut: np.ndarray, upscale: int) -> Image.Image | None:
    """Composite rects into one canvas, palette -> grayscale (bright text over
    black), upscale, binarize, invert to dark text on white with padding."""
    rects = [r for r in rects if r.width > 0 and r.height > 0 and r.planes]
    if not rects:
        return None
    x0 = min(r.x for r in rects); y0 = min(r.y for r in rects)
    x1 = max(r.x + r.width for r in rects); y1 = max(r.y + r.height for r in rects)
    canvas = np.zeros((y1 - y0, x1 - x0), np.float32)
    for r in rects:
        buf = np.frombuffer(bytes(r.planes[0]), np.uint8)
        ls = len(buf) // r.height
        idx = buf[:ls * r.height].reshape(r.height, ls)[:, :r.width]
        g = lut[idx]
        sub = canvas[r.y - y0:r.y - y0 + r.height, r.x - x0:r.x - x0 + r.width]
        np.maximum(sub, g, out=sub)
    if canvas.max() < 40:
        return None
    img = Image.fromarray(canvas.clip(0, 255).astype(np.uint8), "L")
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.Resampling.LANCZOS)
    g = np.asarray(img, np.float32)
    thr = max(0.5 * g.max(), 90.0)
    ink = g >= thr
    out = np.where(ink, 0, 255).astype(np.uint8)
    pad = 20
    out = np.pad(out, pad, constant_values=255)
    return Image.fromarray(out, "L")


_OCR_FIXES = [
    (re.compile(r"(?<![A-Za-z])[|](?![A-Za-z])"), "I"),
    (re.compile(r"(?<=\s)[|]|^[|]", re.M), "I"),
    (re.compile(r"[‘’`´]"), "'"),
    (re.compile(r"[“”]"), '"'),
]


def ocr(img: Image.Image) -> str:
    text = pytesseract.image_to_string(img, lang="eng", config=TESS_CONFIG)
    for rx, rep in _OCR_FIXES:
        text = rx.sub(rep, text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


# --------------------------------------------------------------------------- normalization

_MARKUP = re.compile(r"<[^>]*>|\{[^}]*\}")
_SDH = re.compile(r"\[[^\]]*\]?|\([^)]*\)?")
# Speaker labels: "JOHN:" in mixed-case tracks, "Caldwell:" in all-caps (CC-style) tracks.
_SPEAKER = re.compile(r"^\s*-?\s*(?:[A-Z][A-Z0-9 .'#-]{1,30}|[A-Z][a-z]+(?: [A-Z][a-z]+)?):(?!\d)\s*", re.M)
_PUNCT = re.compile(r"[^\w\s']|_")
_APOS = re.compile(r"(?<!\w)'|'(?!\w)")


def normalize(text: str, keep_apostrophes: bool = False) -> str:
    t = _MARKUP.sub(" ", text)
    t = _SDH.sub(" ", t)
    t = _SPEAKER.sub(" ", t)
    t = t.replace("’", "'").lower()
    t = _PUNCT.sub(" ", t)
    t = _APOS.sub(" ", t)
    if not keep_apostrophes:
        t = t.replace("'", "")
    return " ".join(t.split())


@functools.lru_cache(maxsize=None)
def in_vocab(word: str) -> bool:
    return zipf_frequency(word, "en") > 0


def quality(texts: list[str]) -> tuple[float | None, int]:
    words = [w for t in texts for w in normalize(t, keep_apostrophes=True).split()]
    if not words:
        return None, 0
    return sum(in_vocab(w) for w in words) / len(words), len(words)


# --------------------------------------------------------------------------- windows

def text_of(sub) -> str:
    try:
        raw = sub.dialogue if hasattr(sub, "dialogue") else sub.text
    except Exception:
        raw = getattr(sub, "text", b"")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    return raw.replace("\\N", "\n").replace("\\n", "\n").strip()


def extract_windows(path: Path, info: dict, track: dict | None, pool: ThreadPoolExecutor) -> dict:
    text_tracks = [s["index"] for s in info["streams"] if s["type"] == "subtitle"
                   and s["codec"] in TEXT_CODECS and (track is None or s["index"] != track["index"])]
    windows, captions = [], []
    if track is None and not text_tracks:
        return {"windows": windows, "caption_lines": captions}
    duration = info["duration"] or 0
    with av.open(str(path)) as c:
        main = c.streams[track["index"]] if track else None
        codec = track["codec"] if track else None
        upscale = 1
        if codec == "dvd_subtitle":
            upscale = 3
        wanted = ([main] if main else []) + [c.streams[i] for i in text_tracks]
        for frac in WINDOW_FRACS:
            target = frac * duration
            c.seek(int(target * av.time_base), backward=True, any_frame=False)
            for s in wanted:
                try:
                    s.codec_context.flush_buffers()
                except Exception:
                    pass
            pal = (PgsPalette() if codec == "hdmv_pgs_subtitle"
                   else VobPalette(main.codec_context.extradata) if codec == "dvd_subtitle" else None)
            events = []  # (t, image or text)
            last_t = target
            for pkt in c.demux(wanted):
                if pkt.pts is None:
                    continue
                t = float(pkt.pts * pkt.time_base)
                if t < target - 1:
                    continue
                if t > target + WINDOW_MAX_S:
                    break
                last_t = t
                data = bytes(pkt)
                is_main = main is not None and pkt.stream.index == main.index
                if is_main and pal is not None:
                    pal.feed(data)
                try:
                    subs = pkt.decode()
                except av.error.FFmpegError:
                    continue
                if not is_main:
                    for sub in subs:
                        txt = text_of(sub)
                        if txt:
                            captions.append({"t": round(t, 3), "text": txt})
                    continue
                if codec in BITMAP_CODECS:
                    img = render(list(subs), pal.lut(), upscale)
                    if img is not None:
                        events.append((t, img))
                else:
                    txt = "\n".join(filter(None, (text_of(s) for s in subs)))
                    if txt:
                        events.append((t, txt))
                if len(events) >= WINDOW_MIN_LINES:
                    break
            imgs = [e for e in events if not isinstance(e[1], str)]
            texts = dict(pool.map(lambda e: (e[0], ocr(e[1])), imgs)) if imgs else {}
            if DEBUG_DIR is not None:
                DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                for t, im in imgs:
                    im.save(DEBUG_DIR / f"{path.stem}-{t:08.2f}.png")
            lines, prev = [], None
            for t, item in events:
                txt = item if isinstance(item, str) else texts.get(t, "")
                if not txt or txt == prev:
                    continue
                prev = txt
                lines.append({"t": round(t, 3), "text": txt, "norm": normalize(txt)})
            windows.append({"frac": frac, "start": round(target, 3), "end": round(last_t, 3),
                            "events": len(events), "lines": lines})
    return {"windows": windows, "caption_lines": captions}


# --------------------------------------------------------------------------- driver

def list_files(paths: list[str], skip_dupes: bool) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(x for x in p.iterdir() if x.suffix.lower() in (".mkv", ".mp4", ".m4v")))
        else:
            out.append(p)
    if skip_dupes:
        out = [p for p in out if not DUPE_RE.search(p.name)]
    return out


def fmt_probe(info: dict, chosen: dict | None) -> str:
    lines = [f"{info['path']}  {info['size'] / 1e9:.2f} GB  {info['duration'] or 0:.0f}s"]
    for s in info["streams"]:
        ev = (f" events={s['events']}({s['events_source'] or 'not counted'})"
              if s["type"] == "subtitle" else "")
        flags = "".join([" forced" if s["forced"] else "", " default" if s["default"] else ""])
        mark = "  <== chosen" if chosen and s["index"] == chosen["index"] else ""
        lines.append(f"  #{s['index']} {s['type']:<8} {s['codec']:<20} lang={s['language']} "
                     f"title={s['title']!r}{flags}{ev}{mark}")
    if chosen is None:
        lines.append("  no usable subtitle track (dialogue: not available)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="season directories or files")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--show", help="show slug, e.g. supernatural")
    ap.add_argument("--season", type=int)
    ap.add_argument("--limit", type=int, help="first K files only")
    ap.add_argument("--skip-dupes", action="store_true", help="ignore names like s02e09a.mkv")
    ap.add_argument("--force", action="store_true", help="re-extract files that already have output")
    ap.add_argument("--debug-images", type=Path, help="save OCR input images to this directory")
    args = ap.parse_args()
    global DEBUG_DIR
    DEBUG_DIR = args.debug_images

    files = list_files(args.paths, args.skip_dupes)
    if args.limit:
        files = files[:args.limit]
    if not args.probe_only and (not args.show or args.season is None):
        ap.error("--show and --season are required unless --probe-only")

    outdir = None if args.probe_only else OUT_ROOT / f"{args.show}-s{args.season:02d}"
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(OCR_THREADS) as pool:
        for path in files:
            t0 = time.time()
            if outdir and (outdir / f"{path.stem}.json").exists() and not args.force:
                print(f"skip (exists) {path.name}", file=sys.stderr)
                continue
            info = probe(path, always_count=args.probe_only)
            chosen = select_track(info["streams"])
            if args.probe_only:
                print(fmt_probe(info, chosen))
                continue
            res = extract_windows(path, info, chosen, pool)
            texts = [ln["text"] for w in res["windows"] for ln in w["lines"]]
            if chosen and chosen["codec"] in TEXT_CODECS:
                q, nwords = 1.0, sum(len(normalize(t).split()) for t in texts)
            else:
                q, nwords = quality(texts)
            rec = {
                "path": str(path),
                "basename": path.name,
                "show": args.show,
                "season": args.season,
                "size": info["size"],
                "duration": info["duration"],
                "tracks": info["streams"],
                "chosen_track": None if chosen is None else {
                    k: chosen[k] for k in ("index", "codec", "language", "title", "events", "events_source")},
                "windows": res["windows"],
                "ocr_quality": q,
                "ocr_words": nwords,
                "caption_lines": res["caption_lines"],
                "elapsed_s": round(time.time() - t0, 1),
            }
            (outdir / f"{path.stem}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            nlines = sum(len(w["lines"]) for w in res["windows"])
            qs = "n/a" if q is None else f"{q:.3f}"
            codec = chosen["codec"] if chosen else "-"
            print(f"{path.name}: {codec} lines={nlines} words={nwords} quality={qs} "
                  f"captions={len(res['caption_lines'])} {rec['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
