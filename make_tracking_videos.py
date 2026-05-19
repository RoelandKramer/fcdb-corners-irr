"""
Maak 2D top-down tracking-videos voor elke corner in de IRR-manifest.

Elke output-clip toont de SciSports tracking data over hetzelfde 15-seconden
venster als de camera-clips (5 s voor de kick tot 10 s erna), op 10 fps
(een frame per 100 ms — de native SciSports tracking rate). Spelers worden
weergegeven als cirkels met hun rugnummer in het midden: aanvallers in rood,
verdedigers in blauw, de bal in geel. Coordinaten worden geprojecteerd op
``Full_field_zo_zones.png`` met dezelfde piecewise-lineaire anchor-transform
als ``video_base/app.py`` (de "full_zo" image config).

Om alle clips er hetzelfde uit te laten zien tijdens het labelen, worden
posities 180° gedraaid wanneer de cornernemer aan de negatieve-x kant van
het veld staat — zo staat de aanvallende ploeg altijd bovenin.

Output:
    videos/corner_NN_tracking.mp4

Gebruik:
    python make_tracking_videos.py                # alles renderen (bestaande overslaan)
    python make_tracking_videos.py --force        # bestaande overschrijven
    python make_tracking_videos.py --corner 7     # alleen corner 7
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------
THIS_DIR     = Path(__file__).parent
MANIFEST     = THIS_DIR / "corner_manifest.json"
OUT_DIR      = THIS_DIR / "videos"
DATA_DIR     = Path(r"c:\Users\20203834\OneDrive\Master Data Science Entrepreneurship\FC Den Bosch - Thesis\Aanv. Corners Onderzoek\Data")
FIELD_IMAGE  = Path(r"C:\Users\20203834\OneDrive\Master Data Science Entrepreneurship\FC Den Bosch - Thesis\video_base\Zone Figures\Full_field_zo_zones.png")


# ---------------------------------------------------------------------------
# Veld <-> pixel transform — letterlijk overgenomen uit video_base/app.py
# (de "full_zo" image config). Image = 1227 x 1835.
# ---------------------------------------------------------------------------
FULL_ZO_ANCHORS_X = [   # metric_x  ->  pixel_y (verticale as)
    ( 52.5,   67),      # aanvallende doellijn
    ( 47.0,  164),
    ( 36.0,  358),
    ( 17.5,  644),
    (  0.0,  930),      # middenlijn
    (-17.5, 1216),
    (-36.0, 1503),
    (-47.0, 1696),
    (-52.5, 1793),      # eigen doellijn
]
FULL_ZO_ANCHORS_Y = [   # metric_y  ->  pixel_x (horizontale as)
    (-34.0, 1154),      # TV-rechter zijlijn
    (-20.16, 927),
    ( -9.16, 754),
    (  0.0,  602),
    (  9.16, 449),
    ( 20.16, 278),
    ( 34.0,   64),      # TV-linker zijlijn
]


def piecewise_interp(value: float, anchors: list[tuple[float, float]]) -> float:
    ascending = anchors[0][0] <= anchors[-1][0]
    if not ascending:
        anchors = list(reversed(anchors))
    if value <= anchors[0][0]:
        m0, p0 = anchors[0]; m1, p1 = anchors[1]
        return p0 + (value - m0) * (p1 - p0) / (m1 - m0)
    if value >= anchors[-1][0]:
        m0, p0 = anchors[-2]; m1, p1 = anchors[-1]
        return p1 + (value - m1) * (p1 - p0) / (m1 - m0)
    for i in range(len(anchors) - 1):
        m0, p0 = anchors[i]; m1, p1 = anchors[i + 1]
        if m0 <= value <= m1:
            return p0 + (value - m0) * (p1 - p0) / (m1 - m0)
    return float(anchors[-1][1])


def metric_to_pixel(x_m: float, y_m: float) -> tuple[int, int]:
    py = piecewise_interp(x_m, FULL_ZO_ANCHORS_X)
    px = piecewise_interp(y_m, FULL_ZO_ANCHORS_Y)
    return int(round(px)), int(round(py))


# ---------------------------------------------------------------------------
# JSON-indexering per match
# ---------------------------------------------------------------------------
def build_indices() -> tuple[dict, dict]:
    pos_idx, ev_idx = {}, {}
    for f in DATA_DIR.rglob("*SciSports*.json"):
        m = re.search(r"- (\d+)\.json$", f.name)
        if not m:
            continue
        mid = int(m.group(1))
        if "Positions" in f.name:
            pos_idx[mid] = f
        elif "Events" in f.name:
            ev_idx[mid] = f
    return pos_idx, ev_idx


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------
COL_ATT  = (40, 40, 220)    # rood (BGR)
COL_DEF  = (220, 90, 30)    # blauw (BGR)
COL_BALL = (0, 220, 240)    # geel
COL_RING = (255, 255, 255)
COL_TEXT = (255, 255, 255)


def render_frame(field_img: np.ndarray,
                 frame: dict,
                 att_key: str,
                 def_key: str,
                 flip: bool,
                 time_to_kick_s: float,
                 corner_label: str,
                 radius: int = 30) -> np.ndarray:
    img = field_img.copy()

    def draw_player(p: dict, fill: tuple):
        x, y = p["x"], p["y"]
        if flip:
            x, y = -x, -y
        pxx, pyy = metric_to_pixel(x, y)
        cv2.circle(img, (pxx, pyy), radius + 3, COL_RING, -1)
        cv2.circle(img, (pxx, pyy), radius,     fill,     -1)
        text = str(p["s"])
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0 if len(text) <= 2 else 0.75
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        tx = pxx - tw // 2
        ty = pyy + th // 2
        cv2.putText(img, text, (tx, ty), font, scale, COL_TEXT, thick,
                     cv2.LINE_AA)

    for p in frame.get(att_key, []):
        draw_player(p, COL_ATT)
    for p in frame.get(def_key, []):
        draw_player(p, COL_DEF)

    # Bal
    b = frame.get("b")
    if b is not None:
        bx, by = b.get("x", 0.0), b.get("y", 0.0)
        if flip:
            bx, by = -bx, -by
        bpx, bpy = metric_to_pixel(bx, by)
        cv2.circle(img, (bpx, bpy), 12, (0, 0, 0),   -1)
        cv2.circle(img, (bpx, bpy), 10, COL_BALL,    -1)

    return img


# ---------------------------------------------------------------------------
# Per-corner video
# ---------------------------------------------------------------------------
FPS         = 10        # 100 ms per frame
PRE_MS      = 5_000
POST_MS     = 10_000
TARGET_H    = 720       # output-hoogte in px (breedte volgt aspect ratio ~0.67)


def make_one_video(corner: dict,
                    frames: dict,
                    taker_is_home: bool,
                    flip: bool,
                    field_img: np.ndarray,
                    out_path: Path) -> bool:
    """Render the 15-s tracking clip and encode it as H.264 MP4 via ffmpeg
    piping. H.264 is required for the HTML5 <video> tag used by Streamlit;
    OpenCV's mp4v fourcc produces MPEG-4 Part 2 which most browsers refuse
    to play.
    """
    kick_ms = corner["kick_time_ms"]
    att_key = "h" if taker_is_home else "a"
    def_key = "a" if taker_is_home else "h"

    start_ms = kick_ms - PRE_MS
    end_ms   = kick_ms + POST_MS
    start_tick = (start_ms // 100) * 100
    ticks      = list(range(start_tick, end_ms + 1, 100))

    h_full, w_full = field_img.shape[:2]
    out_h = TARGET_H
    out_w = int(round(w_full * out_h / h_full))
    # H.264 / yuv420p requires even dimensions
    if out_w % 2: out_w += 1
    if out_h % 2: out_h += 1

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(FPS),
        "-i", "-",                       # frames via stdin
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "22",
        "-movflags", "+faststart",       # so the browser can start playing while loading
        str(out_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                              stderr=subprocess.PIPE)

    corner_label = f"Corner {corner['id']:02d}  ({corner['match_clock']})"
    last_frame = None
    try:
        for tick in ticks:
            fr = frames.get(tick)
            if fr is None:
                for off in (-100, 100, -200, 200):
                    fr = frames.get(tick + off)
                    if fr:
                        break
            if fr is None:
                fr = last_frame
            if fr is None:
                img = field_img.copy()
            else:
                time_to_kick_s = (tick - kick_ms) / 1000.0
                img = render_frame(field_img, fr, att_key, def_key, flip,
                                    time_to_kick_s, corner_label)
            last_frame = fr
            img_resized = cv2.resize(img, (out_w, out_h),
                                      interpolation=cv2.INTER_AREA)
            proc.stdin.write(img_resized.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        print(f"  ffmpeg ERROR voor {out_path.name}:\n{err[-1500:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Bestaande tracking-clips overschrijven.")
    ap.add_argument("--corner", type=int, default=None,
                    help="Alleen deze corner-id renderen.")
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit(f"Mist {MANIFEST}. Run select_corners.py eerst.")
    OUT_DIR.mkdir(exist_ok=True)

    field_img = cv2.imread(str(FIELD_IMAGE))
    if field_img is None:
        sys.exit(f"Kan veldfoto niet laden: {FIELD_IMAGE}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    corners  = manifest["corners"]
    if args.corner is not None:
        corners = [c for c in corners if c["id"] == args.corner]
        if not corners:
            sys.exit(f"Geen corner met id {args.corner} in manifest.")

    pos_idx, ev_idx = build_indices()

    by_match: dict[int, list] = {}
    for c in corners:
        by_match.setdefault(c["match_id"], []).append(c)

    total_ok, total_skip, total_fail = 0, 0, 0

    for mid, match_corners in by_match.items():
        outs = [(c, OUT_DIR / f"corner_{c['id']:02d}_tracking.mp4")
                for c in match_corners]
        if not args.force and all(p.exists() for _, p in outs):
            for _, p in outs:
                print(f"  SKIP (bestaat): {p.name}")
                total_skip += 1
            continue

        if mid not in pos_idx:
            print(f"  Geen tracking-JSON voor match_id {mid}; skip.")
            total_fail += len(match_corners)
            continue
        if mid not in ev_idx:
            print(f"  Geen events-JSON voor match_id {mid}; skip.")
            total_fail += len(match_corners)
            continue

        print(f"\nLaad positions voor match {mid}: {pos_idx[mid].name}")
        with open(pos_idx[mid], encoding="utf-8") as f:
            pos_data = json.load(f)["data"]
        frames = {fr["t"]: fr for fr in pos_data}
        del pos_data

        with open(ev_idx[mid], encoding="utf-8") as f:
            ev_root = json.load(f)
        meta    = ev_root["metaData"]
        home_id = meta["homeTeamId"]

        for corner, out_path in outs:
            if out_path.exists() and not args.force:
                print(f"  SKIP (bestaat): {out_path.name}  (gebruik --force om te overschrijven)")
                total_skip += 1
                continue

            kick_ms = corner["kick_time_ms"]
            matching = [e for e in ev_root["data"]
                          if e.get("subTypeName") == "CORNER_CROSSED"
                          and int(e["startTimeMs"]) == kick_ms]
            if not matching:
                print(f"  Geen matching event voor corner {corner['id']} "
                      f"(kick={kick_ms}); skip")
                total_fail += 1
                continue
            taker_is_home = matching[0]["teamId"] == home_id

            kick_tick = (kick_ms // 100) * 100
            kick_frame = frames.get(kick_tick)
            for off in (-100, 100, -200, 200):
                if kick_frame:
                    break
                kick_frame = frames.get(kick_tick + off)
            if kick_frame is None:
                print(f"  Geen frame bij kick voor corner {corner['id']}; skip")
                total_fail += 1
                continue
            att_key = "h" if taker_is_home else "a"
            att_positions = kick_frame.get(att_key, [])
            mean_x = float(np.mean([p["x"] for p in att_positions])) if att_positions else 0.0
            flip = mean_x < 0

            print(f"  render {out_path.name}  kick={kick_ms/1000:.1f}s  "
                   f"taker_is_home={taker_is_home}  flip={flip}")
            ok = make_one_video(corner, frames, taker_is_home, flip,
                                 field_img, out_path)
            if ok:
                total_ok += 1
            else:
                total_fail += 1

        del frames, ev_root

    print(f"\nKlaar. {total_ok} clips geschreven, {total_skip} overgeslagen, "
          f"{total_fail} mislukt.")


if __name__ == "__main__":
    main()
