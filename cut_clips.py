"""
Cut the 30 IRR video clips from the full-match source files using ffmpeg.

Expected source-video layout (next to this file):

    source_videos/
        2561505_espn.mp4         # FC Dordrecht vs Willem II   (file_match_id 2561505)
        2561505_mp4.mp4
        2561505_goal_left.mp4
        2561505_goal_right.mp4
        2561495_espn.mp4         # Almere City FC vs FC Dordrecht
        ...etc — four cameras per match

The file_match_id for each corner is read from corner_manifest.json. The
manifest's ``clip_start_ms`` / ``clip_end_ms`` are SciSports event-clock
times. Source videos don't start at kick-off though (they include warmup
/ pre-match intro), so every cut window is shifted by a per-match,
per-camera offset taken from ``MATCH_OFFSETS_MS`` below.

Output:
    videos/corner_01_espn.mp4
    videos/corner_01_mp4.mp4
    videos/corner_01_goal_left.mp4
    videos/corner_01_goal_right.mp4
    ...

Run:
    python cut_clips.py                # cut everything (skip files that exist)
    python cut_clips.py --force        # re-cut everything, overwriting
    python cut_clips.py --match 2561496  # only that match (combine with --force)
"""
import json
import subprocess
import sys
from pathlib import Path

THIS_DIR    = Path(__file__).parent
MANIFEST    = THIS_DIR / "corner_manifest.json"
SOURCE_DIR  = THIS_DIR / "source_videos"
OUT_DIR     = THIS_DIR / "videos"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Clip-window configuration
# ---------------------------------------------------------------------------
# How wide a clip to cut around each (offset-adjusted) kick moment.
# 5 s before the kick + 10 s after = 15 s total. Plenty of context.
CLIP_PRE_MS  = 5000
CLIP_POST_MS = 10000

# Per-match (and optionally per-camera) offset that maps SciSports event
# time -> video file time. The offset is the video timestamp at which the
# first-half kick-off happens (so positive = video has pre-match content
# before kick-off). Example: ESPN broadcast starts 5:40 before kick-off
# in match 2561496 -> offset = 5*60_000 + 40_000 = 340_000.
#
# Format:
#   MATCH_OFFSETS_MS[<file_match_id>] = <offset_ms>            # all cameras
#   MATCH_OFFSETS_MS[(<file_match_id>, "<cam>")] = <offset_ms> # one camera
#
# A per-(match, cam) entry overrides the per-match entry. Unknown matches
# default to 0 (i.e. assumes the video starts exactly at kick-off).
MATCH_OFFSETS_MS: dict = {
    # FC Dordrecht vs Willem II (2561505)
    #   mp4:                          kick-off at 0:21 ->  21_000 ms
    #   ESPN:                         kick-off at 6:27 -> 387_000 ms
    #   goal_left / goal_right:       start 5 s later  -> 26_000 ms
    2561505:                    21_000,
    (2561505, "espn"):         387_000,
    (2561505, "goal_left"):     26_000,
    (2561505, "goal_right"):    26_000,

    # Almere City FC vs FC Dordrecht (2561495)
    #   mp4:                          kick-off at 0:20 ->  20_000 ms
    #   ESPN:                         kick-off at 7:12 -> 432_000 ms
    #   goal_left / goal_right:       start 8 s later  -> 28_000 ms
    2561495:                    20_000,
    (2561495, "espn"):         432_000,
    (2561495, "goal_left"):     28_000,
    (2561495, "goal_right"):    28_000,

    # De Graafschap vs FC Dordrecht (7647)
    #   goal_left / goal_right / mp4: kick-off at 0:19 ->  19_000 ms
    #   ESPN:                         kick-off at 6:02 -> 362_000 ms
    7647:                 19_000,
    (7647, "espn"):      362_000,

    # Helmond Sport vs VVV-Venlo (2561496)
    #   goal_left / goal_right / mp4: kick-off at 0:20 ->  20_000 ms
    #   ESPN:                         kick-off at 5:40 -> 340_000 ms
    2561496:               20_000,
    (2561496, "espn"):    340_000,

    # Jong PSV vs VVV-Venlo (2561483)
    #   goal_left / goal_right / mp4: kick-off at 0:18 ->  18_000 ms
    #   ESPN:                         kick-off at 5:54 -> 354_000 ms
    2561483:               18_000,
    (2561483, "espn"):    354_000,
}

def offset_for(match_id: int, cam: str) -> int:
    """Resolve the offset to apply to (match_id, cam). Per-camera entries
    override per-match entries; unknown matches default to 0."""
    if (match_id, cam) in MATCH_OFFSETS_MS:
        return MATCH_OFFSETS_MS[(match_id, cam)]
    return MATCH_OFFSETS_MS.get(match_id, 0)


def ffmpeg_cut(src: Path, out: Path, start_ms: int, end_ms: int) -> bool:
    """Cut a clip from src using ffmpeg. Returns True on success.

    Strategy: try stream-copy first (fast, no quality loss, very robust).
    The clip may start at the nearest keyframe BEFORE the requested time
    (typically within ~2-5 s of the request), which is fine for a 15 s
    review clip. If stream-copy fails (e.g. unusual codec), fall back to
    a re-encode pass.
    """
    start_s    = start_ms / 1000
    duration_s = (end_ms - start_ms) / 1000

    # --- Pass 1: stream copy -------------------------------------------------
    cmd_copy = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-i", str(src),
        "-t", f"{duration_s:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out),
    ]
    res = subprocess.run(cmd_copy, capture_output=True, text=True)
    if res.returncode == 0 and out.exists() and out.stat().st_size > 1024:
        return True

    # --- Pass 2: re-encode fallback ------------------------------------------
    print(f"  stream-copy failed for {out.name}, retrying with re-encode...")
    cmd_enc = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-i", str(src),
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac",
        str(out),
    ]
    res = subprocess.run(cmd_enc, capture_output=True, text=True)
    if res.returncode == 0 and out.exists() and out.stat().st_size > 1024:
        return True

    print(f"  ffmpeg ERROR for {out.name} (both passes failed):")
    # Print up to the last 3000 chars of stderr so the real cause is visible
    err = (res.stderr or "").strip()
    print(err[-3000:] if err else "(no stderr produced)")
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing clips instead of skipping them.")
    ap.add_argument("--match", type=int, default=None,
                    help="Only cut clips for this file_match_id.")
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit(f"Missing {MANIFEST}. Run select_corners.py first.")
    if not SOURCE_DIR.exists():
        sys.exit(
            f"Missing {SOURCE_DIR}. Create it and place full-match videos "
            f"there named like '<file_match_id>_<espn|mp4|goal_left|goal_right>.mp4'."
        )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cams = ("espn", "mp4", "goal_left", "goal_right")

    corners = manifest["corners"]
    if args.match is not None:
        corners = [c for c in corners if c["match_id"] == args.match]
        if not corners:
            sys.exit(f"No corners in manifest for match_id {args.match}.")

    print(f"Cutting clips for {len(corners)} corners "
          f"(pre={CLIP_PRE_MS} ms, post={CLIP_POST_MS} ms)...\n")

    ok, fail, missing = 0, 0, []
    for c in corners:
        cid  = c["id"]
        mid  = c["match_id"]
        kick = c["kick_time_ms"]
        for cam in cams:
            src = SOURCE_DIR / f"{mid}_{cam}.mp4"
            out = OUT_DIR / f"corner_{cid:02d}_{cam}.mp4"
            if not src.exists():
                missing.append(src.name)
                continue
            if out.exists() and not args.force:
                print(f"  SKIP (exists): {out.name}  (use --force to overwrite)")
                ok += 1
                continue

            off = offset_for(mid, cam)
            start_ms = kick + off - CLIP_PRE_MS
            end_ms   = kick + off + CLIP_POST_MS
            if start_ms < 0:
                # If the kick is so early in event time that the offset
                # pushes us before the file start, just clamp to 0.
                end_ms  -= start_ms
                start_ms = 0

            print(f"  cutting {out.name}  from {src.name}  "
                  f"kick={kick/1000:.1f}s  off={off/1000:+.1f}s  "
                  f"-> [{start_ms/1000:.1f}s, {end_ms/1000:.1f}s]")
            if ffmpeg_cut(src, out, start_ms, end_ms):
                ok += 1
            else:
                fail += 1

    print(f"\nDone. {ok} clips written/kept, {fail} failures.")
    if missing:
        print(f"\n{len(missing)} source files were not found:")
        for m in sorted(set(missing)):
            print(f"  - source_videos/{m}")
        print("Place them in source_videos/ and rerun this script.")


if __name__ == "__main__":
    main()
