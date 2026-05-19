"""
Select the corners for the inter-rater reliability (IRR) study.

Phase 1 (MATCHES, 2 corners each):
  Picks 2 crossed corners from each of 5 matches  -> manifest ids 1..10.

Phase 2 (EXTRA_CORNERS_MATCHES, 1 corner each):
  Appends 1 additional corner from each listed match, skipping the ones
  already picked in phase 1            -> manifest ids 11..N.

This two-phase layout keeps ids 1..10 stable when new corners are added,
so previously cut clips (videos/corner_01_*.mp4 ... corner_10_*.mp4)
stay valid.

For each corner the manifest records:
  - match info, kick timestamp, corner side, delivery zone
  - all outfield players on the pitch at kick (jersey + team side),
    excluding goalkeepers and the corner taker
  - file paths for the 4-camera clips that cut_clips.py will produce

Output: corner_manifest.json next to this file.
"""
import json
import re
from pathlib import Path

import pandas as pd

THIS_DIR     = Path(__file__).parent
DATA_DIR     = Path(r"c:\Users\20203834\OneDrive\Master Data Science Entrepreneurship\FC Den Bosch - Thesis\Aanv. Corners Onderzoek\Data")
CORNERS_CSV  = THIS_DIR.parent / "output" / "all_corners.csv"
OUT_MANIFEST = THIS_DIR / "corner_manifest.json"

# Phase 1 — 2 corners per match (ids 1..10).
# (match_name_substring_1, match_name_substring_2, date_substring)
MATCHES = [
    ("FC Dordrecht", "Willem II",  "24-04-2026"),
    ("Almere",       "Dordrecht",  "17-04-2026"),
    ("De Graafschap","FC Dordrecht","14-03-2026"),
    ("Helmond",      "VVV",        "17-04-2026"),
    ("Jong PSV",     "VVV",        "06-04-2026"),
]
CORNERS_PER_MATCH = 2

# Phase 2 — 1 extra corner per listed match (ids 11..). The next available
# unused corner (sorted by start_time_ms) is picked.
EXTRA_CORNERS_MATCHES = [
    ("Almere",       "Dordrecht",  "17-04-2026"),  # -> id 11
    ("Jong PSV",     "VVV",        "06-04-2026"),  # -> id 12
]

CLIP_PRE_MS  = 5000
CLIP_POST_MS = 5000


# ---------------------------------------------------------------------------
def build_indices():
    pos_idx, ev_idx = {}, {}
    for f in DATA_DIR.rglob("*SciSports*.json"):
        m = re.search(r"- (\d+)\.json$", f.name)
        if not m: continue
        mid = int(m.group(1))
        if "Positions" in f.name: pos_idx[mid] = f
        elif "Events"  in f.name: ev_idx[mid]  = f
    return pos_idx, ev_idx


def stratified_pick(group_df, n):
    """Pick n corners from a single match maximising variety on side+delivery type."""
    if len(group_df) <= n:
        return group_df.copy()
    # Try one from each side if possible
    chosen = []
    for side in ["L", "R"]:
        side_rows = group_df[group_df["corner_side"] == side]
        if not side_rows.empty:
            # Within side, pick the one with the most "interesting" delivery zone
            preferred = side_rows[side_rows["target_zone"].isin(["CENTRAL", "NEAR"])]
            chosen.append((preferred if not preferred.empty else side_rows).iloc[0])
            if len(chosen) >= n: break
    # Fill remaining
    while len(chosen) < n:
        remaining = group_df[~group_df["event_id"].isin([c["event_id"] for c in chosen])]
        if remaining.empty: break
        chosen.append(remaining.iloc[0])
    return pd.DataFrame(chosen)


# ---------------------------------------------------------------------------
def main():
    print(f"Loading corners from {CORNERS_CSV}...")
    ac = pd.read_csv(CORNERS_CSV)
    ac = ac[ac["sub_type"] == "CORNER_CROSSED"].copy()

    pos_idx, ev_idx = build_indices()

    selected_corners = []
    used_event_ids = set()

    # Phase 1 — 2 corners per match
    print("\nPhase 1 (2 corners per match)...")
    for keys in MATCHES:
        match_hits = ac
        for k in keys:
            match_hits = match_hits[match_hits["match_name"].str.contains(k, case=False, na=False)]
        if match_hits.empty:
            print(f"  SKIP (no match): {keys}")
            continue
        match_name = match_hits["match_name"].iloc[0]
        print(f"  {match_name}  ({len(match_hits)} crossed corners)")
        picked = stratified_pick(match_hits, CORNERS_PER_MATCH)
        for _, row in picked.iterrows():
            selected_corners.append(row)
            used_event_ids.add(row["event_id"])

    # Phase 2 — 1 extra corner from each listed match, skipping used ones
    print("\nPhase 2 (extra corners)...")
    for keys in EXTRA_CORNERS_MATCHES:
        match_hits = ac
        for k in keys:
            match_hits = match_hits[match_hits["match_name"].str.contains(k, case=False, na=False)]
        avail = match_hits[~match_hits["event_id"].isin(used_event_ids)]
        if avail.empty:
            print(f"  SKIP (no unused corners): {keys}")
            continue
        avail = avail.sort_values("start_time_ms")
        picked = avail.iloc[0]
        selected_corners.append(picked)
        used_event_ids.add(picked["event_id"])
        print(f"  +1 from {picked['match_name']} @ {int(picked['start_time_ms'])} ms")

    print(f"\nTotal corners selected: {len(selected_corners)}")
    print("\nLoading tracking data per corner to extract on-pitch rosters...")

    manifest_corners = []
    for i, c in enumerate(selected_corners, start=1):
        # match_id in csv is a hex string; the integer SciSports match id
        # used in filenames is file_match_id.
        file_match_id = c.get("file_match_id")
        try:
            mid_int = int(file_match_id) if pd.notna(file_match_id) else None
        except Exception:
            mid_int = None
        if mid_int is None or mid_int not in pos_idx:
            print(f"  [{i:02d}] No tracking for {c['match_name']} — skipped")
            continue

        # Load the kick frame from the position file
        with open(pos_idx[mid_int], encoding="utf-8") as f:
            pos_data = json.load(f)["data"]
        frames = {fr["t"]: fr for fr in pos_data}
        del pos_data

        # Event metadata for team IDs and GK shirts
        with open(ev_idx[mid_int], encoding="utf-8") as f:
            ev_root = json.load(f)
        meta     = ev_root["metaData"]
        home_id  = meta["homeTeamId"]
        away_id  = meta["awayTeamId"]
        home_nm  = meta["homeTeamName"]
        away_nm  = meta["awayTeamName"]
        gk_home, gk_away = set(), set()
        for p in ev_root.get("players", []):
            if "goalkeeper" in p.get("positionName","").lower():
                if p.get("teamId") == home_id: gk_home.add(p["shirtNumber"])
                elif p.get("teamId") == away_id: gk_away.add(p["shirtNumber"])

        tms = int(c["start_time_ms"])
        kick_tick = (tms // 100) * 100
        fr = frames.get(kick_tick) or next(
            (frames.get(kick_tick + off) for off in (-100, 100, -200, 200, -300)
             if frames.get(kick_tick + off)), None)

        if fr is None:
            print(f"  [{i:02d}] No frame at kick for {c['match_name']} {tms} — skipped")
            continue

        taker_is_home = bool(c.get("is_home", True))
        att_key = "h" if taker_is_home else "a"
        def_key = "a" if taker_is_home else "h"
        gk_att  = gk_home if taker_is_home else gk_away
        gk_def  = gk_away if taker_is_home else gk_home
        att_team_name = home_nm if taker_is_home else away_nm
        def_team_name = away_nm if taker_is_home else home_nm

        attackers = []
        for p in fr.get(att_key, []):
            j = p["s"]
            if j in gk_att: continue
            if j == c.get("player_id"): continue   # skip the corner taker
            attackers.append({"jersey": j, "x": round(p["x"], 2), "y": round(p["y"], 2)})
        defenders = []
        for p in fr.get(def_key, []):
            j = p["s"]
            if j in gk_def: continue
            defenders.append({"jersey": j, "x": round(p["x"], 2), "y": round(p["y"], 2)})

        # Convert kick time to match clock
        seconds = tms // 1000
        mins, secs = divmod(seconds, 60)
        # SciSports event has half field
        part_name = "FIRST_HALF"   # placeholder; we'd need to read it back from events
        match_clock = f"{mins:02d}:{secs:02d}"

        manifest_corners.append({
            "id":             i,
            "corner_id":      f"{mid_int}:{tms}",
            "match_id":       mid_int,
            "match_name":     c["match_name"],
            "match_date":     str(c.get("date", "")),
            "kick_time_ms":   tms,
            "match_clock":    match_clock,
            "clip_start_ms":  tms - CLIP_PRE_MS,
            "clip_end_ms":    tms + CLIP_POST_MS,
            "attacking_team": att_team_name,
            "defending_team": def_team_name,
            "corner_side":    c["corner_side"],
            "swing":          c.get("swing", ""),
            "target_zone":    c.get("target_zone", ""),
            "delivery_type":  c.get("delivery_type", ""),
            "taker_name":     c.get("player_name", ""),
            "taker_jersey":   None,   # the taker's jersey is excluded from att roster
            "attackers":      attackers,
            "defenders":      defenders,
            # File names that cut_clips.py / make_tracking_videos.py produce
            "videos": {
                "espn":       f"videos/corner_{i:02d}_espn.mp4",
                "mp4":        f"videos/corner_{i:02d}_mp4.mp4",
                "goal_left":  f"videos/corner_{i:02d}_goal_left.mp4",
                "goal_right": f"videos/corner_{i:02d}_goal_right.mp4",
                "tracking":   f"videos/corner_{i:02d}_tracking.mp4",
            }
        })
        print(f"  [{i:02d}] {c['match_name']} {match_clock} ({c['corner_side']}, "
              f"{c.get('swing','?')}/{c.get('target_zone','?')})  "
              f"-> {len(attackers)} ATT, {len(defenders)} DEF")

    out = {
        "instructions": (
            f"{len(manifest_corners) * 4} video clips required: "
            "videos/corner_{NN}_{espn|mp4|goal_left|goal_right}.mp4. "
            "cut_clips.py uses kick_time_ms + per-match offset to cut a "
            "15-second window (5 s pre, 10 s post)."
        ),
        "corners": manifest_corners,
    }
    OUT_MANIFEST.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nManifest written -> {OUT_MANIFEST}")
    print(f"Total corners in manifest: {len(manifest_corners)}")


if __name__ == "__main__":
    main()
