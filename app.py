"""
Inter-Rater Reliability — Corner-rollen Labelen App (Nederlandstalig)
======================================================================

Layout naast elkaar: video links, rol-invoerformulier rechts
(Verdedigers / Aanvallers in aparte tabs).

Het formulier zit binnen een `@st.fragment`, zodat het kiezen van een rol
alleen het formulier opnieuw rendert — de video links blijft precies op
zijn plek. Slaat automatisch op naar een per-gebruiker CSV na elke
wijziging.

Lokaal starten:
    streamlit run app.py
"""
from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_DIR        = Path(__file__).parent
MANIFEST_PATH  = APP_DIR / "corner_manifest.json"
LABELS_DIR     = APP_DIR / "labels"
LABELS_DIR.mkdir(exist_ok=True)

# Waar staan de video-clips?
#   "" (leeg)  -> lokaal (videos/-map naast app.py). Streaming gebeurt via een
#                 base64 data-URI ingebed in de custom HTML5-speler.
#   URL        -> externe host (b.v. GitHub Release of een CDN). De HTML5-
#                 speler krijgt dan de directe URL en streamt rechtstreeks.
#                 Filename wordt aan de URL geplakt (zonder "videos/"-prefix).
#
# Voorbeeld voor een GitHub Release:
#     VIDEO_BASE_URL = "https://github.com/USERNAME/REPO/releases/download/clips-v1"
#
# Voor Streamlit Cloud kun je dit ook via st.secrets configureren — zie de
# fallback hieronder.
try:
    VIDEO_BASE_URL = st.secrets.get("VIDEO_BASE_URL", "")
except Exception:
    VIDEO_BASE_URL = ""

# Persistente label-opslag via een private GitHub Gist.
#   GIST_TOKEN = "ghp_..."  — PAT met scope `gist`
#   GIST_ID    = "abc123…"  — id van een bestaande private gist
# Beide via st.secrets (Streamlit Cloud). Leeg = alleen lokale CSV (Streamlit
# Cloud container is niet persistent, dus zet dit aan voor de live deploy).
try:
    GIST_TOKEN = st.secrets.get("GIST_TOKEN", "")
    GIST_ID    = st.secrets.get("GIST_ID", "")
except Exception:
    GIST_TOKEN = ""
    GIST_ID    = ""

GIST_ENABLED       = bool(GIST_TOKEN and GIST_ID)
GIST_SYNC_INTERVAL = 8     # min. seconden tussen automatische sync-pogingen
GIST_API_URL       = "https://api.github.com"

PASSWORD = "denbosch2026"

# 12 corners, 6 vrienden, 6 corners per vriend, 3 reviews per corner
# (6 * 6 = 36 = 12 * 3). Roeland (auteur thesis) labelt alle 12 als gold
# standard. Elke corner wordt exact 3 keer beoordeeld door de 6 vrienden.
USER_ASSIGNMENTS = {
    "RoelandKramer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "PimVeldhuis":   [1, 2, 3, 4, 5, 6],
    "JorisPanman":   [7, 8, 9, 10, 11, 12],
    "MaxHautvast":   [1, 2, 7, 8, 9, 10],
    "RubenDeLaat":   [3, 4, 5, 6, 11, 12],
    "JopZweekhorst": [1, 3, 5, 7, 9, 11],
    "SamVeld":       [2, 4, 6, 8, 10, 12],
    "SimonDirkx":    [12, 11, 9, 8, 7, 6],
    "MeesVoskuijl":  [10, 9, 6, 5, 4, 3], 
    "LuukCornelisse": [8, 7, 4, 3, 2, 1],
}

# Rolcodes (Engels) — blijven Engels zodat de CSV-output bruikbaar blijft
# voor de thesisanalyse. De zichtbare labels (zie *_LABELS_NL) zijn Nederlands.
DEF_ROLES = ["", "MAN", "ZONAL", "SHORT", "COUNTER", "DON'T KNOW"]
ATT_ROLES = ["", "TARGET", "DECOY", "SHORT", "STATIC", "STAY_BACK",
             "SECOND_BALL", "BLOCK_GK", "DON'T KNOW"]
# Vervallen rollen die niet meer in de selectbox staan, maar bestaande
# labels mogen ze nog hebben (worden geflagged zodat de rater corrigeert).
DEPRECATED_ATT_ROLES = {"BLOCK_DEF"}

DEF_LABELS_NL = {
    "":            "— kies —",
    "MAN":         "MAN (mandekker)",
    "ZONAL":       "ZONAL (zonedekking)",
    "SHORT":       "SHORT (korte corner)",
    "COUNTER":     "COUNTER (Blijft voorin)",
    "DON'T KNOW":  "WEET NIET",
}

ATT_LABELS_NL = {
    "":             "— kies —",
    "TARGET":       "TARGET (doelwit)",
    "DECOY":        "DECOY (afleiding)",
    "SHORT":        "SHORT (korte corner)",
    "STATIC":       "STATIC (statisch)",
    "STAY_BACK":    "STAY_BACK (blijft achter)",
    "SECOND_BALL":  "SECOND_BALL (tweede bal)",
    "BLOCK_GK":     "BLOCK_GK (keeper blokken)",
    "BLOCK_DEF":    "⚠️ BLOCK_DEF (vervallen — graag aanpassen)",
    "DON'T KNOW":   "WEET NIET",
}

DEF_ROLE_HELP = {
    "MAN":     "**MAN (Mandekker)** — volgt één specifieke aanvaller "
               "gedurende de hele corner. Blijft dicht bij hem en volgt "
               "zijn beweging totdat de bal arriveert.",
    "ZONAL":   "**ZONAL (Zonedekking)** — bewaakt een vaste zone in of aan de rand van het strafschopgebied."
    " Reageert op de bal, niet op een "
               "specifieke aanvaller; blijft in zijn zone ongeacht wie er "
               "doorheen loopt. Het dekken van de eerste of tweede paal "
               "valt hier ook onder.",
    "SHORT":   "**SHORT (Korte corner)** — staat klaar om de speler die "
               "zich kort heeft aangeboden aan te pakken; opgesteld om "
               "een korte corner te verdedigen.",
    "COUNTER": "**COUNTER (Blijft voorin)** — staat hoog op het veld "
               "(rond de middenlijn) om de tegenaanval te dekken. "
               "Verdedigt het strafschopgebied zelf niet.",
}

ATT_ROLE_HELP = {
    "TARGET":      "**TARGET (Doelwit)** — beweegt *naar* het punt waar "
                    "de bal aankomt. Spelers die richting de bal lopen "
                    "vallen hieronder. Kunnen er meerdere zijn.",
    "DECOY":       "**DECOY (Afleiding)** — beweegt *weg* van het punt "
                    "waar de bal aankomt om verdedigers mee te trekken. "
                    "Spelers die van de bal weg lopen vallen hieronder. "
                    "Kunnen er meerdere zijn.",
    "SHORT":       "**SHORT (Korte corner)** — biedt zich kort aan bij "
                    "de cornernemer om een korte pass te ontvangen "
                    "(staat dicht bij de cornervlag, vaak buiten het "
                    "strafschopgebied).",
    "STATIC":      "**STATIC (Statisch)** — kiest geen looplijn. Blijft "
                    "ongeveer op dezelfde plek staan (vaak een vasthoud- "
                    "of ankerrol).",
    "STAY_BACK":   "**STAY_BACK (Blijft achter)** — gaat niet mee naar "
                    "voren voor de corner. Blijft op de eigen helft of "
                    "net daarbuiten om bij balverlies een tegenaanval "
                    "van de tegenstander te kunnen verdedigen.",
    "SECOND_BALL": "**SECOND_BALL (Tweede bal)** — staat rond de rand van "
                    "het strafschopgebied (16-meterlijn) om uitgekopte "
                    "ballen en rebounds op te pikken.",
    "BLOCK_GK":    "**BLOCK_GK (Keeper blokken)** — staat heel dicht bij "
                    "de keeper van de tegenstander om hem van de bal af "
                    "te schermen.",
}

# ---------------------------------------------------------------------------
# Manifest + paden
# ---------------------------------------------------------------------------
@st.cache_data
def load_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def video_url(rel_path: str) -> str:
    """Build the URL/path used by the HTML5 video player.

    Local mode (VIDEO_BASE_URL == ""): return the relative path; the caller
    base64-encodes the file. External mode: strip the "videos/" prefix and
    glue the filename onto VIDEO_BASE_URL — GitHub Releases (and most CDNs)
    store assets at the bucket root, not under "videos/"."""
    if not VIDEO_BASE_URL:
        return rel_path
    return f"{VIDEO_BASE_URL.rstrip('/')}/{Path(rel_path).name}"


# ---------------------------------------------------------------------------
# Opslag (CSV per gebruiker)
# ---------------------------------------------------------------------------
def user_csv_path(username: str) -> Path:
    return LABELS_DIR / f"labels_{username}.csv"


# ---- Gist persistence helpers --------------------------------------------
def _gist_filename(username: str) -> str:
    return f"labels_{username}.csv"


def _gist_fetch_csv(username: str) -> str | None:
    """Fetch a single file's content from the labels gist. None on failure."""
    if not GIST_ENABLED:
        return None
    try:
        r = requests.get(
            f"{GIST_API_URL}/gists/{GIST_ID}",
            headers={
                "Authorization": f"token {GIST_TOKEN}",
                "Accept":        "application/vnd.github+json",
            },
            timeout=15,
        )
        r.raise_for_status()
        files = r.json().get("files", {})
        f = files.get(_gist_filename(username))
        if not f:
            return None
        # Bij grote bestanden zet GitHub "truncated": True en biedt raw_url.
        if f.get("truncated") and f.get("raw_url"):
            rr = requests.get(f["raw_url"], timeout=15)
            rr.raise_for_status()
            return rr.text
        return f.get("content")
    except Exception as e:
        st.warning(f"Kon labels niet ophalen uit gist: {e}")
        return None


def _gist_push_csv(username: str, csv_text: str) -> bool:
    """Update one file in the labels gist. Returns True on success."""
    if not GIST_ENABLED:
        return False
    try:
        r = requests.patch(
            f"{GIST_API_URL}/gists/{GIST_ID}",
            headers={
                "Authorization": f"token {GIST_TOKEN}",
                "Accept":        "application/vnd.github+json",
            },
            json={"files": {_gist_filename(username): {"content": csv_text}}},
            timeout=15,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False


def load_user_labels(username: str) -> dict:
    """Load labels from the gist (authoritative on Streamlit Cloud), falling
    back to the local CSV. Local CSV gets refreshed when the gist has data so
    subsequent saves can diff cheaply."""
    p = user_csv_path(username)
    if GIST_ENABLED:
        csv_text = _gist_fetch_csv(username)
        if csv_text:
            p.write_text(csv_text, encoding="utf-8")
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for _, r in df.iterrows():
        key = (int(r["corner_id_int"]), r["player_team"], int(r["jersey"]))
        out[key] = {
            "role":  r["role"] if pd.notna(r["role"]) else "",
            "marks": int(r["marks"]) if pd.notna(r["marks"]) else None,
        }
    return out


def _labels_to_csv_text(username: str, labels: dict, manifest: dict) -> str:
    by_corner = {c["id"]: c for c in manifest["corners"]}
    rows = []
    for (cid, team, jersey), v in labels.items():
        c = by_corner.get(cid)
        if not c:
            continue
        rows.append({
            "rater":         username,
            "corner_id_int": cid,
            "corner_id":     c["corner_id"],
            "match_name":    c["match_name"],
            "match_clock":   c["match_clock"],
            "player_team":   team,
            "jersey":        jersey,
            "role":          v.get("role") or "",
            "marks":         v.get("marks"),
            "saved_at":      datetime.utcnow().isoformat(),
        })
    return pd.DataFrame(rows).to_csv(index=False)


def save_user_labels(username: str, labels: dict, manifest: dict,
                     force_remote: bool = False) -> None:
    """Save labels to local CSV (always) and to the gist (throttled, unless
    ``force_remote`` is True). Throttling avoids hammering the GitHub API on
    every selectbox change."""
    csv_text = _labels_to_csv_text(username, labels, manifest)
    user_csv_path(username).write_text(csv_text, encoding="utf-8")

    if not GIST_ENABLED:
        return

    now      = time.time()
    last     = float(st.session_state.get("_gist_last_save_ts", 0))
    last_txt = st.session_state.get("_gist_last_save_text", "")

    # Niets te uploaden als er niks veranderd is sinds vorige sync
    if not force_remote and csv_text == last_txt:
        return
    # Anders: pas pushen als de cooldown voorbij is, of als force_remote
    if not force_remote and (now - last) < GIST_SYNC_INTERVAL:
        st.session_state["_gist_dirty"] = True
        return

    ok = _gist_push_csv(username, csv_text)
    if ok:
        st.session_state["_gist_last_save_ts"]   = now
        st.session_state["_gist_last_save_text"] = csv_text
        st.session_state["_gist_dirty"]          = False
    else:
        st.session_state["_gist_dirty"] = True


# ---------------------------------------------------------------------------
# Inlogscherm
# ---------------------------------------------------------------------------
def login_page():
    st.title("Corner-rollen Labelen — IRR Studie")
    st.write(
        "Hoi, thanks voor 't meehelpen! Voor mijn afstudeeronderzoek bij "
        "FC Den Bosch bouw ik een algoritme dat automatisch rollen van "
        "verdedigers en aanvallers uit corner-data haalt. Om dat algoritme "
        "te bouwen heb ik handmatig een hele hoop corners gelabeld met "
        "rollen van spelers — voor verdedigers of ze zonaal of man-dekken, "
        "en voor aanvallers wat hun rol tijdens de corner is.\n\n"
        "Om dit te rechtvaardigen en mijn handmatige testset als grond van "
        "waarheid te kunnen zien, is een inter-reliability test nodig. "
        "Daar heb ik jouw hulp voor nodig :)  Door 6 corners die ik zelf "
        "ook gelabeld heb te labelen, kan ik zien of de labels die ik "
        "aanwijs logisch zijn en overeenkomen met wat jij en anderen "
        "die ik gevraagd heb me te helpen vinden.\n\n"
        "Log hieronder in om bij de corners te komen. Binnen de webapp "
        "staat meer duiding over de verschillende rollen. Stuur me een "
        "berichtje als je ergens toch niet helemaal uitkomt. Thanks!!"
    )
    # Form zodat Enter na het wachtwoord de Inloggen-knop indrukt
    with st.form("login_form", clear_on_submit=False, border=False):
        name = st.text_input("Je gebruikersnaam (hoofdlettergevoelig)").strip()
        pwd  = st.text_input("Wachtwoord", type="password")
        submitted = st.form_submit_button("Inloggen", type="primary")

    if submitted:
        if pwd != PASSWORD:
            st.error("Verkeerd wachtwoord.")
        elif name not in USER_ASSIGNMENTS:
            st.error(
                f"Gebruikersnaam niet herkend. Verwacht één van: "
                f"{', '.join(USER_ASSIGNMENTS.keys())}"
            )
        else:
            st.session_state.username = name
            st.session_state.labels   = load_user_labels(name)
            st.session_state.cursor   = 0
            st.rerun()


# ---------------------------------------------------------------------------
# Cornerpagina — video (links)
# ---------------------------------------------------------------------------
CAM_ORDER = [
    ("ESPN",        "espn"),
    ("MP4",         "mp4"),
    ("Goal Links",  "goal_left"),
    ("Goal Rechts", "goal_right"),
    ("Tracking 2D", "tracking"),
]


@st.cache_data(show_spinner=False)
def _video_source(rel_path: str) -> str:
    """Return the value to put in the <video src=…> attribute.

    Local mode: read the local file and return a data:video/mp4;base64,…
    URI so it works without any static-serving config.
    External mode (VIDEO_BASE_URL set): return the direct URL so the
    browser streams from the CDN."""
    if VIDEO_BASE_URL:
        return video_url(rel_path)
    full = APP_DIR / rel_path
    if not full.exists():
        return ""
    return ("data:video/mp4;base64,"
            + base64.b64encode(full.read_bytes()).decode("ascii"))


def render_video(corner: dict, corner_idx: int):
    """Custom HTML5 video player with camera-switch buttons. All sources
    embedded as base64 data URIs so switching cameras happens in pure
    JavaScript — the playback position is preserved across switches."""
    sources: dict[str, str] = {}
    for label, key in CAM_ORDER:
        rel = corner["videos"].get(
            key, f"videos/corner_{corner['id']:02d}_{key}.mp4"
        )
        src = _video_source(rel)
        if src:
            sources[label] = src

    if not sources:
        st.warning(
            f"Geen video's gevonden voor corner {corner['id']:02d}. "
            f"Zorg dat de clipbestanden in de map videos/ staan."
        )
        return

    initial = next(label for label, _ in CAM_ORDER if label in sources)
    missing = [label for label, _ in CAM_ORDER if label not in sources]

    # Build buttons (greyed-out for missing sources)
    btn_html_parts = []
    for label, _ in CAM_ORDER:
        if label in sources:
            cls = "cam-btn" + (" active" if label == initial else "")
            btn_html_parts.append(
                f'<button class="{cls}" data-cam="{label}">{label}</button>'
            )
        else:
            btn_html_parts.append(
                f'<button class="cam-btn missing" disabled>{label}</button>'
            )
    btn_html = "".join(btn_html_parts)

    sources_json = json.dumps(sources)
    player_id    = f"player_{corner_idx}"

    missing_note = ""
    if missing:
        missing_note = (
            f'<div class="missing-note">Niet beschikbaar voor deze corner: '
            f'{", ".join(missing)}</div>'
        )

    html = f"""
<style>
  .video-wrap {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .video-wrap video {{
    width: 100%;
    max-height: 720px;
    background: #000;
    border-radius: 6px;
    display: block;
  }}
  .cam-buttons {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
  }}
  .cam-btn {{
    flex: 1 1 auto;
    padding: 8px 10px;
    border: 1px solid #444;
    background: #1e1e1e;
    color: #eee;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    transition: background 0.1s, border-color 0.1s;
  }}
  .cam-btn:hover:not(:disabled) {{
    background: #333;
    border-color: #888;
  }}
  .cam-btn.active {{
    background: #d63333;
    border-color: #d63333;
    color: #fff;
    font-weight: 600;
  }}
  .cam-btn.missing, .cam-btn:disabled {{
    opacity: 0.35;
    cursor: not-allowed;
  }}
  .missing-note {{
    margin-top: 6px;
    color: #aaa;
    font-size: 12px;
  }}
</style>
<div class="video-wrap">
  <div style="color:#bbb; font-size:13px; margin-bottom:6px;">Camerahoek</div>
  <video id="{player_id}" controls preload="auto">
    <source src="{sources[initial]}" type="video/mp4">
  </video>
  <div class="cam-buttons">{btn_html}</div>
  {missing_note}
</div>
<script>
(function() {{
  const sources = {sources_json};
  const player  = document.getElementById("{player_id}");
  const btns    = document.querySelectorAll('.video-wrap .cam-btn[data-cam]');

  btns.forEach(function(b) {{
    b.addEventListener('click', function() {{
      const newCam = b.dataset.cam;
      if (!sources[newCam]) return;

      // Capture current playback state
      const t        = player.currentTime;
      const wasPlay  = !player.paused && !player.ended;
      const rate     = player.playbackRate;

      // Swap source and reseek
      const onMeta = function() {{
        try {{ player.currentTime = t; }} catch (e) {{}}
        player.playbackRate = rate;
        if (wasPlay) {{
          const p = player.play();
          if (p && p.catch) p.catch(function(){{}});
        }}
        player.removeEventListener('loadedmetadata', onMeta);
      }};
      player.addEventListener('loadedmetadata', onMeta);
      player.src = sources[newCam];
      player.load();

      // Update active button styling
      btns.forEach(function(x) {{ x.classList.remove('active'); }});
      b.classList.add('active');
    }});
  }});
}})();
</script>
"""

    # Component height: video (max 720) + buttons (~50) + label (~30) + note (~30)
    components.html(html, height=830, scrolling=False)


# ---------------------------------------------------------------------------
# Cornerpagina — formulier (rechts). In @st.fragment zodat het wijzigen van
# een rol NIET het hele script opnieuw rendert (video blijft op zijn plek).
# ---------------------------------------------------------------------------
def _attacker_row(corner: dict, corner_idx: int, p: dict):
    j = p["jersey"]
    current = st.session_state.labels.get((corner_idx, "ATT", j), {})
    saved   = current.get("role", "")

    # Vervallen rollen (b.v. BLOCK_DEF) blijven in de keuzelijst zolang ze
    # actief geselecteerd staan, zodat oude labels niet stilletjes verdwijnen.
    # Zodra de rater iets anders kiest valt de optie weg.
    if saved in DEPRECATED_ATT_ROLES:
        options = ATT_ROLES + [saved]
    else:
        options = ATT_ROLES

    c1, c2 = st.columns([1, 5], vertical_alignment="center")
    with c1:
        st.markdown(f"#### #{j}")
    with c2:
        idx = options.index(saved) if saved in options else 0
        role = st.selectbox(
            f"Rol voor aanvaller #{j}",
            options,
            index=idx,
            key=f"att_{corner_idx}_{j}_role",
            format_func=lambda x: ATT_LABELS_NL.get(x, x),
            label_visibility="collapsed",
            placeholder="Kies een rol…",
        )
        if role in DEPRECATED_ATT_ROLES:
            st.warning(
                f"⚠️ De rol **{role}** is vervallen. Kies hierboven aub "
                f"een andere rol voor speler #{j}.",
                icon="⚠️",
            )
    st.session_state.labels[(corner_idx, "ATT", j)] = {"role": role, "marks": None}


def _defender_row(corner: dict, corner_idx: int, p: dict,
                  attacker_options: list[str]):
    j = p["jersey"]
    current = st.session_state.labels.get((corner_idx, "DEF", j), {})
    c1, c2, c3 = st.columns([1, 3, 2], vertical_alignment="center")
    with c1:
        st.markdown(f"#### #{j}")
    with c2:
        idx = DEF_ROLES.index(current.get("role")) if current.get("role") in DEF_ROLES else 0
        role = st.selectbox(
            f"Rol voor verdediger #{j}",
            DEF_ROLES,
            index=idx,
            key=f"def_{corner_idx}_{j}_role",
            format_func=lambda x: DEF_LABELS_NL.get(x, x),
            label_visibility="collapsed",
            placeholder="Kies een rol…",
        )
    marks = None
    with c3:
        if role == "MAN":
            cur = current.get("marks")
            cur_str = str(cur) if cur is not None else ""
            m_idx = attacker_options.index(cur_str) if cur_str in attacker_options else 0
            m_str = st.selectbox(
                f"Dekt aanvaller voor verdediger #{j}",
                attacker_options,
                index=m_idx,
                key=f"def_{corner_idx}_{j}_marks",
                format_func=lambda x: f"#{x}" if x else "— kies —",
                label_visibility="collapsed",
                placeholder="Dekt aanvaller #…",
            )
            marks = int(m_str) if m_str and m_str.strip().isdigit() else None
        else:
            st.caption(" ")  # rijhoogte stabiel houden
    st.session_state.labels[(corner_idx, "DEF", j)] = {"role": role, "marks": marks}


@st.fragment
def role_form_fragment(corner: dict, corner_idx: int,
                       username: str, manifest: dict):
    def_tab, att_tab = st.tabs([
        f"🛡️ Verdedigers — {corner['defending_team']} ({len(corner['defenders'])})",
        f"⚔️ Aanvallers — {corner['attacking_team']} ({len(corner['attackers'])})",
    ])

    with def_tab:
        st.caption(
            "Kies voor elke verdediger **MAN / ZONAL / SHORT / COUNTER**. "
            "Bij MAN: kies ook het rugnummer van de aanvaller die hij dekt."
        )
        attacker_options = [""] + [str(a["jersey"]) for a in corner["attackers"]]
        with st.container(height=620, border=False):
            for p in corner["defenders"]:
                _defender_row(corner, corner_idx, p, attacker_options)

    with att_tab:
        st.caption(
            "Kies voor elke aanvaller de rol die zijn looplijn / beweging "
            "tijdens de corner het beste beschrijft."
        )
        with st.container(height=620, border=False):
            for p in corner["attackers"]:
                _attacker_row(corner, corner_idx, p)

    # Auto-save binnen de fragment (alleen wanneer een widget hierbinnen wijzigt)
    save_user_labels(username, st.session_state.labels, manifest)
    st.caption("✓ Automatisch opgeslagen")


# ---------------------------------------------------------------------------
# Cornerpagina — top-level
# ---------------------------------------------------------------------------
def corner_page(manifest: dict):
    username = st.session_state.username
    queue    = USER_ASSIGNMENTS[username]
    cursor   = st.session_state.cursor

    if cursor >= len(queue):
        done_page(username, manifest)
        return

    corner_idx = queue[cursor]
    corner = next(c for c in manifest["corners"] if c["id"] == corner_idx)

    # Header
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"### Corner {cursor + 1} van {len(queue)}")
        st.caption(f"{corner['match_name']} — {corner['match_clock']}")
    with h2:
        # Progress = corners die al zijn afgerond (cursor = index van de
        # huidige, dus nog niet bevestigd). Bij de laatste corner (5/6) zie
        # je dus 83%; pas op de Klaar-pagina is het 100%.
        progress = cursor / len(queue)
        st.progress(progress, text=f"{int(progress*100)} % voltooid")

    # Naast elkaar: video links, formulier rechts
    col_video, col_form = st.columns([1.2, 0.8], gap="large")
    with col_video:
        render_video(corner, corner_idx)
    with col_form:
        role_form_fragment(corner, corner_idx, username, manifest)

    # Navigatie (volledige rerun bij klik — gewenst, want we gaan naar een
    # nieuwe corner). Op deze "definitieve" momenten forceren we ook een
    # remote sync zodat we zeker weten dat de gist up-to-date is.
    st.markdown("---")
    nav_prev, nav_mid, nav_next = st.columns([1, 1, 1])
    with nav_prev:
        if cursor > 0:
            if st.button("← Vorige", use_container_width=True):
                save_user_labels(username, st.session_state.labels,
                                  manifest, force_remote=True)
                st.session_state.cursor -= 1
                st.rerun()
    with nav_mid:
        st.caption(f"Corner {cursor + 1} / {len(queue)}")
    with nav_next:
        if st.button("Volgende →", type="primary", use_container_width=True):
            save_user_labels(username, st.session_state.labels,
                              manifest, force_remote=True)
            st.session_state.cursor += 1
            st.rerun()


# ---------------------------------------------------------------------------
# Klaar-pagina
# ---------------------------------------------------------------------------
def done_page(username: str, manifest: dict):
    st.title("Helemaal klaar!")
    m = re.match(r"^[A-Z][a-z]+", username)
    first_name = m.group(0) if m else username
    st.write(f"Bedankt, **{first_name}** — je labels zijn opgeslagen!")
    save_user_labels(username, st.session_state.labels, manifest,
                      force_remote=True)

    if st.button("Labels opnieuw bewerken"):
        st.session_state.cursor = 0
        st.rerun()


# ---------------------------------------------------------------------------
# Zijbalk (roluitleg blijft hier altijd beschikbaar)
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(f"**Beoordelaar:** {st.session_state.username}")
        queue = USER_ASSIGNMENTS[st.session_state.username]
        st.markdown(f"**Toegewezen corners:** {len(queue)}")
        st.markdown(
            f"**Huidige:** "
            f"{min(st.session_state.cursor + 1, len(queue))} / {len(queue)}"
        )
        if st.button("Uitloggen"):
            # Eerst nog een keer pushen om niets kwijt te raken
            try:
                manifest = load_manifest()
                save_user_labels(st.session_state.username,
                                  st.session_state.labels,
                                  manifest, force_remote=True)
            except Exception:
                pass
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        with st.expander("ℹ️ Uitleg Labelen", expanded=False):
            st.markdown(
                "- Focus eerst op één team (aanvallers óf verdedigers — "
                "zelf doe ik vaak eerst aanvallers). Kijk welke rugnummers "
                "direct opvallen omdat ze iets specifieks doen, noteer die, "
                "en werk zo door de lijst.\n"
                "- Switch veel tussen camerastandpunten. Vooral de "
                "goal-camera's zijn nuttig, want die zitten dichter op de "
                "actie. De afspeelpositie blijft hetzelfde als je wisselt — "
                "je kunt dus gewoon pauzeren en vanuit elke hoek hetzelfde "
                "moment bekijken.\n"
                "- De Tracking 2D view helpt om in één oogopslag te zien "
                "welke spelers niet meedoen — niet mee-aanvallen "
                "(→ STAY_BACK) of niet mee-verdedigen (→ COUNTER). Ook handig "
                "om rugnummers te achterhalen wanneer die op de camera "
                "moeilijk te lezen zijn."
            )
        with st.expander("ℹ️ Uitleg aanvallerrollen", expanded=False):
            for v in ATT_ROLE_HELP.values():
                st.markdown(f"- {v}")
        with st.expander("ℹ️ Uitleg verdedigerrollen", expanded=False):
            for v in DEF_ROLE_HELP.values():
                st.markdown(f"- {v}")
            st.markdown(
                "\nAls je **MAN** kiest, vul dan ook het rugnummer in van "
                "de aanvaller die deze verdediger dekt."
            )

        st.markdown("---")
        if GIST_ENABLED:
            if st.session_state.get("_gist_dirty"):
                st.caption("💾 Wordt opgeslagen… (cloud-sync staat in de wacht)")
            else:
                last = st.session_state.get("_gist_last_save_ts")
                if last:
                    st.caption(f"☁️ Cloud-sync OK ({int(time.time() - last)} s geleden).")
                else:
                    st.caption("☁️ Cloud-sync actief.")
        else:
            st.caption("Wordt automatisch opgeslagen na elke wijziging.")


# ---------------------------------------------------------------------------
# Startpunt
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="IRR — Corner-rollen Labelen",
                       layout="wide")

    if "username" not in st.session_state:
        login_page()
        return

    manifest = load_manifest()
    render_sidebar()
    corner_page(manifest)


if __name__ == "__main__":
    main()
