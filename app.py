from __future__ import annotations
import io, json, math, os, time
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NBA Edge Lab", page_icon="🏀", layout="wide")
st.title("🏀 NBA Edge Lab")
st.caption("Game + quarter + player box-score projections • Monte Carlo • injury review • betting analysis")

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
NBA_STATS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
}

PLAYER_REQUIRED = {"game_date", "player", "team", "minutes", "points", "rebounds", "assists"}
GAME_REQUIRED = {"game_date", "home_team", "away_team", "home_score", "away_score"}
MARKET_REQUIRED = {"game_date", "market_type", "entity", "market", "line", "over_odds", "under_odds", "actual_value"}

def safe_get(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 12):
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r

@st.cache_data(ttl=900, show_spinner=False)
def fetch_scoreboard(day_string: str) -> list[dict]:
    r = safe_get(ESPN_SCOREBOARD, {"dates": day_string})
    payload = r.json()
    games = []
    for event in payload.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((x for x in competitors if x.get("homeAway") == "home"), None)
        away = next((x for x in competitors if x.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        def team(item):
            t = item.get("team", {})
            return {"name": t.get("displayName", "Unknown"), "abbr": t.get("abbreviation", ""),
                    "id": str(t.get("id", "")), "score": item.get("score", "")}
        games.append({
            "game_id": str(event.get("id", "")), "date": event.get("date", ""),
            "name": event.get("name", ""), "status": event.get("status", {}).get("type", {}).get("detail", "Scheduled"),
            "home": team(home), "away": team(away)
        })
    return games

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_injuries() -> list[dict]:
    r = safe_get(ESPN_INJURIES)
    payload = r.json()
    rows = []
    # ESPN's response structure can change; parse defensively.
    def walk(obj):
        if isinstance(obj, dict):
            if ("athlete" in obj or "player" in obj) and any(k in obj for k in ("status", "type", "details", "shortComment", "longComment")):
                athlete = obj.get("athlete") or obj.get("player") or {}
                team = obj.get("team") or {}
                rows.append({
                    "player": athlete.get("displayName") or athlete.get("fullName") or obj.get("name", ""),
                    "team": team.get("abbreviation") or team.get("displayName") or "",
                    "status": obj.get("status") or (obj.get("type") or {}).get("description", ""),
                    "detail": obj.get("shortComment") or obj.get("longComment") or obj.get("details", "")
                })
            for v in obj.values(): walk(v)
        elif isinstance(obj, list):
            for v in obj: walk(v)
    walk(payload)
    # de-duplicate rows
    seen, out = set(), []
    for row in rows:
        key = (row["player"], row["team"], row["status"], row["detail"])
        if row["player"] and key not in seen:
            seen.add(key); out.append(row)
    return out

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_nba_stats_player_logs(season: str) -> pd.DataFrame:
    """Best-effort NBA Stats request. This can be blocked from cloud hosts."""
    from nba_api.stats.endpoints import playergamelogs
    obj = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
        timeout=18,
        headers=NBA_STATS_HEADERS
    )
    df = obj.get_data_frames()[0]
    if df.empty:
        return df
    rename = {
        "GAME_DATE": "game_date", "PLAYER_NAME": "player", "TEAM_ABBREVIATION": "team",
        "MIN": "minutes", "PTS": "points", "REB": "rebounds", "AST": "assists",
        "STL": "steals", "BLK": "blocks", "TOV": "turnovers", "FG3M": "threes",
        "FGM": "fgm", "FGA": "fga", "FTM": "ftm", "FTA": "fta", "PF": "fouls",
        "MATCHUP": "matchup", "GAME_ID": "game_id"
    }
    df = df.rename(columns=rename)
    keep = [c for c in rename.values() if c in df.columns]
    df = df[keep].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df.dropna(subset=["game_date", "player"])

def normalize_player_logs(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    aliases = {
        "date": "game_date", "player_name": "player", "team_abbreviation": "team",
        "min": "minutes", "pts": "points", "reb": "rebounds", "ast": "assists",
        "stl": "steals", "blk": "blocks", "tov": "turnovers", "fg3m": "threes",
        "fgm": "fgm", "fga": "fga", "ftm": "ftm", "fta": "fta", "pf": "fouls",
        "three_pointers": "threes"
    }
    x = x.rename(columns={k:v for k,v in aliases.items() if k in x.columns and v not in x.columns})
    missing = PLAYER_REQUIRED - set(x.columns)
    if missing:
        raise ValueError("Player logs are missing required columns: " + ", ".join(sorted(missing)))
    x["game_date"] = pd.to_datetime(x["game_date"], errors="coerce")
    for c in ["minutes","points","rebounds","assists","steals","blocks","turnovers","threes","fgm","fga","ftm","fta","fouls"]:
        if c not in x.columns: x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["player"] = x["player"].astype(str).str.strip()
    x["team"] = x["team"].astype(str).str.strip().str.upper()
    x = x.dropna(subset=["game_date","player","minutes","points","rebounds","assists"])
    x = x[(x["minutes"] > 0) & (x["minutes"] <= 60)]
    return x.sort_values("game_date")

def normalize_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["game_date"] = pd.to_datetime(x["game_date"], errors="coerce")
    for c in ["home_score","away_score"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=list(GAME_REQUIRED))

def american_implied(odds):
    try: x = float(odds)
    except (TypeError, ValueError): return np.nan
    if x == 0: return np.nan
    return 100/(x+100) if x > 0 else abs(x)/(abs(x)+100)

def american_decimal(odds):
    try: x = float(odds)
    except (TypeError, ValueError): return np.nan
    if x == 0: return np.nan
    return 1+x/100 if x > 0 else 1+100/abs(x)

def no_vig_pair(over_odds, under_odds):
    po, pu = american_implied(over_odds), american_implied(under_odds)
    if not np.isfinite(po) or not np.isfinite(pu) or po+pu <= 0: return np.nan, np.nan
    return po/(po+pu), pu/(po+pu)

def weighted_player_baselines(logs: pd.DataFrame, selected_date: date, lookback: int = 15) -> pd.DataFrame:
    x = logs[logs["game_date"] < pd.Timestamp(selected_date)].copy()
    if x.empty: return pd.DataFrame()
    x = x.sort_values("game_date").groupby("player", group_keys=False).tail(lookback)
    cols = ["minutes","points","rebounds","assists","steals","blocks","turnovers","threes","fgm","fga","ftm","fta","fouls"]
    rows = []
    for player, g in x.groupby("player"):
        g = g.sort_values("game_date")
        n = len(g)
        weights = np.exp(np.linspace(-1.4, 0, n))
        weights = weights / weights.sum()
        row = {"player": player, "team": str(g["team"].iloc[-1]).upper(), "games": n,
               "last_game": g["game_date"].max()}
        for c in cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            valid = vals.notna()
            if valid.any():
                w = weights[valid.to_numpy()]
                row[c] = float(np.average(vals[valid], weights=w))
            else:
                row[c] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    # Preserve historical minutes separately from the projected minutes users may override.
    if "minutes" in out.columns:
        out["historical_minutes"] = out["minutes"].copy()
    # Shrink tiny samples toward a league-wide mean to reduce noisy rates.
    for c in cols:
        if c in out and out[c].notna().any():
            league_mean = float(out[c].mean())
            out[c] = out[c].fillna(league_mean)
            out[c] = out[c] * (out["games"]/(out["games"]+5)) + league_mean * (5/(out["games"]+5))
    return out

def team_baselines(logs: pd.DataFrame, selected_date: date, lookback: int = 12) -> pd.DataFrame:
    x = logs[logs["game_date"] < pd.Timestamp(selected_date)].copy()
    if x.empty: return pd.DataFrame()
    rows = []
    for team in sorted(x["team"].unique()):
        g = x[x["team"] == team].sort_values("game_date").tail(lookback)
        if len(g) < 2: continue
        rows.append({"team": team, "team_points": g["points"].mean(), "team_pace_proxy": g["minutes"].mean()})
    return pd.DataFrame(rows)

def apply_status_adjustment(baselines: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    x = baselines.copy()
    if "availability" not in x.columns: x["availability"] = "Available"
    if "minutes_override" not in x.columns: x["minutes_override"] = np.nan
    if "status_note" not in x.columns: x["status_note"] = ""
    status_mult = {"available":1.0, "probable":0.97, "questionable":0.82, "doubtful":0.25, "out":0.0}
    x["availability"] = x["availability"].fillna("Available")
    x["minutes"] = pd.to_numeric(x["minutes"], errors="coerce")
    for idx, row in x.iterrows():
        status = str(row.get("availability","Available")).lower()
        mult = status_mult.get(status, 1.0)
        override = row.get("minutes_override", np.nan)
        if pd.notna(override):
            x.at[idx, "minutes"] = max(0, min(48, float(override)))
        else:
            x.at[idx, "minutes"] = max(0, min(48, float(row["minutes"] or 0) * mult))
    x.loc[x["availability"].str.lower().eq("out"), "minutes"] = 0
    return x

def project_players(player_base: pd.DataFrame, home_team: str, away_team: str,
                    selected_players: pd.DataFrame, n_sims: int, rng: np.random.Generator,
                    home_target: float, away_target: float, home_minutes_cap=240, away_minutes_cap=240):
    """Generate player box-score distributions and reconcile projected points to team totals."""
    outputs = []
    quarter_rows = []
    team_targets = {home_team: home_target, away_team: away_target}
    for team in [away_team, home_team]:
        p = selected_players[selected_players["team"].astype(str).str.upper().eq(team.upper())].copy()
        if p.empty:
            continue
        p["minutes"] = pd.to_numeric(p["minutes"], errors="coerce").fillna(0).clip(0, 48)
        # Ensure expected minutes do not exceed 240. Scale active players if necessary.
        total_mins = p["minutes"].sum()
        if total_mins > 0:
            p["minutes"] *= min(1.0, 240.0 / total_mins)
        # No reliable lineup? Do not invent a roster: use only supplied/retrieved players.
        active = p[p["minutes"] > 0].copy()
        if active.empty: continue
        team_sims = {c: np.zeros((n_sims, len(active))) for c in ["points","rebounds","assists","steals","blocks","turnovers","threes","fgm","fga","ftm","fta","fouls"]}
        mins = active["minutes"].to_numpy(float)
        for col in team_sims:
            base = pd.to_numeric(active.get(col, pd.Series(np.nan, index=active.index)), errors="coerce")
            base = base.fillna(0).to_numpy(float)
            # Use historical minutes as the denominator; projected-minute overrides should change opportunity.
            hist_minutes = pd.to_numeric(active.get("historical_minutes", active["minutes"]), errors="coerce").fillna(0).to_numpy(float)
            rate = np.divide(base, np.maximum(hist_minutes, 1), out=np.zeros_like(base), where=hist_minutes > 0)
            # A modest over-dispersion creates realistic variation; not a guarantee of calibration.
            for i in range(n_sims):
                minutes_i = np.clip(rng.normal(mins, np.maximum(1.2, mins*0.09)), 0, 48)
                if col in ["points","rebounds","assists","steals","blocks","turnovers","threes","fgm","fga","ftm","fta","fouls"]:
                    mu = np.maximum(0, rate * minutes_i)
                    if col in ["points","rebounds","assists","steals","blocks","turnovers","threes","fgm","fga","ftm","fta","fouls"]:
                        vals = rng.poisson(np.maximum(mu, 0.02))
                        team_sims[col][i,:] = vals
        # Rescale player points in each simulation to match the team scoring simulation target.
        raw_pts = team_sims["points"]
        raw_means = raw_pts.mean(axis=1)
        target = max(0, team_targets[team])
        for i in range(n_sims):
            if raw_pts[i].sum() > 0:
                raw_pts[i] = np.floor(raw_pts[i] * target / raw_pts[i].sum()).astype(float)
                # Correct rounding by assigning residual to high-usage player.
                residual = int(round(target)) - int(raw_pts[i].sum())
                if residual > 0: raw_pts[i, int(np.argmax(mins))] += residual
            else:
                raw_pts[i, int(np.argmax(mins))] = round(target)
        team_sims["points"] = raw_pts
        # Make basic shot counts consistent with points in aggregate, approximate player mix.
        for i in range(n_sims):
            for j in range(len(active)):
                pts = int(team_sims["points"][i,j])
                threes = int(min(team_sims["threes"][i,j], max(0, round(pts/3))))
                fgm = int(team_sims["fgm"][i,j])
                fga = max(fgm, int(team_sims["fga"][i,j]))
                ftm = int(min(team_sims["ftm"][i,j], pts))
                fta = max(ftm, int(team_sims["fta"][i,j]))
                team_sims["threes"][i,j] = threes
                team_sims["fgm"][i,j] = fgm
                team_sims["fga"][i,j] = fga
                team_sims["ftm"][i,j] = ftm
                team_sims["fta"][i,j] = fta
        for j, (_, pl) in enumerate(active.iterrows()):
            row = {"player": pl["player"], "team": team, "minutes": float(mins[j])}
            for col, arr in team_sims.items():
                row[col] = float(np.mean(arr[:,j]))
            for stat in ["points","rebounds","assists","steals","blocks","turnovers","threes"]:
                row[f"{stat}_p10"] = float(np.percentile(team_sims[stat][:,j], 10))
                row[f"{stat}_p50"] = float(np.percentile(team_sims[stat][:,j], 50))
                row[f"{stat}_p90"] = float(np.percentile(team_sims[stat][:,j], 90))
            row["fg_pct"] = row["fgm"]/row["fga"] if row["fga"] else np.nan
            row["ft_pct"] = row["ftm"]/row["fta"] if row["fta"] else np.nan
            outputs.append(row)
    return pd.DataFrame(outputs)

def simulate_quarters(home_score: float, away_score: float, n_sims: int, rng: np.random.Generator):
    # NBA quarter scoring weights: Q1/Q3 near average; Q2/Q4 slightly more variable.
    weights = np.array([0.249, 0.251, 0.249, 0.251])
    rows = []
    for team, score in [("Home", home_score), ("Away", away_score)]:
        sims = rng.multinomial(max(0, int(round(score))), weights, size=n_sims)
        means = sims.mean(axis=0)
        rows.append({"team":team, "Q1":means[0], "Q2":means[1], "Q3":means[2], "Q4":means[3],
                     "Halftime":means[0]+means[1], "Final":means.sum()})
    return pd.DataFrame(rows)

def simulate_game(home_pts: float, away_pts: float, n_sims: int, seed: int):
    rng = np.random.default_rng(seed)
    # Game totals are anchored to recent player-derived team baselines with realistic variance.
    home = np.maximum(70, rng.normal(home_pts, 11.5, n_sims))
    away = np.maximum(70, rng.normal(away_pts, 11.5, n_sims))
    ties = home == away
    # Simple OT handling for final-score distributions; expected extra points conditional on tie.
    for i in np.where(ties)[0]:
        home[i] += rng.poisson(4.5)
        away[i] += rng.poisson(4.5)
    return home, away

def make_default_availability(players: pd.DataFrame, injury_rows: list[dict]) -> pd.DataFrame:
    out = players[["player","team","minutes"]].copy()
    out["availability"] = "Available"
    out["minutes_override"] = np.nan
    out["status_note"] = ""
    if injury_rows:
        for idx, row in out.iterrows():
            match = next((inj for inj in injury_rows if inj.get("player","").lower() == str(row["player"]).lower()), None)
            if match:
                s = str(match.get("status","")).lower()
                if "out" in s: out.at[idx,"availability"] = "Out"
                elif "doubt" in s: out.at[idx,"availability"] = "Doubtful"
                elif "question" in s: out.at[idx,"availability"] = "Questionable"
                elif "prob" in s: out.at[idx,"availability"] = "Probable"
                out.at[idx,"status_note"] = str(match.get("detail",""))
    return out

def show_data_status(source: str, logs: pd.DataFrame | None, injury_ok: bool):
    st.markdown("#### Data status")
    c1,c2,c3 = st.columns(3)
    c1.metric("Player log source", source)
    c2.metric("Player-log rows", f"{len(logs):,}" if logs is not None else "0")
    c3.metric("Injury feed", "Retrieved" if injury_ok else "Unavailable")
    st.caption("Projections are estimates, not guarantees. Verify late scratches and starters before tip-off.")

with st.sidebar:
    st.header("Model controls")
    selected_day = st.date_input("Game date", value=date.today())
    lookback = st.slider("Recent player games", 5, 30, 15)
    sims = st.select_slider("Monte Carlo simulations", options=[500,1000,2500,5000,10000], value=2500)
    seed = st.number_input("Random seed (reproducibility)", min_value=1, max_value=999999, value=42)
    st.caption("Use a fixed seed to compare revisions; don't tune parameters on the final test period.")

st.header("1. Load data")
c1,c2 = st.columns(2)
with c1:
    st.markdown("**Historical player game logs**")
    player_upload = st.file_uploader("Upload player logs CSV", type=["csv"], key="player_csv",
        help="Required columns: game_date, player, team, minutes, points, rebounds, assists. See templates below.")
with c2:
    st.markdown("**Historical game results (optional for validation)**")
    game_upload = st.file_uploader("Upload team game results CSV", type=["csv"], key="game_csv")

use_nba_stats = st.checkbox("Try automatic NBA Stats player logs (may be blocked on cloud hosts)", value=False)
season = st.selectbox("NBA Stats season", ["2026-27","2025-26","2024-25","2023-24","2022-23"], index=0)
player_logs, source = None, "Not loaded"
if player_upload is not None:
    try:
        player_logs = normalize_player_logs(pd.read_csv(player_upload))
        source = "Uploaded CSV"
    except Exception as e:
        st.error(f"Could not read player CSV: {e}")
elif use_nba_stats:
    with st.spinner("Trying NBA Stats (may be blocked by the hosting provider)…"):
        try:
            raw = fetch_nba_stats_player_logs(season)
            player_logs = normalize_player_logs(raw)
            source = f"NBA Stats {season}"
        except Exception as e:
            st.warning(f"NBA Stats could not be retrieved: {type(e).__name__}: {e}")
            st.info("Upload a historical player-log CSV to continue. The app will not fabricate missing player data.")
if player_logs is not None:
    st.success(f"Loaded {len(player_logs):,} player-game rows from {source}.")
else:
    st.warning("Player-level projections require real historical player-game data. Upload the CSV or try NBA Stats. No player stats are invented.")
    st.download_button("Download player-log CSV template",
        pd.DataFrame(columns=["game_date","player","team","minutes","points","rebounds","assists","steals","blocks","turnovers","threes","fgm","fga","ftm","fta","fouls"]).to_csv(index=False),
        "player_game_logs_template.csv","text/csv")

st.header("2. Schedule, injuries and matchup")
games, injury_rows, injury_ok = [], [], False
try:
    games = fetch_scoreboard(selected_day.strftime("%Y%m%d"))
    st.success(f"ESPN scoreboard retrieved: {len(games)} game(s) for {selected_day}.")
except Exception as e:
    st.warning(f"ESPN scoreboard unavailable: {type(e).__name__}: {e}. You can still use manual matchup selection.")
try:
    injury_rows = fetch_injuries()
    injury_ok = True
except Exception as e:
    st.warning(f"Injury feed unavailable: {type(e).__name__}: {e}. Review official reports manually.")

team_options = sorted(set(
    (player_logs["team"].astype(str).str.upper().tolist() if player_logs is not None else [])
    + [g["home"]["abbr"] for g in games] + [g["away"]["abbr"] for g in games]
))
game_label_to_obj = {}
for g in games:
    label = f'{g["away"]["abbr"]} @ {g["home"]["abbr"]} — {g["status"]}'
    game_label_to_obj[label] = g
if game_label_to_obj:
    chosen_label = st.selectbox("Select game", list(game_label_to_obj))
    chosen = game_label_to_obj[chosen_label]
    home_team, away_team = chosen["home"]["abbr"], chosen["away"]["abbr"]
else:
    st.info("No live schedule available; choose teams manually.")
    if len(team_options) < 2:
        team_options = ["BOS","NYK","LAL","GSW","DEN","MIL","PHX","OKC","CLE","DAL","MIN","IND","MIA","ORL","SAC","PHI","ATL","CHI","HOU","MEM","NOP","POR","SAS","TOR","UTA","WAS","BKN","CHA","DET"]
    home_team = st.selectbox("Home team", team_options, index=min(0,len(team_options)-1))
    away_default = 1 if len(team_options)>1 else 0
    away_team = st.selectbox("Away team", team_options, index=away_default)
    if home_team == away_team: st.warning("Choose two different teams.")

if player_logs is not None and not player_logs.empty:
    base = weighted_player_baselines(player_logs, selected_day, lookback)
    if base.empty:
        st.error("No player games exist before the selected date. Choose a later date or upload historical logs.")
    else:
        # Keep the most recent team assignment for each player. Users can edit availability/minutes.
        team_names = {home_team.upper(), away_team.upper()}
        matchup_base = base[base["team"].astype(str).str.upper().isin(team_names)].copy()
        if matchup_base.empty:
            st.error(f"No historical players found for {home_team} or {away_team}. Check team abbreviations in the CSV.")
        else:
            st.subheader("Player availability and minutes")
            avail = make_default_availability(matchup_base, injury_rows)
            # Avoid duplicate rows when multiple same-name entries occur.
            edited = st.data_editor(avail, use_container_width=True, hide_index=True,
                column_config={
                    "availability": st.column_config.SelectboxColumn("Availability", options=["Available","Probable","Questionable","Doubtful","Out"], required=True),
                    "minutes_override": st.column_config.NumberColumn("Minutes override", min_value=0, max_value=48, step=1),
                    "status_note": st.column_config.TextColumn("Injury note")
                }, key="availability_editor")
            adjusted = apply_status_adjustment(matchup_base.merge(edited[["player","team","availability","minutes_override","status_note"]],
                on=["player","team"], how="left"), edited)
            # Team scoring baseline scales each player's recent production by projected vs historical minutes.
            # This is a transparent baseline, not a trained opponent-adjusted model.
            hplayers = adjusted[adjusted["team"].astype(str).str.upper().eq(home_team.upper())]
            aplayers = adjusted[adjusted["team"].astype(str).str.upper().eq(away_team.upper())]
            def team_pts(p):
                if p.empty: return 112.0
                hist = pd.to_numeric(p.get("historical_minutes", p["minutes"]), errors="coerce").fillna(0).to_numpy(float)
                proj = pd.to_numeric(p["minutes"], errors="coerce").fillna(0).to_numpy(float)
                pts = pd.to_numeric(p["points"], errors="coerce").fillna(0).to_numpy(float)
                scaled = np.divide(pts * proj, np.maximum(hist, 1), out=np.zeros_like(pts), where=hist > 0)
                return float(np.clip(scaled.sum(), 75, 145))
            home_target = team_pts(hplayers) + 2.0
            away_target = team_pts(aplayers)
            # The 2-point home-court prior is an explicit placeholder; calibrate it with historical validation.
            st.subheader("3. Game and quarter projections")
            game_home_sims, game_away_sims = simulate_game(home_target, away_target, int(sims), int(seed))
            home_mean, away_mean = float(game_home_sims.mean()), float(game_away_sims.mean())
            c1,c2,c3,c4 = st.columns(4)
            c1.metric(f"{home_team} projected score", f"{home_mean:.1f}")
            c2.metric(f"{away_team} projected score", f"{away_mean:.1f}")
            c3.metric("Projected total", f"{home_mean+away_mean:.1f}")
            c4.metric(f"{home_team} win probability", f"{(game_home_sims>game_away_sims).mean()*100:.1f}%")
            st.caption("Current baseline uses recent player averages and a simple scoring distribution. Opponent-adjusted quarter/player calibration is not yet proven.")
            qhome = simulate_quarters(home_mean, away_mean, int(sims), np.random.default_rng(int(seed)+1))
            qhome = qhome.rename(columns={"team":"Side"})
            st.markdown("**Expected quarter scores**")
            qtab = pd.DataFrame([
                {"Team":home_team,"Q1":home_mean*.249,"Q2":home_mean*.251,"Q3":home_mean*.249,"Q4":home_mean*.251,"Halftime":home_mean*.5,"Final":home_mean},
                {"Team":away_team,"Q1":away_mean*.249,"Q2":away_mean*.251,"Q3":away_mean*.249,"Q4":away_mean*.251,"Halftime":away_mean*.5,"Final":away_mean}
            ])
            st.dataframe(qtab.round(1), use_container_width=True, hide_index=True)
            st.markdown("**Score distribution (10th / 50th / 90th percentile)**")
            dist = pd.DataFrame([
                {"Team":home_team,"P10":np.percentile(game_home_sims,10),"Median":np.percentile(game_home_sims,50),"P90":np.percentile(game_home_sims,90)},
                {"Team":away_team,"P10":np.percentile(game_away_sims,10),"Median":np.percentile(game_away_sims,50),"P90":np.percentile(game_away_sims,90)}
            ])
            st.dataframe(dist.round(1), use_container_width=True, hide_index=True)
            st.subheader("4. Full projected player box scores")
            # Use target points as team total, but player means are recent player logs; reconcile player points.
            player_box = project_players(base, home_team, away_team, adjusted, int(sims), np.random.default_rng(int(seed)+2), home_mean, away_mean)
            if player_box.empty:
                st.warning("No active players available to project.")
            else:
                show_cols = ["player","team","minutes","points","rebounds","assists","steals","blocks","turnovers","threes","fgm","fga","fg_pct","ftm","fta","ft_pct","fouls","points_p10","points_p50","points_p90","rebounds_p10","rebounds_p90","assists_p10","assists_p90"]
                show_cols = [c for c in show_cols if c in player_box.columns]
                st.dataframe(player_box[show_cols].round(2).sort_values(["team","minutes"], ascending=[True,False]), use_container_width=True, hide_index=True)
                st.download_button("Download projected player box scores CSV", player_box.to_csv(index=False).encode(), "nba_projected_player_box_scores.csv","text/csv")
                st.download_button("Download game projections JSON", json.dumps({"date":str(selected_day),"home_team":home_team,"away_team":away_team,
                    "home_score":home_mean,"away_score":away_mean,"total":home_mean+away_mean,
                    "home_win_probability":float((game_home_sims>game_away_sims).mean())}, indent=2).encode(),
                    "nba_game_projection.json","application/json")
            st.subheader("5. Player prop analysis")
            if not player_box.empty:
                prop_player = st.selectbox("Player", sorted(player_box["player"].unique()))
                prop_market = st.selectbox("Stat", ["points","rebounds","assists","threes","steals","blocks","turnovers"])
                prop_line = st.number_input("Prop line", min_value=0.0, max_value=100.0, value=20.5, step=0.5)
                col1,col2,col3 = st.columns(3)
                player_row = player_box[player_box["player"]==prop_player].iloc[0]
                mean_val = float(player_row.get(prop_market,0))
                # Approximate distribution using mean and variance proxy; explicitly marked estimate.
                sd = max(1.0, math.sqrt(max(0.1, mean_val*1.25)))
                over_p = 1 - 0.5*(1+math.erf((prop_line-mean_val)/(sd*math.sqrt(2))))
                col1.metric("Model mean", f"{mean_val:.1f}")
                col2.metric("Approx. over probability", f"{over_p*100:.1f}%")
                col3.metric("Approx. under probability", f"{(1-over_p)*100:.1f}%")
                st.caption("Prop probability is a rough distribution approximation in this starter build. Don't treat it as a validated betting edge.")
            st.subheader("6. Odds and expected value calculator")
            with st.form("ev_form"):
                side = st.radio("Bet side", ["Over","Under"], horizontal=True)
                model_prob = st.number_input("Your model probability (%)", min_value=0.1, max_value=99.9, value=55.0, step=0.5)/100
                odds = st.number_input("American odds", min_value=-10000, max_value=10000, value=-110, step=5)
                submitted = st.form_submit_button("Calculate expected value")
            if submitted:
                dec = american_decimal(odds)
                implied = american_implied(odds)
                ev = model_prob*(dec-1)-(1-model_prob)
                st.write({"decimal_odds":round(dec,4),"implied_probability":round(implied*100,2),
                          "model_edge_percentage_points":round((model_prob-implied)*100,2),
                          "expected_profit_per_1_unit":round(ev,4)})
            st.subheader("7. Export and validation")
            st.info("Use the separate validation_app.py to evaluate chronological predictions against held-out actuals. Do not infer profitability from in-sample fit.")
else:
    st.info("Once player logs are loaded, the app will calculate player baselines and generate projections for the selected matchup.")

with st.expander("Required CSV formats and limitations"):
    st.markdown("""
**Player logs** — one row per player per game. Required: `game_date, player, team, minutes, points, rebounds, assists`. Optional: `steals, blocks, turnovers, threes, fgm, fga, ftm, fta, fouls`.

**Team game results** — `game_date, home_team, away_team, home_score, away_score`.

**Market history for validation** — `game_date, market_type, entity, market, line, over_odds, under_odds, actual_value`. Include only odds captured before the game.

The ESPN scoreboard/injury feeds are best-effort and can change. NBA Stats can block cloud-hosted requests. Manual lineup/minutes overrides are available. This model is a research prototype, not a proven betting system. Quarter scores are currently allocated by expected quarter weights; historical quarter-specific calibration is a future improvement. Never treat stale or missing data as live.
""")
