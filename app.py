from __future__ import annotations

import json
import math
import os
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NBA Edge Lab", page_icon="🏀", layout="wide")
st.title("🏀 NBA Edge Lab")
st.caption("NBA schedule • injury review • player projections • score simulations • prop analysis")

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
BDL_STATS = "https://api.balldontlie.io/v1/player_game_stats"

PLAYER_REQUIRED = {"game_date", "player", "team", "minutes", "points", "rebounds", "assists"}
STAT_COLS = ["minutes", "points", "rebounds", "assists", "steals", "blocks", "turnovers", "threes", "fgm", "fga", "ftm", "fta", "fouls"]


def http_get(url: str, *, params=None, headers=None, timeout=15):
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


@st.cache_data(ttl=900, show_spinner=False)
def fetch_scoreboard(day_string: str) -> list[dict]:
    payload = http_get(ESPN_SCOREBOARD, params={"dates": day_string}).json()
    games = []
    for event in payload.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        def team(c):
            t = c.get("team") or {}
            return {"name": t.get("displayName", "Unknown"), "abbr": t.get("abbreviation", ""),
                    "id": str(t.get("id", "")), "score": c.get("score", "")}
        games.append({"game_id": str(event.get("id", "")), "date": event.get("date", ""),
                      "name": event.get("name", ""),
                      "status": (event.get("status") or {}).get("type", {}).get("detail", "Scheduled"),
                      "home": team(home), "away": team(away)})
    return games


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_injuries() -> list[dict]:
    payload = http_get(ESPN_INJURIES).json()
    rows = []
    def walk(obj):
        if isinstance(obj, dict):
            if ("athlete" in obj or "player" in obj) and any(k in obj for k in ("status", "type", "details", "shortComment", "longComment")):
                athlete = obj.get("athlete") or obj.get("player") or {}
                team = obj.get("team") or {}
                status_obj = obj.get("type") or {}
                rows.append({"player": athlete.get("displayName") or athlete.get("fullName") or obj.get("name", ""),
                             "team": team.get("abbreviation") or team.get("displayName") or "",
                             "status": obj.get("status") or status_obj.get("description", ""),
                             "detail": obj.get("shortComment") or obj.get("longComment") or obj.get("details", "")})
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
    walk(payload)
    unique, result = set(), []
    for row in rows:
        key = tuple(row.values())
        if row["player"] and key not in unique:
            unique.add(key)
            result.append(row)
    return result


def fetch_balldontlie_logs(season: str, api_key: str, max_pages: int = 100) -> pd.DataFrame:
    """Fetch real player game logs. Requires an API plan with player_game_stats access."""
    if not api_key:
        raise ValueError("Add BALLDONTLIE_API_KEY in Streamlit Secrets first.")
    year = int(season.split("-")[0])
    headers = {"Authorization": api_key, "Accept": "application/json"}
    rows, cursor = [], None
    with requests.Session() as session:
        for _ in range(max_pages):
            params = {"seasons[]": year, "per_page": 100}
            if cursor is not None:
                params["cursor"] = cursor
            response = session.get(BDL_STATS, headers=headers, params=params, timeout=(5, 25))
            if response.status_code in (401, 403):
                raise RuntimeError("API key invalid or plan does not include player-game statistics.")
            if response.status_code == 429:
                raise RuntimeError("API rate limit reached. Try again later.")
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("data", [])
            for item in batch:
                player, team, game = item.get("player") or {}, item.get("team") or {}, item.get("game") or {}
                rows.append({
                    "game_date": game.get("date"),
                    "player": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                    "team": team.get("abbreviation", ""), "minutes": item.get("min"),
                    "points": item.get("pts"), "rebounds": item.get("reb"), "assists": item.get("ast"),
                    "steals": item.get("stl"), "blocks": item.get("blk"), "turnovers": item.get("turnover"),
                    "threes": item.get("fg3m"), "fgm": item.get("fgm"), "fga": item.get("fga"),
                    "ftm": item.get("ftm"), "fta": item.get("fta"), "fouls": item.get("pf")})
            cursor = (payload.get("meta") or {}).get("next_cursor")
            if not cursor or not batch:
                break
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No player-game rows returned. Check season and API access.")
    return df


def normalize_player_logs(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    aliases = {"date":"game_date", "player_name":"player", "team_abbreviation":"team", "min":"minutes",
               "pts":"points", "reb":"rebounds", "ast":"assists", "stl":"steals", "blk":"blocks",
               "tov":"turnovers", "fg3m":"threes", "three_pointers":"threes", "pf":"fouls"}
    x = x.rename(columns={k:v for k,v in aliases.items() if k in x.columns and v not in x.columns})
    missing = PLAYER_REQUIRED - set(x.columns)
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(sorted(missing)))
    x["game_date"] = pd.to_datetime(x["game_date"], errors="coerce")
    for col in STAT_COLS:
        if col not in x.columns:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["player"] = x["player"].astype(str).str.strip()
    x["team"] = x["team"].astype(str).str.strip().str.upper()
    x = x.dropna(subset=["game_date", "player", "minutes", "points", "rebounds", "assists"])
    x = x[(x["minutes"] > 0) & (x["minutes"] <= 60)]
    if x.empty:
        raise ValueError("No valid player-game rows remain after cleaning the file.")
    return x.sort_values("game_date")


def weighted_baselines(logs: pd.DataFrame, selected_date: date, lookback: int) -> pd.DataFrame:
    x = logs[logs.game_date < pd.Timestamp(selected_date)].copy()
    if x.empty:
        return pd.DataFrame()
    x = x.sort_values("game_date").groupby("player", group_keys=False).tail(lookback)
    rows = []
    for player, group in x.groupby("player"):
        group = group.sort_values("game_date")
        weights = np.exp(np.linspace(-1.4, 0, len(group)))
        weights /= weights.sum()
        row = {"player": player, "team": str(group.team.iloc[-1]).upper(), "games": len(group),
               "last_game": group.game_date.max()}
        for col in STAT_COLS:
            vals = pd.to_numeric(group[col], errors="coerce")
            mask = vals.notna()
            row[col] = float(np.average(vals[mask], weights=weights[mask.to_numpy()])) if mask.any() else np.nan
        row["historical_minutes"] = row["minutes"]
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in STAT_COLS:
        if out[col].notna().any():
            mean = float(out[col].mean())
            out[col] = out[col].fillna(mean)
            out[col] = out[col] * out.games / (out.games + 5) + mean * 5 / (out.games + 5)
    return out


def make_availability(players: pd.DataFrame, injury_rows: list[dict]) -> pd.DataFrame:
    out = players[["player", "team", "minutes"]].copy()
    out["availability"] = "Available"
    out["minutes_override"] = np.nan
    out["status_note"] = ""
    by_name = {str(r.get("player", "")).strip().lower(): r for r in injury_rows}
    for idx, row in out.iterrows():
        injury = by_name.get(str(row.player).strip().lower())
        if not injury:
            continue
        status = str(injury.get("status", "")).lower()
        if "out" in status: out.at[idx, "availability"] = "Out"
        elif "doubt" in status: out.at[idx, "availability"] = "Doubtful"
        elif "question" in status: out.at[idx, "availability"] = "Questionable"
        elif "prob" in status: out.at[idx, "availability"] = "Probable"
        out.at[idx, "status_note"] = str(injury.get("detail", ""))
    return out


def apply_availability(base: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    x = base.merge(edited[["player", "team", "availability", "minutes_override", "status_note"]],
                   on=["player", "team"], how="left")
    mult = {"available":1.0, "probable":0.97, "questionable":0.82, "doubtful":0.25, "out":0.0}
    for idx, row in x.iterrows():
        status = str(row.get("availability", "Available")).lower()
        override = row.get("minutes_override", np.nan)
        mins = float(override) if pd.notna(override) else float(row.get("minutes", 0) or 0) * mult.get(status, 1.0)
        x.at[idx, "minutes"] = 0.0 if status == "out" else float(np.clip(mins, 0, 48))
    x["minutes"] = pd.to_numeric(x.minutes, errors="coerce").fillna(0)
    return x


def simulate_game(home_mean: float, away_mean: float, n: int, seed: int):
    rng = np.random.default_rng(seed)
    home = np.maximum(70, rng.normal(home_mean, 11.5, n))
    away = np.maximum(70, rng.normal(away_mean, 11.5, n))
    ties = np.isclose(home, away, atol=0.5)
    for i in np.where(ties)[0]:
        home[i] += rng.poisson(4.5)
        away[i] += rng.poisson(4.5)
    return home, away


def project_players(players: pd.DataFrame, home_team: str, away_team: str, home_target: float,
                    away_target: float, n_sims: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    outputs = []
    targets = {home_team.upper(): home_target, away_team.upper(): away_target}
    for team in [away_team.upper(), home_team.upper()]:
        p = players[players.team.str.upper().eq(team)].copy()
        if p.empty:
            continue
        p["minutes"] = pd.to_numeric(p.minutes, errors="coerce").fillna(0).clip(0, 48)
        if p.minutes.sum() > 240:
            p["minutes"] *= 240 / p.minutes.sum()
        p = p[p.minutes > 0].copy()
        if p.empty:
            continue
        historical = pd.to_numeric(p.historical_minutes, errors="coerce").fillna(0).to_numpy(float)
        mins = p.minutes.to_numpy(float)
        sim_stats = {}
        for col in ["points", "rebounds", "assists", "steals", "blocks", "turnovers", "threes"]:
            base = pd.to_numeric(p[col], errors="coerce").fillna(0).to_numpy(float)
            rate = np.divide(base, np.maximum(historical, 1), out=np.zeros_like(base), where=historical > 0)
            sim_stats[col] = rng.poisson(np.maximum(0.02, rate[None, :] * rng.normal(mins, np.maximum(1, mins * .08), (n_sims, len(p)))))
        pts = sim_stats["points"].astype(float)
        target = max(0, targets[team])
        for i in range(n_sims):
            total = pts[i].sum()
            if total > 0:
                pts[i] *= target / total
                pts[i] = np.floor(pts[i])
                residual = int(round(target) - pts[i].sum())
                if residual > 0:
                    pts[i, int(np.argmax(mins))] += residual
        sim_stats["points"] = pts
        for j, (_, player) in enumerate(p.iterrows()):
            row = {"player": player.player, "team": team, "minutes": mins[j]}
            for col, values in sim_stats.items():
                row[col] = float(values[:, j].mean())
                row[f"{col}_p10"] = float(np.percentile(values[:, j], 10))
                row[f"{col}_p50"] = float(np.percentile(values[:, j], 50))
                row[f"{col}_p90"] = float(np.percentile(values[:, j], 90))
            row["fgm"] = float(player.fgm) if pd.notna(player.fgm) else np.nan
            row["fga"] = float(player.fga) if pd.notna(player.fga) else np.nan
            row["ftm"] = float(player.ftm) if pd.notna(player.ftm) else np.nan
            row["fta"] = float(player.fta) if pd.notna(player.fta) else np.nan
            row["fouls"] = float(player.fouls) if pd.notna(player.fouls) else np.nan
            row["fg_pct"] = row["fgm"] / row["fga"] if pd.notna(row["fga"]) and row["fga"] else np.nan
            row["ft_pct"] = row["ftm"] / row["fta"] if pd.notna(row["fta"]) and row["fta"] else np.nan
            outputs.append(row)
    return pd.DataFrame(outputs)


def american_implied(odds):
    try: x = float(odds)
    except (TypeError, ValueError): return np.nan
    if x == 0: return np.nan
    return 100 / (x + 100) if x > 0 else abs(x) / (abs(x) + 100)


def american_decimal(odds):
    x = float(odds)
    if x == 0: return np.nan
    return 1 + x / 100 if x > 0 else 1 + 100 / abs(x)


with st.sidebar:
    st.header("Model controls")
    selected_day = st.date_input("Game date", value=date.today())
    lookback = st.slider("Recent games per player", 5, 30, 15)
    sims = st.select_slider("Monte Carlo simulations", options=[500, 1000, 2500, 5000, 10000], value=2500)
    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42)

st.header("1. Load historical player data")
up1, up2 = st.columns(2)
with up1:
    uploaded = st.file_uploader("Upload player-game logs CSV", type=["csv"])
with up2:
    st.markdown("Required columns: `game_date, player, team, minutes, points, rebounds, assists`.")

with st.expander("Optional: retrieve player logs from BALLDONTLIE API"):
    st.write("Requires an API key and a plan that includes player-game statistics. If you don't have one, use the CSV upload.")
    api_key = st.text_input("BALLDONTLIE API key", type="password", value=st.secrets.get("BALLDONTLIE_API_KEY", os.environ.get("BALLDONTLIE_API_KEY", "")))
    api_season = st.selectbox("API season", ["2026-27", "2025-26", "2024-25", "2023-24"], index=0)
    fetch_api = st.button("Fetch historical player logs")

logs, source = None, "Not loaded"
if uploaded is not None:
    try:
        logs = normalize_player_logs(pd.read_csv(uploaded))
        source = "Uploaded CSV"
        st.success(f"Loaded {len(logs):,} player-game records from CSV.")
    except Exception as exc:
        st.error(f"CSV could not be loaded: {exc}")
elif "fetch_api" in locals() and fetch_api:
    try:
        with st.spinner("Fetching real player-game data..."):
            logs = normalize_player_logs(fetch_balldontlie_logs(api_season, api_key))
        source = f"BALLDONTLIE {api_season}"
        st.session_state["api_logs"] = logs
        st.success(f"Loaded {len(logs):,} player-game records.")
    except Exception as exc:
        st.error(f"Player data could not be loaded: {exc}")
elif "api_logs" in st.session_state:
    logs = st.session_state["api_logs"]
    source = "BALLDONTLIE (session cache)"

if logs is None:
    st.info("No historical player logs loaded. The app will not invent player projections. Upload a CSV or use the API option above.")
    template = pd.DataFrame(columns=["game_date", "player", "team", *STAT_COLS])
    st.download_button("Download player CSV template", template.to_csv(index=False), "player_game_logs_template.csv", "text/csv")

st.header("2. Schedule and injury review")
games, injuries, injury_ok = [], [], False
try:
    games = fetch_scoreboard(selected_day.strftime("%Y%m%d"))
    st.success(f"ESPN scoreboard retrieved: {len(games)} game(s) for {selected_day}.")
except Exception as exc:
    st.warning(f"ESPN scoreboard unavailable: {exc}")
try:
    injuries = fetch_injuries()
    injury_ok = True
    st.success(f"ESPN injury feed retrieved: {len(injuries)} entries.")
except Exception as exc:
    st.warning(f"Injury feed unavailable: {exc}. Verify official reports manually.")

team_options = sorted(set((logs.team.tolist() if logs is not None else []) + [g["home"]["abbr"] for g in games] + [g["away"]["abbr"] for g in games]))
if len(team_options) < 2:
    team_options = sorted(set(team_options + "ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS".split()))
labels = {f'{g["away"]["abbr"]} @ {g["home"]["abbr"]} — {g["status"]}': g for g in games}
if labels:
    chosen = st.selectbox("Select game", list(labels))
    game = labels[chosen]
    home_team, away_team = game["home"]["abbr"], game["away"]["abbr"]
else:
    c1, c2 = st.columns(2)
    home_team = c1.selectbox("Home team", team_options, index=0)
    away_team = c2.selectbox("Away team", team_options, index=1)

if logs is not None:
    base = weighted_baselines(logs, selected_day, lookback)
    if base.empty:
        st.error("No historical games before the selected date. Choose a later date or upload older player logs.")
    else:
        base = base[base.team.isin([home_team.upper(), away_team.upper()])].copy()
        if base.empty:
            st.error(f"No player history found for {home_team} or {away_team}. Check team abbreviations in the CSV.")
        else:
            st.subheader("Player availability and projected minutes")
            avail = make_availability(base, injuries)
            edited = st.data_editor(avail, use_container_width=True, hide_index=True, key="availability_editor",
                column_config={"availability": st.column_config.SelectboxColumn("Availability", options=["Available", "Probable", "Questionable", "Doubtful", "Out"]),
                               "minutes_override": st.column_config.NumberColumn("Minutes override", min_value=0, max_value=48, step=1),
                               "status_note": st.column_config.TextColumn("Injury note")})
            players = apply_availability(base, edited)
            def team_target(team):
                p = players[players.team.eq(team.upper())]
                if p.empty: return 112.0
                hist = p.historical_minutes.fillna(0).to_numpy(float)
                mins = p.minutes.fillna(0).to_numpy(float)
                pts = p.points.fillna(0).to_numpy(float)
                scaled = np.divide(pts * mins, np.maximum(hist, 1), out=np.zeros_like(pts), where=hist > 0)
                return float(np.clip(scaled.sum(), 75, 145))
            home_target = team_target(home_team) + 2.0
            away_target = team_target(away_team)
            st.subheader("3. Game and quarter projections")
            home_sims, away_sims = simulate_game(home_target, away_target, int(sims), int(seed))
            hm, am = float(home_sims.mean()), float(away_sims.mean())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{home_team} projected points", f"{hm:.1f}")
            c2.metric(f"{away_team} projected points", f"{am:.1f}")
            c3.metric("Projected total", f"{hm + am:.1f}")
            c4.metric(f"{home_team} win probability", f"{(home_sims > away_sims).mean() * 100:.1f}%")
            st.caption("Research prototype only: these projections are not yet historically calibrated or proven profitable.")
            qweights = [0.249, 0.251, 0.249, 0.251]
            qtab = pd.DataFrame([{"Team": home_team, "Q1": hm*qweights[0], "Q2": hm*qweights[1], "Q3": hm*qweights[2], "Q4": hm*qweights[3], "Halftime": hm*.5, "Final": hm},
                                 {"Team": away_team, "Q1": am*qweights[0], "Q2": am*qweights[1], "Q3": am*qweights[2], "Q4": am*qweights[3], "Halftime": am*.5, "Final": am}])
            st.dataframe(qtab.round(1), use_container_width=True, hide_index=True)
            dist = pd.DataFrame([{"Team": home_team, "P10": np.percentile(home_sims, 10), "Median": np.percentile(home_sims, 50), "P90": np.percentile(home_sims, 90)},
                                 {"Team": away_team, "P10": np.percentile(away_sims, 10), "Median": np.percentile(away_sims, 50), "P90": np.percentile(away_sims, 90)}])
            st.markdown("**Score range (10th / median / 90th percentile)**")
            st.dataframe(dist.round(1), use_container_width=True, hide_index=True)
            st.subheader("4. Projected player box scores")
            box = project_players(players, home_team, away_team, hm, am, int(sims), int(seed) + 2)
            if box.empty:
                st.warning("No active players available for this matchup.")
            else:
                show = [c for c in ["player", "team", "minutes", "points", "rebounds", "assists", "steals", "blocks", "turnovers", "threes", "fgm", "fga", "fg_pct", "ftm", "fta", "ft_pct", "fouls", "points_p10", "points_p50", "points_p90", "rebounds_p10", "rebounds_p90", "assists_p10", "assists_p90"] if c in box.columns]
                st.dataframe(box[show].round(2).sort_values(["team", "minutes"], ascending=[True, False]), use_container_width=True, hide_index=True)
                st.download_button("Download projected player box scores", box.to_csv(index=False), "nba_projected_player_box_scores.csv", "text/csv")
                st.download_button("Download game projection JSON", json.dumps({"date": str(selected_day), "home_team": home_team, "away_team": away_team, "home_score": hm, "away_score": am, "total": hm+am, "home_win_probability": float((home_sims > away_sims).mean())*100}, indent=2), "nba_game_projection.json", "application/json")
                st.subheader("5. Player prop estimate")
                player_name = st.selectbox("Player", sorted(box.player.unique()))
                market = st.selectbox("Stat", ["points", "rebounds", "assists", "threes", "steals", "blocks", "turnovers"])
                line = st.number_input("Prop line", min_value=0.0, max_value=100.0, value=20.5, step=0.5)
                row = box[box.player.eq(player_name)].iloc[0]
                mean = float(row.get(market, 0))
                sd = max(1.0, math.sqrt(max(.1, mean * 1.25)))
                over = 1 - .5 * (1 + math.erf((line - mean) / (sd * math.sqrt(2))))
                a, b, c = st.columns(3)
                a.metric("Model mean", f"{mean:.1f}")
                b.metric("Approx. over", f"{over*100:.1f}%")
                c.metric("Approx. under", f"{(1-over)*100:.1f}%")
                st.warning("Prop probabilities are rough estimates, not validated betting edges. No odds feed is connected.")
            st.subheader("6. Odds and expected value calculator")
            with st.form("ev_form"):
                prob = st.number_input("Model probability (%)", min_value=0.1, max_value=99.9, value=55.0, step=0.5) / 100
                odds = st.number_input("American odds", min_value=-10000, max_value=10000, value=-110, step=5)
                submitted = st.form_submit_button("Calculate EV")
            if submitted:
                if odds == 0:
                    st.error("American odds cannot be zero.")
                else:
                    dec = american_decimal(odds)
                    implied = american_implied(odds)
                    ev = prob * (dec - 1) - (1 - prob)
                    st.write({"decimal_odds": round(dec, 4), "implied_probability_pct": round(implied*100, 2), "edge_percentage_points": round((prob-implied)*100, 2), "expected_profit_per_unit": round(ev, 4)})
else:
    st.info("Load real historical player logs to unlock player and game projections.")

with st.expander("Data formats and limitations"):
    st.markdown("""
Player CSV: one row per player per game. Required columns: `game_date, player, team, minutes, points, rebounds, assists`. Optional columns: `steals, blocks, turnovers, threes, fgm, fga, ftm, fta, fouls`.

ESPN schedule and injury feeds are best-effort and may change. NBA Stats requests can time out from cloud hosts. BALLDONTLIE historical player logs require an API plan that permits the endpoint. If data is unavailable, this app does not invent player histories. Quarter scores use fixed quarter weights and player prop probabilities are rough approximations; neither is calibrated or proven profitable. Verify injury reports and starting lineups before tip-off.
""")
