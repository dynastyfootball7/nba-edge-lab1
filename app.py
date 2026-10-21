from __future__ import annotations

import json
import math
import os
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NBA Edge Lab", page_icon="🏀", layout="wide")
st.title("🏀 NBA Edge Lab — Model v2")
st.caption("Opponent-aware team baselines • player-level projections • Monte Carlo box scores • transparent validation")

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
BDL_STATS = "https://api.balldontlie.io/v1/player_game_stats"
TEAM_CODES = "ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS".split()
PLAYER_REQUIRED = {"game_date", "player", "team", "minutes", "points", "rebounds", "assists"}
TEAM_REQUIRED = {"game_date", "home_team", "away_team", "home_score", "away_score"}
STAT_COLS = ["minutes", "points", "rebounds", "assists", "steals", "blocks", "turnovers", "threes", "fgm", "fga", "ftm", "fta", "fouls"]
COUNT_STATS = ["points", "rebounds", "assists", "steals", "blocks", "turnovers", "threes"]


def get_json(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 15) -> dict:
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_scoreboard(day_string: str) -> list[dict]:
    payload = get_json(ESPN_SCOREBOARD, {"dates": day_string})
    games = []
    for event in payload.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        def team(c: dict) -> dict:
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
    payload = get_json(ESPN_INJURIES)
    rows: list[dict] = []
    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if ("athlete" in obj or "player" in obj) and any(k in obj for k in ("status", "type", "details", "shortComment", "longComment")):
                athlete = obj.get("athlete") or obj.get("player") or {}
                team = obj.get("team") or {}
                typ = obj.get("type") or {}
                rows.append({"player": athlete.get("displayName") or athlete.get("fullName") or obj.get("name", ""),
                             "team": team.get("abbreviation") or team.get("displayName") or "",
                             "status": obj.get("status") or typ.get("description", ""),
                             "detail": obj.get("shortComment") or obj.get("longComment") or obj.get("details", "")})
            for v in obj.values(): walk(v)
        elif isinstance(obj, list):
            for v in obj: walk(v)
    walk(payload)
    unique, result = set(), []
    for r in rows:
        key = tuple(r.values())
        if r["player"] and key not in unique:
            unique.add(key); result.append(r)
    return result


def normalize_player_logs(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    aliases = {"date":"game_date", "player_name":"player", "team_abbreviation":"team", "min":"minutes",
               "pts":"points", "reb":"rebounds", "ast":"assists", "stl":"steals", "blk":"blocks",
               "tov":"turnovers", "fg3m":"threes", "three_pointers":"threes", "pf":"fouls"}
    x = x.rename(columns={k:v for k,v in aliases.items() if k in x.columns and v not in x.columns})
    missing = PLAYER_REQUIRED - set(x.columns)
    if missing: raise ValueError("Player logs missing required columns: " + ", ".join(sorted(missing)))
    x["game_date"] = pd.to_datetime(x["game_date"], errors="coerce")
    for c in STAT_COLS:
        if c not in x: x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["player"] = x["player"].astype(str).str.strip()
    x["team"] = x["team"].astype(str).str.strip().str.upper()
    x = x.dropna(subset=["game_date", "player", "minutes", "points", "rebounds", "assists"])
    x = x[(x.minutes > 0) & (x.minutes <= 60) & x.player.ne("")]
    if x.empty: raise ValueError("No valid player-game rows remain after cleaning.")
    return x.sort_values("game_date").reset_index(drop=True)


def normalize_team_games(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(); x.columns = [str(c).strip().lower() for c in x.columns]
    aliases = {"date":"game_date", "home":"home_team", "away":"away_team", "home_points":"home_score", "away_points":"away_score"}
    x = x.rename(columns={k:v for k,v in aliases.items() if k in x.columns and v not in x.columns})
    missing = TEAM_REQUIRED - set(x.columns)
    if missing: raise ValueError("Team results missing required columns: " + ", ".join(sorted(missing)))
    x["game_date"] = pd.to_datetime(x.game_date, errors="coerce")
    for c in ["home_score", "away_score"]: x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in ["home_team", "away_team"]: x[c] = x[c].astype(str).str.strip().str.upper()
    x = x.dropna(subset=list(TEAM_REQUIRED))
    x = x[(x.home_score > 50) & (x.away_score > 50) & (x.home_score < 180) & (x.away_score < 180)]
    return x.sort_values("game_date").reset_index(drop=True)


def weighted_avg(values: pd.Series | np.ndarray, decay: float = 0.90, default: float = 0.0) -> float:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(v) == 0: return default
    w = np.power(decay, np.arange(len(v)-1, -1, -1))
    return float(np.average(v, weights=w))


def build_player_baselines(logs: pd.DataFrame, selected_date: date, lookback: int) -> pd.DataFrame:
    prior = logs[logs.game_date < pd.Timestamp(selected_date)].copy()
    if prior.empty: return pd.DataFrame()
    rows = []
    for player, g in prior.sort_values("game_date").groupby("player", sort=False):
        g = g.tail(lookback).copy()
        row: dict[str, Any] = {"player": player, "team": str(g.team.iloc[-1]).upper(), "games": len(g),
                               "last_game": g.game_date.max(), "historical_minutes": weighted_avg(g.minutes, .88)}
        # Blend recent form with season-to-date average to reduce small-sample overreaction.
        for stat in STAT_COLS:
            vals = pd.to_numeric(g[stat], errors="coerce")
            recent = weighted_avg(vals, .88, np.nan)
            season_vals = pd.to_numeric(prior.loc[prior.player.eq(player), stat], errors="coerce")
            season = float(season_vals.mean()) if season_vals.notna().any() else recent
            n = int(vals.notna().sum())
            row[stat] = float((n / (n + 5)) * recent + (5 / (n + 5)) * season) if pd.notna(recent) else season
        row["points_per_min"] = row["points"] / max(row["historical_minutes"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


def team_profile(team_games: pd.DataFrame, selected_date: date, team: str, lookback: int = 12) -> dict:
    if team_games is None or team_games.empty:
        return {"games": 0, "points_for": np.nan, "points_against": np.nan, "last_game": pd.NaT, "rest_days": np.nan}
    x = team_games[team_games.game_date < pd.Timestamp(selected_date)].copy()
    team = team.upper()
    home_rows = x[x.home_team.eq(team)].copy()
    away_rows = x[x.away_team.eq(team)].copy()
    home_rows["pf"], home_rows["pa"] = home_rows.home_score, home_rows.away_score
    away_rows["pf"], away_rows["pa"] = away_rows.away_score, away_rows.home_score
    t = pd.concat([home_rows, away_rows], ignore_index=True).sort_values("game_date")
    if t.empty: return {"games": 0, "points_for": np.nan, "points_against": np.nan, "last_game": pd.NaT, "rest_days": np.nan}
    recent = t.tail(lookback)
    last_game = t.game_date.max()
    rest_days = max(0, (pd.Timestamp(selected_date) - last_game.normalize()).days - 1)
    return {"games": len(t), "points_for": weighted_avg(recent.pf, .88), "points_against": weighted_avg(recent.pa, .88),
            "last_game": last_game, "rest_days": rest_days}


def project_team_scores(team_games: pd.DataFrame, selected_date: date, home: str, away: str,
                        player_box: pd.DataFrame, home_adv: float = 2.0) -> tuple[float, float, str, dict]:
    hp, ap = team_profile(team_games, selected_date, home), team_profile(team_games, selected_date, away)
    league = 114.0
    if team_games is not None and not team_games.empty:
        historical = team_games[team_games.game_date < pd.Timestamp(selected_date)]
        if not historical.empty:
            league = float(np.mean([historical.home_score.mean(), historical.away_score.mean()]))
    if hp["games"] >= 3 and ap["games"] >= 3:
        h_off = .55 * hp["points_for"] + .45 * ap["points_against"]
        a_off = .55 * ap["points_for"] + .45 * hp["points_against"]
        # Regress small samples toward league scoring; home-court effect is explicit.
        h_off = hp["games"]/(hp["games"]+8)*h_off + 8/(hp["games"]+8)*league
        a_off = ap["games"]/(ap["games"]+8)*a_off + 8/(ap["games"]+8)*league
        home_score, away_score = h_off + home_adv, a_off
        source = "Historical team scoring + opponent points allowed (rolling, shrunk toward league average)"
    else:
        hpts = float(player_box.loc[player_box.team.eq(home.upper()), "points"].sum()) if not player_box.empty else league
        apts = float(player_box.loc[player_box.team.eq(away.upper()), "points"].sum()) if not player_box.empty else league
        home_score, away_score = hpts + home_adv, apts
        source = "Player-baseline fallback: team game history is missing or too short; confidence is lower"
    # Player box score and team forecast are jointly anchored without inventing missing data.
    return float(np.clip(home_score, 75, 150)), float(np.clip(away_score, 75, 150)), source, {"home": hp, "away": ap, "league_avg": league}


def make_availability(players: pd.DataFrame, injury_rows: list[dict]) -> pd.DataFrame:
    out = players[["player", "team", "minutes", "historical_minutes"]].copy()
    out["availability"] = "Available"; out["minutes_override"] = np.nan; out["status_note"] = ""
    by_name = {str(r.get("player", "")).strip().casefold(): r for r in injury_rows}
    for idx, row in out.iterrows():
        inj = by_name.get(str(row.player).strip().casefold())
        if not inj: continue
        s = str(inj.get("status", "")).casefold()
        if "out" in s: out.at[idx, "availability"] = "Out"
        elif "doubt" in s: out.at[idx, "availability"] = "Doubtful"
        elif "question" in s: out.at[idx, "availability"] = "Questionable"
        elif "prob" in s: out.at[idx, "availability"] = "Probable"
        out.at[idx, "status_note"] = str(inj.get("detail", ""))
    return out


def apply_availability(base: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    x = base.merge(edited[["player", "team", "availability", "minutes_override", "status_note"]], on=["player", "team"], how="left")
    mult = {"available": 1.0, "probable": .97, "questionable": .82, "doubtful": .25, "out": 0.0}
    x["minutes"] = pd.to_numeric(x.minutes, errors="coerce").fillna(0)
    for i, r in x.iterrows():
        status = str(r.get("availability", "Available")).casefold()
        override = r.get("minutes_override", np.nan)
        raw = float(override) if pd.notna(override) else float(r.minutes) * mult.get(status, 1.0)
        x.at[i, "minutes"] = 0.0 if status == "out" else float(np.clip(raw, 0, 48))
    # Normalize team rotations to 240 minutes. Do not silently add minutes to an unlisted player.
    for team, idx in x.groupby("team").groups.items():
        active = x.loc[idx, "minutes"] > 0
        total = x.loc[idx[active], "minutes"].sum()
        if total > 0:
            x.loc[idx[active], "minutes"] *= 240 / total
            x.loc[idx, "minutes"] = x.loc[idx, "minutes"].clip(0, 48)
    return x


def simulate_matchup(players: pd.DataFrame, home: str, away: str, home_mean: float, away_mean: float,
                     n_sims: int, seed: int, home_sd: float = 12.0, away_sd: float = 12.0) -> tuple[pd.DataFrame, dict, dict]:
    rng = np.random.default_rng(seed)
    score_draws = {home.upper(): np.maximum(70, rng.normal(home_mean, home_sd, n_sims)),
                   away.upper(): np.maximum(70, rng.normal(away_mean, away_sd, n_sims))}
    # Prevent impossible ties in the displayed final score by sampling a small overtime increment.
    tied = np.isclose(score_draws[home.upper()], score_draws[away.upper()], atol=.5)
    if tied.any():
        score_draws[home.upper()][tied] += rng.integers(1, 8, tied.sum())
    rows, draws_by_player = [], {}
    for team in [away.upper(), home.upper()]:
        p = players[players.team.str.upper().eq(team)].copy()
        p = p[p.minutes.gt(0)].copy()
        if p.empty: continue
        mins = pd.to_numeric(p.minutes, errors="coerce").fillna(0).clip(0, 48).to_numpy(float)
        hist_mins = pd.to_numeric(p.historical_minutes, errors="coerce").fillna(0).to_numpy(float)
        # Use per-minute rates; Gamma-Poisson style rate variation makes tails wider than a fixed Poisson.
        sampled_mins = np.clip(rng.normal(mins[None, :], np.maximum(1.5, mins[None, :] * .09), (n_sims, len(p))), 0, 48)
        stat_draws: dict[str, np.ndarray] = {}
        for stat in COUNT_STATS:
            stat_values = pd.to_numeric(p[stat], errors="coerce").fillna(0).to_numpy(float)
            rates = np.divide(stat_values, np.maximum(hist_mins, 1), out=np.zeros_like(stat_values), where=hist_mins > 0)
            # Blend individual rate with role/team mean for sparse histories.
            team_rate = float(np.mean(rates[rates > 0])) if np.any(rates > 0) else 0.0
            games = pd.to_numeric(p.games, errors="coerce").fillna(0).to_numpy(float)
            shrink = games / (games + 5)
            rates = shrink * rates + (1 - shrink) * team_rate
            intensity = np.maximum(0, sampled_mins * rates[None, :])
            # Player/game overdispersion: draw a multiplier per player simulation.
            gamma_mult = rng.gamma(shape=8.0, scale=1/8.0, size=intensity.shape)
            stat_draws[stat] = rng.poisson(intensity * gamma_mult)
        # Allocate team point totals to players according to simulated player scoring shares.
        point_weights = stat_draws["points"].astype(float) + .05
        point_weights /= point_weights.sum(axis=1, keepdims=True)
        stat_draws["points"] = np.rint(point_weights * score_draws[team][:, None]).astype(int)
        # Reconcile points so the player's simulated sum matches the simulated team score exactly.
        residual = score_draws[team] - stat_draws["points"].sum(axis=1)
        for sim_idx in range(n_sims):
            j = int(np.argmax(point_weights[sim_idx]))
            stat_draws["points"][sim_idx, j] += int(round(residual[sim_idx]))
        for j, (_, player) in enumerate(p.iterrows()):
            d = {stat: stat_draws[stat][:, j] for stat in COUNT_STATS}
            d["minutes"] = sampled_mins[:, j]
            # Approximate composite markets from their component draws, preserving dependence where possible.
            d["pra"] = d["points"] + d["rebounds"] + d["assists"]
            d["pr"] = d["points"] + d["rebounds"]
            d["pa"] = d["points"] + d["assists"]
            draws_by_player[str(player.player)] = d
            row: dict[str, Any] = {"player": player.player, "team": team, "games": int(player.games), "minutes": float(np.mean(d["minutes"]))}
            for stat, values in d.items():
                row[stat] = float(np.mean(values))
                row[f"{stat}_p10"] = float(np.percentile(values, 10))
                row[f"{stat}_p50"] = float(np.percentile(values, 50))
                row[f"{stat}_p90"] = float(np.percentile(values, 90))
            for c in ["fgm", "fga", "ftm", "fta", "fouls"]:
                v = pd.to_numeric(pd.Series([player.get(c, np.nan)]), errors="coerce").iloc[0]
                row[c] = float(v) if pd.notna(v) else np.nan
            row["fg_pct"] = row["fgm"] / row["fga"] if pd.notna(row["fgm"]) and pd.notna(row["fga"]) and row["fga"] > 0 else np.nan
            row["ft_pct"] = row["ftm"] / row["fta"] if pd.notna(row["ftm"]) and pd.notna(row["fta"]) and row["fta"] > 0 else np.nan
            rows.append(row)
    box = pd.DataFrame(rows)
    home_draw = score_draws[home.upper()]; away_draw = score_draws[away.upper()]
    totals = home_draw + away_draw; margins = home_draw - away_draw
    summary = {"home_score": float(home_draw.mean()), "away_score": float(away_draw.mean()),
               "home_win_pct": float(np.mean(home_draw > away_draw) * 100), "away_win_pct": float(np.mean(away_draw > home_draw) * 100),
               "total": float(totals.mean()), "margin": float(margins.mean()),
               "total_p10": float(np.percentile(totals, 10)), "total_p90": float(np.percentile(totals, 90)),
               "home_p10": float(np.percentile(home_draw, 10)), "home_p90": float(np.percentile(home_draw, 90)),
               "away_p10": float(np.percentile(away_draw, 10)), "away_p90": float(np.percentile(away_draw, 90)),
               "home_draws": home_draw, "away_draws": away_draw}
    return box, summary, draws_by_player


def american_implied(odds: float) -> float:
    x = float(odds)
    if x == 0: return np.nan
    return 100 / (x + 100) if x > 0 else abs(x) / (abs(x) + 100)


def american_decimal(odds: float) -> float:
    x = float(odds)
    if x == 0: return np.nan
    return 1 + x / 100 if x > 0 else 1 + 100 / abs(x)


def rolling_team_backtest(team_games: pd.DataFrame, min_history: int = 8) -> pd.DataFrame:
    """Chronological baseline test: every prediction uses only games before the target date."""
    if team_games is None or team_games.empty: return pd.DataFrame()
    x = team_games.sort_values("game_date").reset_index(drop=True)
    results = []
    for _, g in x.iterrows():
        d = pd.Timestamp(g.game_date).date()
        hp = team_profile(x, d, str(g.home_team), 12)
        ap = team_profile(x, d, str(g.away_team), 12)
        if hp["games"] < min_history or ap["games"] < min_history: continue
        league_rows = x[x.game_date < pd.Timestamp(d)]
        league = float(np.mean([league_rows.home_score.mean(), league_rows.away_score.mean()]))
        pred_home = .55 * hp["points_for"] + .45 * ap["points_against"]
        pred_away = .55 * ap["points_for"] + .45 * hp["points_against"]
        pred_home = hp["games"]/(hp["games"]+8)*pred_home + 8/(hp["games"]+8)*league + 2.0
        pred_away = ap["games"]/(ap["games"]+8)*pred_away + 8/(ap["games"]+8)*league
        results.append({"game_date": g.game_date, "home_team": g.home_team, "away_team": g.away_team,
                        "pred_home": pred_home, "actual_home": g.home_score, "pred_away": pred_away, "actual_away": g.away_score,
                        "pred_total": pred_home + pred_away, "actual_total": g.home_score + g.away_score,
                        "pred_margin": pred_home - pred_away, "actual_margin": g.home_score - g.away_score})
    out = pd.DataFrame(results)
    if not out.empty:
        out["home_abs_error"] = (out.pred_home - out.actual_home).abs()
        out["away_abs_error"] = (out.pred_away - out.actual_away).abs()
        out["total_abs_error"] = (out.pred_total - out.actual_total).abs()
        out["margin_abs_error"] = (out.pred_margin - out.actual_margin).abs()
    return out


def fetch_bdl_logs(season_year: int, api_key: str, max_pages: int = 100) -> pd.DataFrame:
    if not api_key: raise ValueError("Enter a BALLDONTLIE API key. Player-game access depends on your plan.")
    headers = {"Authorization": api_key}
    rows, cursor = [], None
    for _ in range(max_pages):
        params = {"seasons[]": season_year, "per_page": 100}
        if cursor is not None: params["cursor"] = cursor
        payload = get_json(BDL_STATS, params, headers, timeout=25)
        batch = payload.get("data", [])
        for item in batch:
            player, team, game = item.get("player") or {}, item.get("team") or {}, item.get("game") or {}
            rows.append({"game_date": game.get("date"), "player": f"{player.get('first_name','')} {player.get('last_name','')}".strip(),
                         "team": team.get("abbreviation", ""), "minutes": item.get("min"), "points": item.get("pts"),
                         "rebounds": item.get("reb"), "assists": item.get("ast"), "steals": item.get("stl"), "blocks": item.get("blk"),
                         "turnovers": item.get("turnover"), "threes": item.get("fg3m"), "fgm": item.get("fgm"), "fga": item.get("fga"),
                         "ftm": item.get("ftm"), "fta": item.get("fta"), "fouls": item.get("pf")})
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor or not batch: break
    if not rows: raise RuntimeError("No rows returned; check API plan, season year and access permissions.")
    return pd.DataFrame(rows)


with st.sidebar:
    st.header("Model settings")
    selected_date = st.date_input("Slate date", value=date.today())
    lookback = st.slider("Player recent-game window", 5, 30, 15)
    simulations = st.select_slider("Monte Carlo simulations", options=[1000, 2500, 5000, 10000], value=5000)
    home_adv = st.slider("Home-court adjustment (points)", 0.0, 5.0, 2.0, .25)
    seed = st.number_input("Reproducible simulation seed", min_value=1, max_value=999999, value=42)
    st.caption("Do not tune model settings on the final holdout period.")

# Data inputs
tab_model, tab_validation, tab_help = st.tabs(["Game & player model", "Historical validation", "Setup & limitations"])
with tab_model:
    st.subheader("1. Load real historical data")
    c1, c2 = st.columns(2)
    with c1:
        player_file = st.file_uploader("Player-game logs CSV", type=["csv"], key="player_logs")
        st.caption("One row per player per game. Include multiple seasons where possible.")
    with c2:
        team_file = st.file_uploader("Team game results CSV", type=["csv"], key="team_games")
        st.caption("One row per game. Used for opponent-aware score forecasts and walk-forward tests.")
    with st.expander("Optional: fetch player logs from BALLDONTLIE"):
        st.caption("Requires an API key and a plan with player-game statistics. Free availability can vary by plan.")
        secret_key = st.secrets.get("BALLDONTLIE_API_KEY", os.environ.get("BALLDONTLIE_API_KEY", ""))
        api_key = st.text_input("BALLDONTLIE API key", type="password", value=secret_key)
        api_year = st.selectbox("Season start year", [2026, 2025, 2024, 2023], index=0)
        if st.button("Fetch player logs", key="fetch_bdl"):
            try:
                with st.spinner("Fetching player-game records..."):
                    got = normalize_player_logs(fetch_bdl_logs(int(api_year), api_key))
                st.session_state["nba_v2_player_logs"] = got
                st.success(f"Loaded {len(got):,} player-game records from BALLDONTLIE.")
            except Exception as exc: st.error(f"Could not fetch logs: {exc}")
    logs = None; team_games = None
    if player_file is not None:
        try:
            logs = normalize_player_logs(pd.read_csv(player_file)); st.session_state["nba_v2_player_logs"] = logs
            st.success(f"Player logs loaded: {len(logs):,} rows from uploaded CSV.")
        except Exception as exc: st.error(str(exc))
    elif "nba_v2_player_logs" in st.session_state:
        logs = st.session_state["nba_v2_player_logs"]
        st.info(f"Using cached player logs in this session ({len(logs):,} rows).")
    if team_file is not None:
        try:
            team_games = normalize_team_games(pd.read_csv(team_file)); st.session_state["nba_v2_team_games"] = team_games
            st.success(f"Team results loaded: {len(team_games):,} games.")
        except Exception as exc: st.error(str(exc))
    elif "nba_v2_team_games" in st.session_state:
        team_games = st.session_state["nba_v2_team_games"]
        st.info(f"Using cached team results in this session ({len(team_games):,} games).")

    st.subheader("2. Schedule and injury status")
    if st.button("Refresh ESPN schedule and injury feed"):
        fetch_scoreboard.clear(); fetch_injuries.clear()
    games: list[dict] = []; injury_rows: list[dict] = []
    try:
        games = fetch_scoreboard(selected_date.strftime("%Y%m%d"))
        st.success(f"ESPN schedule feed responded: {len(games)} game(s) for {selected_date}.")
    except Exception as exc: st.warning(f"Schedule feed unavailable: {exc}")
    try:
        injury_rows = fetch_injuries()
        st.success(f"ESPN injury feed responded: {len(injury_rows)} injury entries. Name matching remains best-effort.")
    except Exception as exc: st.warning(f"Injury feed unavailable; verify lineups manually: {exc}")

    game_labels = {f'{g["away"]["abbr"]} @ {g["home"]["abbr"]} — {g["status"]}': g for g in games}
    if game_labels:
        chosen_label = st.selectbox("Choose matchup", list(game_labels), key="chosen_matchup")
        chosen_game = game_labels[chosen_label]
        home_team, away_team = chosen_game["home"]["abbr"].upper(), chosen_game["away"]["abbr"].upper()
    else:
        t1, t2 = st.columns(2)
        home_team = t1.selectbox("Home team", TEAM_CODES, index=0)
        away_team = t2.selectbox("Away team", TEAM_CODES, index=1)

    if logs is None:
        st.warning("Player projections require real historical player logs. Upload them or use the API option; no player stats are fabricated.")
    else:
        base_all = build_player_baselines(logs, selected_date, lookback)
        base = base_all[base_all.team.isin([home_team, away_team])].copy()
        if base.empty:
            st.error(f"No prior player logs found for {home_team} or {away_team}. Check team abbreviations and selected date.")
        else:
            st.subheader("3. Availability and minutes review")
            availability = make_availability(base, injury_rows)
            edited = st.data_editor(availability, use_container_width=True, hide_index=True, key="availability_editor_v2",
                column_config={"availability": st.column_config.SelectboxColumn("Availability", options=["Available", "Probable", "Questionable", "Doubtful", "Out"]),
                               "minutes_override": st.column_config.NumberColumn("Minutes override", min_value=0, max_value=48, step=1),
                               "status_note": st.column_config.TextColumn("Injury note")})
            players = apply_availability(base, edited)
            home_base_points = float(players.loc[players.team.eq(home_team), "points"].sum())
            away_base_points = float(players.loc[players.team.eq(away_team), "points"].sum())
            # Player-baseline sums are only a fallback; team historical results are the preferred scoring signal.
            home_score, away_score, score_source, profiles = project_team_scores(team_games, selected_date, home_team, away_team, players, home_adv)
            st.info(f"Score forecast source: {score_source}")
            if team_games is None or team_games.empty:
                st.warning("Upload historical team results for opponent-aware forecasts and validation. Current team-score forecast is lower confidence.")
            else:
                hp, ap = profiles["home"], profiles["away"]
                st.caption(f"Team-history sample sizes — {home_team}: {hp['games']} games; {away_team}: {ap['games']} games. Rest days: home {hp['rest_days']}, away {ap['rest_days']}.")
            if st.button("RUN FULL MODEL", type="primary", use_container_width=True):
                with st.spinner(f"Running {simulations:,} simulations..."):
                    box, summary, player_draws = simulate_matchup(players, home_team, away_team, home_score, away_score,
                                                                    int(simulations), int(seed))
                    st.session_state["nba_v2_result"] = {"box": box, "summary": summary, "draws": player_draws,
                                                          "matchup": (home_team, away_team), "date": str(selected_date),
                                                          "score_source": score_source}
            result = st.session_state.get("nba_v2_result")
            if result and result.get("matchup") == (home_team, away_team) and result.get("date") == str(selected_date):
                box = result["box"]; summary = result["summary"]; player_draws = result["draws"]
                st.divider(); st.subheader("4. Game projection")
                a, b, c, d = st.columns(4)
                a.metric(home_team, f"{summary['home_score']:.1f}", f"80% range {summary['home_p10']:.0f}–{summary['home_p90']:.0f}")
                b.metric(away_team, f"{summary['away_score']:.1f}", f"80% range {summary['away_p10']:.0f}–{summary['away_p90']:.0f}")
                c.metric("Projected total", f"{summary['total']:.1f}", f"80% range {summary['total_p10']:.0f}–{summary['total_p90']:.0f}")
                d.metric(f"{home_team} win probability", f"{summary['home_win_pct']:.1f}%")
                st.metric("Projected home margin", f"{summary['margin']:+.1f} points", f"Away win probability {summary['away_win_pct']:.1f}%")
                st.caption("The Monte Carlo range represents uncertainty under the model assumptions; it is not a guarantee or proof of betting value.")
                st.subheader("5. Full projected player box scores")
                display_cols = [c for c in ["player", "team", "minutes", "points", "rebounds", "assists", "threes", "steals", "blocks", "turnovers", "pra", "pr", "pa", "fgm", "fga", "fg_pct", "ftm", "fta", "ft_pct", "points_p10", "points_p50", "points_p90", "rebounds_p10", "rebounds_p90", "assists_p10", "assists_p90"] if c in box.columns]
                st.dataframe(box[display_cols].round(2).sort_values(["team", "minutes"], ascending=[True, False]), use_container_width=True, hide_index=True)
                st.download_button("Download full projected box scores CSV", box.to_csv(index=False).encode(), "nba_edge_lab_projected_box_scores.csv", "text/csv")
                st.download_button("Download game projection JSON", json.dumps({"date": str(selected_date), "home_team": home_team, "away_team": away_team,
                    "home_score": summary["home_score"], "away_score": summary["away_score"], "total": summary["total"],
                    "home_win_probability_pct": summary["home_win_pct"], "away_win_probability_pct": summary["away_win_pct"],
                    "score_source": score_source}, indent=2), "nba_edge_lab_game_projection.json", "application/json")
                st.subheader("6. Prop probability explorer")
                names = sorted(box.player.astype(str).unique())
                selected_player = st.selectbox("Player", names, key="prop_player_v2")
                stat_labels = {"Points":"points", "Rebounds":"rebounds", "Assists":"assists", "3-pointers made":"threes", "Steals":"steals", "Blocks":"blocks", "Turnovers":"turnovers", "PRA":"pra", "Points + rebounds":"pr", "Points + assists":"pa"}
                label = st.selectbox("Market", list(stat_labels), key="prop_market_v2")
                line = st.number_input("Prop line", min_value=0.0, max_value=100.0, value=20.5, step=0.5, key="prop_line_v2")
                chosen_draws = player_draws[selected_player][stat_labels[label]]
                p_over = float(np.mean(chosen_draws > line)); p_under = float(np.mean(chosen_draws < line)); p_push = max(0.0, 1 - p_over - p_under)
                p1, p2, p3 = st.columns(3)
                p1.metric("Over probability", f"{p_over*100:.1f}%"); p2.metric("Under probability", f"{p_under*100:.1f}%"); p3.metric("Push probability", f"{p_push*100:.1f}%")
                odds_col1, odds_col2 = st.columns(2)
                over_odds = odds_col1.number_input("Available over odds (American)", min_value=-10000, max_value=10000, value=-110, step=5, key="over_odds_v2")
                under_odds = odds_col2.number_input("Available under odds (American)", min_value=-10000, max_value=10000, value=-110, step=5, key="under_odds_v2")
                if over_odds != 0 and under_odds != 0:
                    imp_over, imp_under = american_implied(over_odds), american_implied(under_odds)
                    no_vig_over = imp_over / (imp_over + imp_under); no_vig_under = imp_under / (imp_over + imp_under)
                    ev_over = p_over * (american_decimal(over_odds)-1) - p_under
                    ev_under = p_under * (american_decimal(under_odds)-1) - p_over
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Over EV per unit", f"{ev_over:+.3f}", f"Model minus no-vig {p_over-no_vig_over:+.1%}")
                    e2.metric("Under EV per unit", f"{ev_under:+.3f}", f"Model minus no-vig {p_under-no_vig_under:+.1%}")
                    e3.metric("Book overround", f"{(imp_over+imp_under-1)*100:.1f}%")
                st.caption("Enter the actual price available to you. A positive estimated EV is only a model estimate until the strategy is validated on held-out historical lines.")

with tab_validation:
    st.subheader("Chronological team-score validation")
    st.write("Upload historical team results in the Model tab. Every prediction uses only games before the target game date; no future games are included in the rolling features.")
    team_games = st.session_state.get("nba_v2_team_games")
    min_history = st.slider("Minimum prior games per team", 3, 20, 8, key="min_history_v2")
    if team_games is None or team_games.empty:
        st.info("Load a real team-results CSV in the Model tab to run this test.")
    else:
        bt = rolling_team_backtest(team_games, min_history)
        if bt.empty: st.warning("Not enough prior history to evaluate games. Load multiple seasons or reduce the minimum history threshold.")
        else:
            cutoff = bt.game_date.quantile(.8)
            dev = bt[bt.game_date <= cutoff]; holdout = bt[bt.game_date > cutoff]
            cols = st.columns(4)
            cols[0].metric("Tested games", f"{len(bt):,}")
            cols[1].metric("Total MAE", f"{bt.total_abs_error.mean():.2f} points")
            cols[2].metric("Margin MAE", f"{bt.margin_abs_error.mean():.2f} points")
            cols[3].metric("Chronological holdout", f"{len(holdout):,} games")
            st.dataframe(pd.DataFrame([{"period":"Development", "games":len(dev), "total_MAE":dev.total_abs_error.mean(), "margin_MAE":dev.margin_abs_error.mean()},
                                       {"period":"Later holdout", "games":len(holdout), "total_MAE":holdout.total_abs_error.mean(), "margin_MAE":holdout.margin_abs_error.mean()}]).round(3), use_container_width=True, hide_index=True)
            st.dataframe(bt.tail(100), use_container_width=True, hide_index=True)
            st.download_button("Download chronological backtest CSV", bt.to_csv(index=False).encode(), "nba_edge_lab_team_backtest.csv", "text/csv")
            st.warning("This validates score forecasts, not betting profitability. A betting ROI test also needs timestamped historical odds, correct settlements, vig and a frozen selection rule.")

with tab_help:
    st.subheader("Data format templates")
    st.markdown("**Player logs CSV required columns:** `game_date,player,team,minutes,points,rebounds,assists`. Optional: `steals,blocks,turnovers,threes,fgm,fga,ftm,fta,fouls`. One row per player per game; include opponent column if you have it.")
    st.download_button("Download player logs template", pd.DataFrame(columns=["game_date", "player", "team", *STAT_COLS]).to_csv(index=False).encode(), "player_logs_template.csv", "text/csv")
    st.markdown("**Team results CSV required columns:** `game_date,home_team,away_team,home_score,away_score`. One row per completed game. Use team abbreviations consistently.")
    st.download_button("Download team results template", pd.DataFrame(columns=sorted(TEAM_REQUIRED)).to_csv(index=False).encode(), "team_results_template.csv", "text/csv")
    st.markdown("**Data source limitations**")
    st.markdown("""
- ESPN's public schedule and injury endpoints are best-effort; they can change or fail. Injury name matching is not guaranteed.
- BALLDONTLIE player-game access requires a key and an eligible plan. The model does not silently assume the endpoint is free.
- Player projections need real logs and expected active players. The app uses recent player rates and user-reviewed minutes; it cannot guarantee automatic confirmed starting lineups.
- The team score baseline is opponent-aware only when real historical team game results are uploaded. Without them, the app falls back to player-baseline totals and flags the lower confidence.
- The player simulation conditions point allocation on the team score forecast. This keeps projected player scoring consistent with the game forecast but is still an approximation; full player correlations, play-by-play possessions, on/off effects and matchup-specific defensive schemes remain future model upgrades.
- No odds are fetched or fabricated. Prop EV is calculated only from the actual odds entered by the user.
- Model outputs are not guaranteed profitable. Do not use a favorable small sample as proof; retain a final untouched holdout and evaluate CLV/ROI using real historical sportsbook snapshots.
""")

st.caption("NBA Edge Lab v2 • Separate from NFL/NHL projects • Research tool, not financial advice")
