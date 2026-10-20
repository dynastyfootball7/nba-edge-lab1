from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA Edge Lab — Validation", page_icon="🧪", layout="wide")
st.title("🧪 NBA Edge Lab — Walk-forward validation")
st.caption("Chronological holdout testing for player projections and sportsbook prices.")

st.warning("A backtest is only meaningful when each prediction uses information available before tip-off. Avoid leakage from same-game or future rows.")

uploaded = st.file_uploader("Upload historical player game logs CSV", type=["csv"])
market_uploaded = st.file_uploader("Upload timestamped historical market snapshots CSV (optional)", type=["csv"])
min_history = st.slider("Minimum prior player games", 3, 20, 8)
lookback = st.slider("Rolling lookback", 5, 30, 15)
if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"date":"game_date","player_name":"player","pts":"points","reb":"rebounds","ast":"assists","min":"minutes"}
    df = df.rename(columns={k:v for k,v in aliases.items() if k in df.columns and v not in df.columns})
    required = {"game_date","player","team","minutes","points","rebounds","assists"}
    if not required.issubset(df.columns):
        st.error("Missing required columns: " + ", ".join(sorted(required-set(df.columns))))
    else:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        for c in ["minutes","points","rebounds","assists","steals","blocks","turnovers","threes"]:
            if c not in df: df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["game_date","player","minutes","points","rebounds","assists"]).sort_values("game_date")
        stats = ["minutes","points","rebounds","assists","steals","blocks","turnovers","threes"]
        preds = []
        # Strictly chronological: each row prediction uses only earlier games for the same player.
        for player, g in df.groupby("player", sort=False):
            g = g.sort_values("game_date").reset_index(drop=True)
            for i in range(min_history, len(g)):
                prior = g.iloc[max(0,i-lookback):i]
                actual = g.iloc[i]
                row = {"game_date":actual["game_date"],"player":player,"team":actual["team"],"games_prior":len(prior)}
                for stat in stats:
                    row[f"pred_{stat}"] = prior[stat].mean() if prior[stat].notna().any() else np.nan
                    row[f"actual_{stat}"] = actual[stat]
                preds.append(row)
        pred_df = pd.DataFrame(preds)
        if pred_df.empty:
            st.warning("Not enough player history to validate. Add more games or lower the minimum history.")
        else:
            st.subheader("Out-of-sample projection errors")
            metric_rows = []
            for stat in stats:
                valid = pred_df[[f"pred_{stat}",f"actual_{stat}"]].dropna()
                if len(valid):
                    err = valid[f"pred_{stat}"]-valid[f"actual_{stat}"]
                    metric_rows.append({"Stat":stat,"Rows":len(valid),"MAE":abs(err).mean(),"RMSE":np.sqrt((err**2).mean()),"Bias (pred-actual)":err.mean()})
            metrics = pd.DataFrame(metric_rows)
            st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
            st.line_chart(metrics.set_index("Stat")[["MAE"]])
            st.download_button("Download walk-forward predictions", pred_df.to_csv(index=False).encode(), "nba_walk_forward_predictions.csv","text/csv")
            st.subheader("Time-based split")
            split_date = pred_df["game_date"].quantile(0.8)
            dev = pred_df[pred_df["game_date"] < split_date]
            test = pred_df[pred_df["game_date"] >= split_date]
            st.write(f"Development rows: {len(dev):,} • untouched final-test rows: {len(test):,} • split date: {split_date.date()}")
            st.caption("Do not tune parameters using the final-test rows. This baseline tests rolling means, not the complete simulation's profitability.")
if market_uploaded:
    m = pd.read_csv(market_uploaded)
    m.columns = [str(c).strip().lower() for c in m.columns]
    required = {"game_date","market_type","entity","market","line","over_odds","under_odds","actual_value"}
    if not required.issubset(m.columns):
        st.error("Market CSV missing: " + ", ".join(sorted(required-set(m.columns))))
    else:
        st.subheader("Market data quality")
        st.write(f"Rows: {len(m):,}; date range: {m['game_date'].min()} to {m['game_date'].max()}")
        st.info("ROI cannot be calculated honestly without a frozen, timestamped selection rule and model probabilities matched to these exact lines. This app does not fabricate odds or bet outcomes.")
else:
    st.caption("Optional market schema: game_date, market_type, entity, market, line, over_odds, under_odds, actual_value, captured_at.")
