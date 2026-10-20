# NBA Edge Lab

A separate Streamlit project for NBA game, quarter, player box-score, prop, and validation research. It does not modify your NFL/NHL projects.

## Included

- ESPN public scoreboard lookup by date
- Best-effort ESPN injury feed with editable availability and minute overrides
- Optional NBA Stats historical player-game-log retrieval (may be blocked from cloud hosts)
- CSV upload path for real historical player logs when automatic data retrieval fails
- Recent weighted player baselines and player-level Monte Carlo box-score distributions
- Projected game scores, total, home win probability, quarter allocations and score ranges
- Player points/rebounds/assists/steals/blocks/turnovers/threes estimates and P10/P50/P90
- Odds-to-probability and expected-value calculator
- CSV/JSON exports
- Separate `validation_app.py` for chronological walk-forward evaluation
- Railway start command and Streamlit Community Cloud deployment instructions

## Important honesty about accuracy

This is a functioning research prototype, **not a proven high-accuracy or profitable betting model**. It cannot guarantee accurate predictions. Current model baselines use recent historical player game logs, a simple scoring distribution, and approximate quarter weights. It does not yet have a fully trained opponent-adjusted possession model, a verified real-time player-prop odds feed, or calibrated quarter-by-quarter historical distributions. The app flags missing feeds and allows manual review rather than inventing data.

Player box scores require real historical player-game data. NBA Stats requests are attempted only if you enable them and may fail on Railway or other cloud hosts. If they fail, upload a CSV containing one row per player per game. Do not use made-up or future data.

The ESPN public scoreboard and injury feed are best-effort public endpoints and may change, lag, or fail. Confirm injuries and starting lineups against reliable current sources before tipoff. Season defaults should be checked; early-season projections may need prior-season data.

## Required player log CSV

Required columns:

`game_date,player,team,minutes,points,rebounds,assists`

Optional columns:

`steals,blocks,turnovers,threes,fgm,fga,ftm,fta,fouls`

One row per player per game. `game_date` should be a real date/time before the game being projected. Team should be a consistent abbreviation such as `BOS` or `NYK`. Use actual historical results, not projected values.

Templates are included:
- `player_game_logs_template.csv`
- `team_game_results_template.csv`
- `market_history_template.csv`

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run validation separately:

```bash
streamlit run validation_app.py
```

## Upload to GitHub

1. Create a **new** repository, e.g. `nba-edge-lab`. Keep it separate from the NFL repository.
2. Extract this ZIP.
3. Upload the *contents* of the `nba-edge-lab` folder to the repository root.
4. Commit the files.

## Deploy on Railway

1. Create a new Railway project/service connected to this repository.
2. Railway should use `Procfile`. If you set a custom start command, use:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

3. Under Settings → Networking, generate a public domain. The app listens on `$PORT`, so configure Railway's target port according to the service's assigned port.
4. Watch Deploy logs, then open the domain.
5. NBA Stats requests may be blocked from Railway. Upload historical player logs through the app if the automatic fetch fails.

## Deployment to Streamlit Community Cloud

Create a separate app from the new GitHub repository and select `app.py` as the main file. `validation_app.py` is a second app if you want the validation dashboard too.

## Model design and limitations

- Recent player averages are weighted toward newer games and shrunk modestly for small samples.
- Availability labels and minutes can be manually overridden.
- Player point projections are reconciled to the team scoring projection; this is a consistency feature, not evidence that the underlying estimates are accurate.
- Quarter scoring is currently allocated using approximate weights. A later version should fit quarter-specific data from historical play-by-play/box scores.
- Rebounds, assists, steals, blocks and turnovers are noisy estimates based on player historical averages and minutes.
- The EV calculator only calculates from the probability and odds you enter. No sportsbook odds are automatically supplied.
- The validation prototype evaluates rolling-mean player predictions chronologically. It does not claim profit from MAE. A proper betting backtest requires historical odds captured before games, settled results, a frozen selection rule, vig handling and a held-out test period.

## Next development priorities

1. Obtain reliable historical player game logs and quarter splits.
2. Add opponent-adjusted team/player features and player-minute allocation that sums to 240 minutes per team.
3. Add confirmed lineup and injury-source reconciliation.
4. Add a licensed/verified odds feed only if budget allows.
5. Run season-based walk-forward tests and calibration checks before real-money use.
