"""
StressFreeFantasy - Start/Sit Optimizer for NFL Fantasy Football
Powered by TabPFNRegressor trained on real NFL history (nflverse via nflreadpy).
"""
import json
import os
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Allow >1000 training rows on CPU for modern TabPFN releases
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

# ----------------------------------------------------------------------------
# Global Configuration
# ----------------------------------------------------------------------------
POSITIONS = ["QB", "RB", "WR", "TE"]
N_TRAIN_DEFAULT = 2000      # Historical player-weeks provided to TabPFN context
HISTORY_SEASONS = 5         # Past seasons to pull
MIN_PRIOR_AVG = 3.0         # Filter out non-fantasy relevant scrubs (< 3 pts avg)

FEATURES = [
    "pos_QB", "pos_RB", "pos_WR", "pos_TE",
    "Avg_Pts", "Last_Week_Pts", "Form_L3",
    "Opp_Def_Rank", "Is_Home", "Is_Dome", "Vegas_Total", "Impl_Team_Total",
    "Volume_Proj"
]

ABBR_FIX = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"}


def norm_abbr(a):
    a = str(a or "").upper().strip()
    return ABBR_FIX.get(a, a)


def _num(x, default=0.0):
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


# ============================================================================
# SCORING ENGINE: Custom Rules, Presets & ESPN Import
# ============================================================================
def _col(df, name):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


RULES = {
    "passing_yards":         ("Passing Yards (per yard)",             lambda d: _col(d, "passing_yards")),
    "passing_tds":           ("Passing TDs",                          lambda d: _col(d, "passing_tds")),
    "passing_interceptions": ("Interceptions Thrown",                 lambda d: _col(d, "passing_interceptions")),
    "passing_2pt":           ("2-Pt Conversions (Pass)",              lambda d: _col(d, "passing_2pt_conversions")),
    "completions":           ("Pass Completions",                     lambda d: _col(d, "completions")),
    "attempts":              ("Pass Attempts",                        lambda d: _col(d, "attempts")),
    "sacks_suffered":        ("Sacks Suffered (QB)",                  lambda d: _col(d, "sacks_suffered")),
    "passing_first_downs":   ("Passing 1st Downs",                    lambda d: _col(d, "passing_first_downs")),
    "carries":               ("Rushing Attempts (Carries)",           lambda d: _col(d, "carries")),
    "rushing_yards":         ("Rushing Yards (per yard)",             lambda d: _col(d, "rushing_yards")),
    "rushing_tds":           ("Rushing TDs",                          lambda d: _col(d, "rushing_tds")),
    "rushing_2pt":           ("2-Pt Conversions (Rush)",              lambda d: _col(d, "rushing_2pt_conversions")),
    "rushing_first_downs":   ("Rushing 1st Downs",                    lambda d: _col(d, "rushing_first_downs")),
    "targets":               ("Targets",                              lambda d: _col(d, "targets")),
    "receptions":            ("Receptions (PPR Point)",               lambda d: _col(d, "receptions")),
    "receiving_yards":       ("Receiving Yards (per yard)",           lambda d: _col(d, "receiving_yards")),
    "receiving_tds":         ("Receiving TDs",                        lambda d: _col(d, "receiving_tds")),
    "receiving_2pt":         ("2-Pt Conversions (Receiving)",         lambda d: _col(d, "receiving_2pt_conversions")),
    "receiving_first_downs": ("Receiving 1st Downs",                  lambda d: _col(d, "receiving_first_downs")),
    "fumbles_lost":          ("Fumbles Lost",                         lambda d: _col(d, "sack_fumbles_lost") + _col(d, "rushing_fumbles_lost") + _col(d, "receiving_fumbles_lost")),
    "fumbles_all":           ("Total Fumbles",                        lambda d: _col(d, "sack_fumbles") + _col(d, "rushing_fumbles") + _col(d, "receiving_fumbles")),
    "special_teams_tds":     ("Special Teams Return TDs",             lambda d: _col(d, "special_teams_tds")),
    "pass_300_399":          ("Bonus: 300–399 Passing Yds",           lambda d: ((_col(d, "passing_yards") >= 300) & (_col(d, "passing_yards") < 400)).astype(float)),
    "pass_400_plus":         ("Bonus: 400+ Passing Yds",              lambda d: (_col(d, "passing_yards") >= 400).astype(float)),
    "rush_100_199":          ("Bonus: 100–199 Rushing Yds",           lambda d: ((_col(d, "rushing_yards") >= 100) & (_col(d, "rushing_yards") < 200)).astype(float)),
    "rush_200_plus":         ("Bonus: 200+ Rushing Yds",              lambda d: (_col(d, "rushing_yards") >= 200).astype(float)),
    "rec_100_199":           ("Bonus: 100–199 Receiving Yds",         lambda d: ((_col(d, "receiving_yards") >= 100) & (_col(d, "receiving_yards") < 200)).astype(float)),
    "rec_200_plus":          ("Bonus: 200+ Receiving Yds",            lambda d: (_col(d, "receiving_yards") >= 200).astype(float)),
}

BASE_RULES = {
    "passing_yards": 0.04, "passing_tds": 4, "passing_interceptions": -2, "passing_2pt": 2,
    "rushing_yards": 0.1, "rushing_tds": 6, "rushing_2pt": 2,
    "receiving_yards": 0.1, "receiving_tds": 6, "receiving_2pt": 2,
    "fumbles_lost": -2, "special_teams_tds": 6,
}
PRESET_NAMES = ["PPR", "Half-PPR", "Standard"]


def preset_rules(name: str) -> dict:
    r = dict(BASE_RULES)
    r["receptions"] = {"Standard": 0.0, "Half-PPR": 0.5, "PPR": 1.0}[name]
    return r


def make_key(rules: dict) -> tuple:
    return tuple(sorted((k, round(_num(v), 4)) for k, v in rules.items() if k in RULES and abs(_num(v)) > 1e-9))


def compute_points(df: pd.DataFrame, rules_key: tuple) -> pd.Series:
    pts = pd.Series(0.0, index=df.index)
    for k, w in rules_key:
        pts = pts + RULES[k][1](df) * w
    return pts


ESPN_STAT_MAP = {
    3: "passing_yards", 4: "passing_tds", 20: "passing_interceptions", 19: "passing_2pt",
    0: "attempts", 1: "completions", 64: "sacks_suffered",
    23: "carries", 24: "rushing_yards", 25: "rushing_tds", 26: "rushing_2pt",
    41: "receptions", 53: "receptions", 42: "receiving_yards", 43: "receiving_tds", 44: "receiving_2pt",
    58: "targets", 72: "fumbles_lost", 68: "fumbles_all", 101: "special_teams_tds", 102: "special_teams_tds",
    17: "pass_300_399", 18: "pass_400_plus", 37: "rush_100_199", 38: "rush_200_plus",
    56: "rec_100_199", 57: "rec_200_plus",
}
ESPN_UNSUPPORTED_OFFENSE = {15, 16, 21, 22, 35, 36, 39, 40, 45, 46, 59, 60, 61, 63}


def rules_from_espn(scoring_format: list):
    rules, unsupported = {}, []
    for item in scoring_format or []:
        sid, pts = item.get("id"), _num(item.get("points"), 0.0)
        if abs(pts) < 1e-9:
            continue
        if sid == 2:
            rules["attempts"] = rules.get("attempts", 0.0) + pts
            rules["completions"] = rules.get("completions", 0.0) - pts
        elif sid == 73:
            rules["passing_interceptions"] = rules.get("passing_interceptions", 0.0) + pts
            rules["fumbles_lost"] = rules.get("fumbles_lost", 0.0) + pts
        elif sid == 62:
            for k in ("passing_2pt", "rushing_2pt", "receiving_2pt"):
                rules[k] = rules.get(k, 0.0) + pts
        elif sid in ESPN_STAT_MAP:
            rules[ESPN_STAT_MAP[sid]] = pts
        elif sid in ESPN_UNSUPPORTED_OFFENSE:
            unsupported.append(str(item.get("label") or item.get("abbr") or sid))
    return rules, unsupported


def parse_rules_file(uploaded) -> dict:
    if uploaded.name.lower().endswith(".json"):
        data = json.load(uploaded)
    else:
        df = pd.read_csv(uploaded)
        df.columns = [c.strip().lower() for c in df.columns]
        kc = next(c for c in df.columns if c in ("stat", "key", "rule"))
        vc = next(c for c in df.columns if c in ("points", "value"))
        data = dict(zip(df[kc].astype(str), df[vc]))
    rules = {k: _num(v) for k, v in data.items() if k in RULES}
    if not rules:
        raise ValueError("No recognized scoring rule keys found.")
    return rules


# ============================================================================
# DATA PIPELINE: Load and Build Training Set from nflverse
# ============================================================================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_nflverse(seasons: tuple):
    import nflreadpy as nfl
    frames, loaded = [], []
    for s in seasons:
        try:
            df = nfl.load_player_stats([s]).to_pandas()
            if len(df):
                frames.append(df)
                loaded.append(s)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("Failed to fetch player stats from nflverse.")
    return pd.concat(frames, ignore_index=True), loaded


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_schedules(seasons: tuple):
    import nflreadpy as nfl
    frames = []
    for s in seasons:
        try:
            frames.append(nfl.load_schedules([s]).to_pandas())
        except Exception:
            continue
    if not frames:
        raise RuntimeError("Failed to fetch schedules from nflverse.")
    return pd.concat(frames, ignore_index=True)


def prep_points(ps: pd.DataFrame, rules_key: tuple) -> pd.DataFrame:
    df = ps[(ps["season_type"] == "REG") & (ps["position"].isin(POSITIONS))].copy()
    df["team"] = df["team"].map(norm_abbr)
    df["opponent_team"] = df["opponent_team"].map(norm_abbr)
    df["pts"] = compute_points(df, rules_key)
    
    carries = _col(df, "carries")
    targets = _col(df, "targets")
    attempts = _col(df, "attempts")
    df["actual_volume"] = np.where(df["position"] == "QB", attempts, carries + targets)
    
    return df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def defense_allowed(df: pd.DataFrame) -> pd.DataFrame:
    d = (
        df.groupby(["season", "week", "opponent_team", "position"], as_index=False)["pts"].sum()
        .rename(columns={"opponent_team": "defense", "pts": "allowed"})
        .sort_values(["defense", "position", "season", "week"])
        .reset_index(drop=True)
    )
    g = d.groupby(["defense", "position"])["allowed"]
    d["allowed_prior"] = g.transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    d["allowed_now"] = g.transform(lambda s: s.rolling(8, min_periods=3).mean())
    d["opp_rank"] = d.groupby(["season", "week", "position"])["allowed_prior"].rank(method="min", ascending=True)
    return d


def build_training_frame(ps: pd.DataFrame, sc: pd.DataFrame, rules_key: tuple) -> pd.DataFrame:
    df = prep_points(ps, rules_key)

    g = df.groupby("player_id")["pts"]
    df["Avg_Pts"] = g.transform(lambda s: s.shift(1).rolling(16, min_periods=3).mean())
    df["Last_Week_Pts"] = g.shift(1)
    df["Form_L3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())

    g_vol = df.groupby("player_id")["actual_volume"]
    df["Volume_Proj"] = g_vol.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean()).fillna(12.0)

    d = defense_allowed(df)
    df = df.merge(
        d[["season", "week", "defense", "position", "opp_rank"]],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "defense", "position"],
        how="left",
    ).drop(columns=["defense"])

    sc = sc[sc["game_type"] == "REG"][["game_id", "home_team", "away_team", "spread_line", "total_line", "roof"]].copy()
    sc["home_team"] = sc["home_team"].map(norm_abbr)
    sc["away_team"] = sc["away_team"].map(norm_abbr)
    df = df.merge(sc, on="game_id", how="left")

    df["Is_Home"] = (df["team"] == df["home_team"]).astype(int)
    team_spread = np.where(df["Is_Home"] == 1, df["spread_line"], -df["spread_line"])
    df["Vegas_Total"] = df["total_line"]
    df["Impl_Team_Total"] = (df["total_line"] + team_spread) / 2.0
    df["Is_Dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    df["Opp_Def_Rank"] = df["opp_rank"]
    df["Position"] = df["position"]
    df["Y"] = df["pts"]

    keep = ["season", "week", "player_display_name", "Position", "Avg_Pts", "Last_Week_Pts", "Form_L3",
            "Opp_Def_Rank", "Is_Home", "Is_Dome", "Vegas_Total", "Impl_Team_Total", "Volume_Proj", "Y"]
    out = df[keep].dropna(subset=[c for c in keep if c != "player_display_name"])
    out = out[out["Avg_Pts"] >= MIN_PRIOR_AVG]
    return out.reset_index(drop=True)


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame(index=df.index)
    for p in POSITIONS:
        X[f"pos_{p}"] = (df["Position"] == p).astype(float)
    for c in ["Avg_Pts", "Last_Week_Pts", "Form_L3", "Opp_Def_Rank", "Vegas_Total", "Impl_Team_Total", "Volume_Proj"]:
        X[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").astype(float)
    X["Is_Home"] = df["Is_Home"].astype(int).astype(float)
    X["Is_Dome"] = df["Is_Dome"].astype(int).astype(float)
    return X[FEATURES].fillna(X[FEATURES].median(numeric_only=True)).values


def pick_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@st.cache_resource(show_spinner=False)
def get_model(rules_key: tuple, season_year: int, n_train: int):
    from tabpfn import TabPFNRegressor

    seasons = tuple(range(season_year - HISTORY_SEASONS, season_year + 1))
    ps, loaded = load_nflverse(seasons)
    sc = load_schedules(tuple(loaded))
    frame = build_training_frame(ps, sc, rules_key)
    if len(frame) > n_train:
        frame = frame.sample(n_train, random_state=42)

    model = TabPFNRegressor(device=pick_device())
    model.fit(to_matrix(frame), frame["Y"].values)
    return model, len(frame), loaded


def run_validation(rules_key: tuple, season_year: int, n_train: int, n_test: int = 600):
    from tabpfn import TabPFNRegressor

    seasons = tuple(range(season_year - HISTORY_SEASONS, season_year + 1))
    ps, loaded = load_nflverse(seasons)
    sc = load_schedules(tuple(loaded))
    frame = build_training_frame(ps, sc, rules_key)

    test_season = max(s for s in loaded if s < season_year)
    train = frame[frame["season"] < test_season]
    test = frame[frame["season"] == test_season]
    train = train.sample(min(n_train, len(train)), random_state=1)
    test = test.sample(min(n_test, len(test)), random_state=1)

    model = TabPFNRegressor(device=pick_device())
    model.fit(to_matrix(train), train["Y"].values)
    pred = model.predict(to_matrix(test))
    y = test["Y"].values

    test_reset = test.copy().reset_index(drop=True)
    test_reset["pred"] = pred
    pairs_total, pairs_correct = 0, 0
    for pos in POSITIONS:
        pos_df = test_reset[test_reset["Position"] == pos]
        if len(pos_df) >= 2:
            n_pairs = min(300, len(pos_df) * (len(pos_df) - 1) // 2)
            for _ in range(n_pairs):
                p1, p2 = pos_df.sample(2).iloc
                actual_diff = p1["Y"] - p2["Y"]
                pred_diff = p1["pred"] - p2["pred"]
                if abs(actual_diff) > 1.0:
                    pairs_total += 1
                    if (actual_diff > 0 and pred_diff > 0) or (actual_diff < 0 and pred_diff < 0):
                        pairs_correct += 1

    pairwise_acc = (pairs_correct / pairs_total * 100) if pairs_total > 0 else 50.0

    return {
        "test_season": int(test_season),
        "n_train": len(train),
        "n_test": len(test),
        "mae_model": float(np.mean(np.abs(y - pred))),
        "mae_avg": float(np.mean(np.abs(y - test["Avg_Pts"].values))),
        "mae_form": float(np.mean(np.abs(y - test["Form_L3"].values))),
        "corr_model": float(np.corrcoef(pred, y)[0, 1]),
        "pairwise_acc": pairwise_acc
    }


# ============================================================================
# LIVE MATCHUP & ODDS INGESTION
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def espn_scoreboard_odds(year: int, week: int) -> dict:
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    r = requests.get(url, params={"seasontype": 2, "week": week, "dates": year}, timeout=10)
    r.raise_for_status()
    out = {}
    for ev in r.json().get("events", []):
        try:
            comp = ev["competitions"][0]
            t = {c["homeAway"]: norm_abbr(c["team"]["abbreviation"]) for c in comp["competitors"]}
            odds = (comp.get("odds") or [{}])[0]
            total = odds.get("overUnder")
            details = str(odds.get("details") or "")
            home_spread = 0.0
            m = re.match(r"([A-Za-z]{2,3})\s+([+-]?\d+(?:\.\d+)?)", details)
            if m:
                margin = abs(float(m.group(2)))
                home_spread = margin if norm_abbr(m.group(1)) == t["home"] else -margin
            out[(t["home"], t["away"])] = (_num(total, None) if total is not None else None, home_spread)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_matchups(season: int, week: int) -> dict:
    out = {}
    try:
        sc = load_schedules((season,))
        rows = sc[(sc["week"] == week) & (sc["game_type"] == "REG")]
    except Exception:
        rows = pd.DataFrame()

    espn_odds = {}
    need_espn = rows.empty or rows["total_line"].isna().any()
    if need_espn:
        try:
            espn_odds = espn_scoreboard_odds(season, week)
        except Exception:
            espn_odds = {}

    def put(home, away, total, home_spread, dome):
        out[home] = {"opp": away, "home": True, "total": total, "spread": home_spread, "dome": dome}
        out[away] = {"opp": home, "home": False, "total": total, "spread": -home_spread, "dome": dome}

    for r in rows.itertuples():
        home, away = norm_abbr(r.home_team), norm_abbr(r.away_team)
        total = None if pd.isna(r.total_line) else float(r.total_line)
        spread = 0.0 if pd.isna(r.spread_line) else float(r.spread_line)
        if total is None and (home, away) in espn_odds:
            total, spread = espn_odds[(home, away)]
        dome = str(r.roof) in ("dome", "closed")
        put(home, away, total, spread, dome)

    if not out:
        for (home, away), (total, hs) in espn_odds.items():
            put(home, away, total, hs, False)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def live_opp_ranks(rules_key: tuple, season_year: int) -> dict:
    seasons = tuple(range(season_year - 1, season_year + 1))
    ps, _ = load_nflverse(seasons)
    df = prep_points(ps, rules_key)
    d = defense_allowed(df).dropna(subset=["allowed_now"])
    latest = d.sort_values(["season", "week"]).groupby(["defense", "position"]).tail(1).copy()
    latest["rank"] = latest.groupby("position")["allowed_now"].rank(method="min", ascending=True)
    return {(r.defense, r.position): int(r.rank) for r in latest.itertuples()}


def _recent_points(stats: dict, week: int, n: int = 3):
    vals = []
    for w in range(week - 1, 0, -1):
        s = stats.get(w) or {}
        pts = s.get("points", s.get("applied_total"))
        if pts is not None:
            vals.append(_num(pts, 0.0))
        if len(vals) >= n:
            break
    return vals


def build_roster_from_espn(team, week: int, season: int, rules_key: tuple):
    notes = []
    week = int(min(max(week, 1), 18))

    try:
        matchups = fetch_matchups(season, week)
    except Exception as e:
        matchups = {}
        notes.append(f"Failed to fetch matchup/odds ({e}). Using baseline averages.")
    try:
        ranks = live_opp_ranks(rules_key, season)
    except Exception as e:
        ranks = {}
        notes.append(f"Could not compute opponent rankings ({e}). Ranked 16 by default.")

    rows, missing_odds, skipped = [], 0, 0
    for p in team.roster:
        pos = p.position
        if pos not in POSITIONS:
            skipped += 1
            continue

        pro_team = norm_abbr(getattr(p, "proTeam", None) or getattr(p, "pro_team", None))
        m = matchups.get(pro_team)
        has_game = m is not None

        opp = m["opp"] if m else None
        is_home = m["home"] if m else True
        is_dome = m["dome"] if m else False
        total = m["total"] if m else None
        spread = m["spread"] if m else 0.0
        if has_game and total is None:
            missing_odds += 1
        if total is None:
            total = 44.0
        impl = (total + (spread or 0.0)) / 2.0

        stats = getattr(p, "stats", None) or {}
        wk = stats.get(week) or {}
        recent = _recent_points(stats, week, 3)

        avg_pts = _num(getattr(p, "avg_points", None), 0.0)
        proj = _num(wk.get("projected_points"), 0.0)
        if proj <= 0:
            proj = _num(getattr(p, "projected_avg_points", None), 0.0)
        if proj <= 0:
            proj = avg_pts
        if avg_pts <= 0:
            avg_pts = float(np.mean(recent)) if recent else proj
        last_pts = recent[0] if recent else avg_pts
        form3 = float(np.mean(recent)) if recent else avg_pts

        opp_rank = ranks.get((opp, pos), 16) if opp else 16

        status_raw = str(getattr(p, "injuryStatus", "ACTIVE") or "ACTIVE").upper()
        if not has_game:
            health, label = 0.0, "Bye"
        elif status_raw in ("OUT", "IR", "INJURY_RESERVE", "SUSPENSION"):
            health, label = 0.0, "Out"
        elif status_raw == "DOUBTFUL":
            health, label = 0.4, "Doubtful"
        elif status_raw == "QUESTIONABLE":
            health, label = 0.75, "Questionable"
        else:
            health, label = 1.0, "Active"

        vol = 35.0 if pos == "QB" else (18.0 if pos == "RB" else (9.0 if pos == "WR" else 6.0))

        rows.append({
            "Player": p.name, "Position": pos, "Status": label,
            "ESPN_Proj": round(proj, 1), "Avg_Pts": round(avg_pts, 1),
            "Last_Week_Pts": round(last_pts, 1), "Form_L3": round(form3, 1),
            "Volume_Proj": float(vol),
            "Opp_Def_Rank": int(opp_rank), "Is_Home": bool(is_home), "Is_Dome": bool(is_dome),
            "Vegas_Total": float(total), "Impl_Team_Total": round(float(impl), 1), "Health": float(health),
        })

    if missing_odds:
        notes.append(f"Vegas total unavailable for {missing_odds} players. Baseline 44.0 used.")
    if skipped:
        notes.append(f"{skipped} players (K/DST) bypassed. Model focuses on QB/RB/WR/TE.")
    return pd.DataFrame(rows), notes


# ============================================================================
# CSV / TEXT IMPORT PARSERS
# ============================================================================
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "Status": "Active", "ESPN_Proj": 10.0, "Opp_Def_Rank": 16, "Is_Home": True, "Is_Dome": False,
        "Vegas_Total": 44.0, "Volume_Proj": 12.0, "Health": 1.0,
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    if "Avg_Pts" not in df.columns:
        df["Avg_Pts"] = df["ESPN_Proj"]
    if "Last_Week_Pts" not in df.columns:
        df["Last_Week_Pts"] = df["Avg_Pts"]
    if "Form_L3" not in df.columns:
        df["Form_L3"] = df["Last_Week_Pts"]
    if "Impl_Team_Total" not in df.columns:
        df["Impl_Team_Total"] = df["Vegas_Total"] / 2.0
    order = ["Player", "Position", "Status", "ESPN_Proj", "Avg_Pts", "Last_Week_Pts", "Form_L3",
             "Volume_Proj", "Opp_Def_Rank", "Is_Home", "Is_Dome", "Vegas_Total", "Impl_Team_Total", "Health"]
    return df[[c for c in order if c in df.columns]]


def parse_raw_espn_text(text):
    rows = []
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    for line in lines:
        parts = re.split(r"\t+|\s{2,}", line)
        if len(parts) >= 4:
            try:
                pos = parts[0].upper()
                name = parts[1]
                nums = [float(p.replace(",", ".")) for p in parts if re.match(r"^-?\d+(\.\d+)?$", p.replace(",", "."))]
                proj = nums[0] if len(nums) > 0 else 10.0
                avg = nums[1] if len(nums) > 1 else proj
                last = nums[2] if len(nums) > 2 else avg
                rows.append({
                    "Player": name, "Position": pos if pos in POSITIONS else "WR", "Status": "Active",
                    "ESPN_Proj": proj, "Avg_Pts": avg, "Last_Week_Pts": last,
                })
            except Exception:
                continue
    return ensure_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()


# ============================================================================
# STREAMLIT UI
# ============================================================================
st.set_page_config(page_title="StressFreeFantasy Pro", page_icon="🏈", layout="wide")

st.title("🏈 StressFreeFantasy: Automated Lineup & Start/Sit Engine")
st.caption("Powered by TabPFN tabular foundation model trained on official NFL history (nflverse).")

# ---- Sidebar Settings ------------------------------------------------------
MODE_ESPN_LABEL = "My League Rules (ESPN Live Sync)"
if st.session_state.pop("_switch_to_espn", False):
    st.session_state["scoring_mode"] = MODE_ESPN_LABEL

with st.sidebar:
    st.header("⚙️ Configuration")
    MODE_ESPN = "My League Rules (ESPN Live Sync)"
    MODE_CUSTOM = "Custom Scoring Rules"
    scoring_mode = st.selectbox(
        "Scoring System", PRESET_NAMES + [MODE_ESPN, MODE_CUSTOM], key="scoring_mode",
        help="Model re-trains on nflverse history using your exact scoring structure.",
    )
    n_train = st.slider("Context Samples (TabPFN Training Rows)", 500, 4000, N_TRAIN_DEFAULT, step=250,
                        help="More rows increase contextual accuracy.")
    st.subheader("Roster Slots")
    n_qb = st.number_input("QB", 0, 3, 1)
    n_rb = st.number_input("RB", 0, 4, 2)
    n_wr = st.number_input("WR", 0, 4, 2)
    n_te = st.number_input("TE", 0, 3, 1)
    n_flex = st.number_input("FLEX", 0, 3, 1)
    flex_pos = st.multiselect("Eligible Positions for FLEX", POSITIONS, default=["RB", "WR", "TE"])

season_default = 2026

# ---- Scoring Rule Assignment -----------------------------------------------
if scoring_mode in PRESET_NAMES:
    rules = preset_rules(scoring_mode)
elif scoring_mode == MODE_ESPN:
    rules = st.session_state.get("espn_rules")
    if not rules:
        st.warning("Connect to your ESPN league under the ESPN tab to auto-fetch rules. Defaulting to PPR.")
        rules = preset_rules("PPR")
    elif st.session_state.get("espn_rules_unsupported"):
        with st.sidebar.expander("ℹ️ Scoring Rule Notes", expanded=False):
            st.caption(
                "Niche big-play bonuses (e.g. 40+/50+ yd TD bonuses) are bypassed during training "
                "as baseline weekly aggregates are used."
            )
else:
    with st.expander("📐 Custom Scoring Rules Matrix", expanded=True):
        if "custom_rules" not in st.session_state:
            st.session_state["custom_rules"] = preset_rules("PPR")
            st.session_state["custom_ver"] = 0
        up = st.file_uploader("Upload rules (JSON or CSV with columns stat,points)", type=["json", "csv"], key="rules_upload")
        if up is not None and st.session_state.get("_rules_sig") != (up.name, up.size):
            try:
                st.session_state["custom_rules"] = parse_rules_file(up)
                st.session_state["custom_ver"] += 1
                st.session_state["_rules_sig"] = (up.name, up.size)
            except Exception as e:
                st.error(f"Failed to parse rule file: {e}")
        base = st.session_state["custom_rules"]
        table = pd.DataFrame({
            "Key": list(RULES),
            "Metric": [RULES[k][0] for k in RULES],
            "Points": [float(base.get(k, 0.0)) for k in RULES],
        })
        edited_rules = st.data_editor(table, hide_index=True, disabled=["Key", "Metric"],
                                      key=f"rules_editor_{st.session_state['custom_ver']}", width="stretch")
        rules = {r.Key: _num(r.Points) for r in edited_rules.itertuples()}
        st.download_button("⬇️ Export Rules (JSON)", json.dumps(rules, indent=2, ensure_ascii=False),
                           file_name="scoring_rules.json", mime="application/json")
rules_key = make_key(rules)

# ---- TabPFN In-Context Fit -------------------------------------------------
model, n_rows, loaded_seasons = None, 0, []
try:
    with st.spinner("Compiling nflverse datasets & priming TabPFN foundation weights..."):
        model, n_rows, loaded_seasons = get_model(rules_key, season_default, n_train)
    st.caption(f"✅ Foundation model primed with {n_rows:,} real player-weeks across seasons {min(loaded_seasons)}–{max(loaded_seasons)} "
               f"under scoring format: **{scoring_mode}**.")
except Exception as e:
    st.error(f"Failed to fit model: {e}")
    st.stop()

with st.expander("🧪 Validation Benchmark: TabPFN vs Traditional Baselines"):
    st.write("Temporal validation: fits on historical seasons and tests on the most recent completed NFL season.")
    if st.button("Run Benchmark"):
        with st.spinner("Evaluating validation splits..."):
            try:
                v = run_validation(rules_key, season_default, min(n_train, 2000))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("TabPFN MAE", f"{v['mae_model']:.2f}")
                c2.metric("Pairwise Start/Sit Accuracy", f"{v['pairwise_acc']:.1f}%")
                c3.metric("Baseline MAE (16-G Avg)", f"{v['mae_avg']:.2f}")
                c4.metric("Pearson Correlation", f"{v['corr_model']:.2f}")
                st.caption(
                    f"Test dataset: {v['n_test']} player-weeks from season {v['test_season']}. "
                    "Pairwise accuracy measures the exact rate at which TabPFN correctly selects the higher scoring player in head-to-head comparisons."
                )
            except Exception as e:
                st.error(f"Benchmark error: {e}")

# ---- Roster Import Tabs ----------------------------------------------------
tab_manual, tab_api = st.tabs(["📋 Option 1: Paste Text / CSV Upload",
                               "🔌 Option 2: Live ESPN Private League Sync"])

with tab_manual:
    st.markdown("### Paste Roster or Upload CSV")
    st.write("Copy your lineup table from ESPN/Yahoo/Sleeper (`Ctrl+C`) and paste below:")
    raw_text = st.text_area("Raw roster text:", height=130,
                            placeholder="QB\tMatthew Stafford\t...\t41.0\t41.3\t55.0\nRB\tBijan Robinson\t...\t20.1\t25.0\t34.0")
    uploaded_file = st.file_uploader("Or upload CSV file:", type=["csv"])

    if uploaded_file is not None:
        try:
            st.session_state["active_roster"] = ensure_columns(pd.read_csv(uploaded_file))
            st.success("CSV roster loaded successfully!")
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
    elif raw_text:
        parsed = parse_raw_espn_text(raw_text)
        if not parsed.empty:
            st.session_state["active_roster"] = parsed
            st.success(f"Parsed {len(parsed)} players from text!")

with tab_api:
    st.markdown("### Automatic Private ESPN Sync")
    with st.expander("ℹ️ How to locate ESPN private keys (20 seconds)"):
        st.markdown("""
        1. Open your league in Brave / Chrome.
        2. Press **F12** to toggle Developer Tools.
        3. Navigate to **Application** -> expand **Cookies** -> `https://fantasy.espn.com`.
        4. Copy the values for **`espn_s2`** and **`SWID`**.
        """)

    c1, c2 = st.columns(2)
    with c1:
        league_id_input = st.text_input("League ID", value="1189143735")
        season_year = st.number_input("Season Year", value=season_default, step=1)
    with c2:
        swid_input = st.text_input("SWID", placeholder="{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}")
        espn_s2_input = st.text_input("espn_s2", type="password", placeholder="Long cookie string...")

    if st.button("Connect to ESPN"):
        try:
            from espn_api.football import League

            league = League(
                league_id=int(league_id_input),
                year=int(season_year),
                espn_s2=espn_s2_input.strip() if espn_s2_input else None,
                swid=swid_input.strip() if swid_input else None,
            )
            st.session_state["espn_league"] = league
            st.session_state["team_names"] = [f"{t.team_name} (ID: {t.team_id})" for t in league.teams]
            msg = f"Connected! Found {len(league.teams)} teams."
            try:
                espn_rules, unsupported = rules_from_espn(league.settings.scoring_format)
                if espn_rules:
                    st.session_state["espn_rules"] = espn_rules
                    st.session_state["espn_rules_unsupported"] = unsupported
                    st.session_state["_switch_to_espn"] = True
                    msg += " League scoring rules imported and applied."
            except Exception as e:
                msg += f" (Could not parse rules automatically: {e})"
            st.session_state["connect_msg"] = msg
            st.rerun()
        except Exception as e:
            st.error(f"Connection failed: {e}. Verify cookie values.")

    if "connect_msg" in st.session_state:
        st.success(st.session_state["connect_msg"])

    if "espn_league" in st.session_state:
        selected_team_str = st.selectbox("Select Your Team:", st.session_state["team_names"])
        if st.button("Fetch Team Roster"):
            league = st.session_state["espn_league"]
            chosen_id = int(re.search(r"ID: (\d+)", selected_team_str).group(1))
            team = next(t for t in league.teams if t.team_id == chosen_id)
            with st.spinner("Compiling matchups, Vegas odds, and defensive ranks..."):
                roster_df, notes = build_roster_from_espn(team, league.current_week, int(season_year), rules_key)
            st.session_state["active_roster"] = roster_df
            st.session_state["roster_notes"] = notes
            st.rerun()

for n in st.session_state.get("roster_notes", []):
    st.info(n)

# ---- Default Demo Lineup ---------------------------------------------------
if "active_roster" in st.session_state:
    active_roster = st.session_state["active_roster"]
else:
    active_roster = ensure_columns(pd.DataFrame([
        {"Player": "Matthew Stafford", "Position": "QB", "Status": "Active", "ESPN_Proj": 41.0, "Avg_Pts": 41.3, "Last_Week_Pts": 55.0, "Volume_Proj": 36.0, "Opp_Def_Rank": 17, "Is_Home": False, "Is_Dome": False, "Vegas_Total": 46.5, "Health": 1.0},
        {"Player": "Lamar Jackson", "Position": "QB", "Status": "Active", "ESPN_Proj": 40.9, "Avg_Pts": 39.3, "Last_Week_Pts": 35.0, "Volume_Proj": 38.0, "Opp_Def_Rank": 2, "Is_Home": True, "Is_Dome": False, "Vegas_Total": 41.0, "Health": 1.0},
        {"Player": "Bijan Robinson", "Position": "RB", "Status": "Active", "ESPN_Proj": 20.1, "Avg_Pts": 25.0, "Last_Week_Pts": 34.0, "Volume_Proj": 22.0, "Opp_Def_Rank": 26, "Is_Home": False, "Is_Dome": True, "Vegas_Total": 44.0, "Health": 1.0},
        {"Player": "Ashton Jeanty", "Position": "RB", "Status": "Active", "ESPN_Proj": 18.0, "Avg_Pts": 17.3, "Last_Week_Pts": 11.0, "Volume_Proj": 19.0, "Opp_Def_Rank": 15, "Is_Home": True, "Is_Dome": True, "Vegas_Total": 47.5, "Health": 1.0},
        {"Player": "Garrett Wilson", "Position": "WR", "Status": "Active", "ESPN_Proj": 15.9, "Avg_Pts": 18.3, "Last_Week_Pts": 26.0, "Volume_Proj": 10.5, "Opp_Def_Rank": 8, "Is_Home": False, "Is_Dome": False, "Vegas_Total": 39.5, "Health": 1.0},
        {"Player": "Zay Flowers", "Position": "WR", "Status": "Questionable", "ESPN_Proj": 14.6, "Avg_Pts": 21.5, "Last_Week_Pts": 15.0, "Volume_Proj": 8.5, "Opp_Def_Rank": 17, "Is_Home": True, "Is_Dome": False, "Vegas_Total": 41.0, "Health": 0.75},
        {"Player": "DJ Moore", "Position": "WR", "Status": "Active", "ESPN_Proj": 12.7, "Avg_Pts": 11.3, "Last_Week_Pts": 12.0, "Volume_Proj": 7.5, "Opp_Def_Rank": 19, "Is_Home": True, "Is_Dome": False, "Vegas_Total": 43.5, "Health": 1.0},
        {"Player": "Isaiah Likely", "Position": "TE", "Status": "Active", "ESPN_Proj": 11.7, "Avg_Pts": 12.7, "Last_Week_Pts": 3.0, "Volume_Proj": 6.0, "Opp_Def_Rank": 27, "Is_Home": True, "Is_Dome": False, "Vegas_Total": 41.0, "Health": 1.0},
        {"Player": "Justin Jefferson", "Position": "WR", "Status": "Out", "ESPN_Proj": 0.0, "Avg_Pts": 14.7, "Last_Week_Pts": 5.0, "Volume_Proj": 0.0, "Opp_Def_Rank": 4, "Is_Home": True, "Is_Dome": True, "Vegas_Total": 45.0, "Health": 0.0},
    ]))

# ---- Roster Inspector & Optimization ---------------------------------------
st.divider()
st.subheader("Review & Fine-Tune Roster Inputs")

COLUMN_CONFIG = {
    "Player": st.column_config.TextColumn("Player"),
    "Position": st.column_config.TextColumn("Pos", width="small"),
    "Status": st.column_config.TextColumn("Status", width="small"),
    "ESPN_Proj": st.column_config.NumberColumn("ESPN Proj", format="%.1f pts"),
    "Avg_Pts": st.column_config.NumberColumn("Season Avg", format="%.1f pts"),
    "Last_Week_Pts": st.column_config.NumberColumn("Last Week", format="%.1f pts"),
    "Form_L3": st.column_config.NumberColumn("Form (L3)", format="%.1f pts"),
    "Volume_Proj": st.column_config.NumberColumn("Proj Volume", format="%.1f"),
    "Opp_Def_Rank": st.column_config.NumberColumn("Opp Rank", help="1 = Toughest, 32 = Easiest"),
    "Is_Home": st.column_config.CheckboxColumn("Home"),
    "Is_Dome": st.column_config.CheckboxColumn("Dome"),
    "Vegas_Total": st.column_config.NumberColumn("Vegas O/U", format="%.1f"),
    "Impl_Team_Total": st.column_config.NumberColumn("Implied Total", format="%.1f"),
    "Health": st.column_config.NumberColumn("Health", format="%.2f", min_value=0.0, max_value=1.0),
}

edited_df = st.data_editor(
    active_roster,
    num_rows="dynamic",
    width="stretch",
    column_config=COLUMN_CONFIG
)

if st.button("Run TabPFN Lineup Optimization", type="primary"):
    if edited_df.empty:
        st.warning("Roster table is empty.")
        st.stop()

    res = ensure_columns(edited_df).reset_index(drop=True)
    X = to_matrix(res)

    mean_pred = np.asarray(model.predict(X), dtype=float)
    try:
        q = model.predict(X, output_type="quantiles", quantiles=[0.1, 0.9])
        floor_pred, ceil_pred = np.asarray(q[0], dtype=float), np.asarray(q[1], dtype=float)
    except Exception:
        floor_pred, ceil_pred = None, None

    health = pd.to_numeric(res["Health"], errors="coerce").fillna(1.0).clip(0, 1).values
    res["TabPFN_Proj"] = np.round(np.maximum(mean_pred, 0.0) * health, 1)
    res["Floor"] = np.round(np.maximum(floor_pred, 0.0) * health, 1) if floor_pred is not None else np.nan
    res["Ceiling"] = np.round(np.maximum(ceil_pred, 0.0) * health, 1) if ceil_pred is not None else np.nan
    res["Edge"] = np.round(res["TabPFN_Proj"] - pd.to_numeric(res["ESPN_Proj"], errors="coerce").fillna(0), 1)
    res = res.sort_values("TabPFN_Proj", ascending=False)

    assigned, starters = set(), []

    def add_starter(slot, idx, row):
        assigned.add(idx)
        starters.append({
            "Slot": slot, "Player": row["Player"], "Position": row["Position"],
            "TabPFN Projected": row["TabPFN_Proj"], "Floor (P10)": row["Floor"], "Ceiling (P90)": row["Ceiling"],
            "ESPN Proj": row["ESPN_Proj"], "Matchup Edge": f"{row['Edge']:+0.1f}",
        })

    def pick(positions, count, label):
        avail = res[res["Position"].isin(positions) & (~res.index.isin(assigned)) & (res["Health"] > 0)]
        for idx, row in avail.head(int(count)).iterrows():
            add_starter(label, idx, row)

    pick(["QB"], n_qb, "QB")
    pick(["RB"], n_rb, "RB")
    pick(["WR"], n_wr, "WR")
    pick(["TE"], n_te, "TE")
    pick(flex_pos or ["RB", "WR"], n_flex, "FLEX")

    bench_df = res[~res.index.isin(assigned)][["Player", "Position", "Status", "TabPFN_Proj", "Floor", "Ceiling", "ESPN_Proj", "Edge"]]
    bench_df.columns = ["Player", "Position", "Status", "TabPFN Projected", "Floor (P10)", "Ceiling (P90)", "ESPN Proj", "Matchup Edge"]
    starters_df = pd.DataFrame(starters)

    st.markdown("---")
    st.subheader("Optimal Starting Lineup")
    if starters_df.empty:
        st.warning("No eligible players found to start. Verify position assignments and health factors.")
        st.stop()
    st.metric("Total Projected Points", f"{starters_df['TabPFN Projected'].sum():.1f} pts")

    col_s, col_b = st.columns([3, 2])
    with col_s:
        st.success("### 🟢 START THESE PLAYERS")
        st.dataframe(starters_df, width="stretch", hide_index=True)
    with col_b:
        st.warning("### 🟡 BENCH")
        st.dataframe(bench_df, width="stretch", hide_index=True)

    st.info(
        "💡 **TabPFN Decision Matrix:** TabPFN prioritizes opportunities (targets/carries) adjusted for opponent defensive resistance, "
        "dome conditions, and Vegas implied totals. Floor (P10) provides safe variance-limiting starters, while Ceiling (P90) highlights upside plays."
    )