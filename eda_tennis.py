"""
Exploratory data analysis of the Tennis Abstract Match Charting Project data.

Reads match metadata, shot-by-shot point records, and per-match aggregate stats
for both the men's and women's tours from the SQLite database built by
tennis_db.py, prints summary tables to the console, and saves a set of charts
to an output directory. If the database does not exist yet it is built first.

Usage:
    python eda_tennis.py                 # writes figures to ./eda_output
    python eda_tennis.py --out figs      # choose another output directory
    python eda_tennis.py --show          # also open each figure in a window
    python eda_tennis.py --db other.db   # read a different database file
    python eda_tennis.py --rebuild       # rebuild the database from the CSVs first

Data: Jeff Sackmann / Tennis Abstract Match Charting Project, licensed
CC BY-NC-SA 4.0 (see tennis_MatchChartingProject/README.md).
"""

import argparse
from collections.abc import Mapping, Sequence
import os
import sqlite3

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

import tennis_db

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TOURS = {"M": "Men", "W": "Women"}
SURFACES = ["Hard", "Clay", "Grass"]

# Shot letters in the MCP notation (serves are digits 0/4/5/6; see the
# 'Instructions' tab of the MatchChart workbook).
SHOT_CHARS = r"[fbrsvzopuylmhijktq]"
SERVE_DIRECTIONS = {"4": "Wide", "5": "Body", "6": "T", "0": "Unknown"}
OUTCOME_ORDER = ["Ace", "Unreturned serve", "Double fault",
                 "Winner", "Forced error", "Unforced error"]
# 'Pts' is recorded server-first, e.g. '30-40' means the server trails.
BREAK_POINT_SCORES = {"0-40", "15-40", "30-40", "40-AD"}
GAME_POINT_SCORES = {"40-0", "40-15", "40-30", "AD-40"}

# Chart styling: a quiet light theme with one fixed color per tour.
SURFACE_BG = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
TOUR_COLORS = {"Men": "#2a78d6", "Women": "#eb6834"}
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32b2a"])
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pct(numerator: npt.ArrayLike, denominator: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Percentage that returns NaN instead of dividing by zero."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 100.0 * num / den, np.nan)


def query(con: sqlite3.Connection, sql: str,
          params: Mapping[str, str | int | float | None] | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


def setup_matplotlib(show: bool) -> None:
    if not show:
        plt.switch_backend("Agg")  # render to files only
    plt.rcParams.update({
        "figure.facecolor": SURFACE_BG,
        "axes.facecolor": SURFACE_BG,
        "savefig.facecolor": SURFACE_BG,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.edgecolor": BASELINE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    })


def save(fig: Figure, out_dir: str, name: str, show: bool) -> None:
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def grouped_bars(ax: Axes, categories: Sequence[str], series: Mapping[str, npt.ArrayLike],
                 horizontal: bool = False, value_fmt: str = "{:.1f}") -> None:
    """Draw one bar per tour for each category; `series` maps tour -> values."""
    n = len(series)
    width = 0.8 / max(n, 1)
    positions = np.arange(len(categories))
    for i, (tour, values) in enumerate(series.items()):
        offsets = positions - 0.4 + width * (i + 0.5)
        heights = np.asarray(values, dtype=float)
        if horizontal:
            bars = ax.barh(offsets, heights, height=width, color=TOUR_COLORS[tour],
                           edgecolor=SURFACE_BG, linewidth=1, label=tour)
        else:
            bars = ax.bar(offsets, heights, width=width, color=TOUR_COLORS[tour],
                          edgecolor=SURFACE_BG, linewidth=1, label=tour)
        ax.bar_label(bars, labels=[value_fmt.format(v) if np.isfinite(v) else ""
                                   for v in heights],
                     padding=3, fontsize=8, color=INK_SECONDARY)
    if horizontal:
        ax.set_yticks(positions, categories)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    else:
        ax.set_xticks(positions, categories)
        ax.grid(axis="x", visible=False)
    ax.legend()


# ----------------------------------------------------------------------------
# Loading and cleaning
# ----------------------------------------------------------------------------

def database_inventory(con: sqlite3.Connection, db_path: str) -> None:
    section("1. DATABASE INVENTORY")
    tables = query(con, """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name""")["name"]
    rows = []
    for table in tables:
        n_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        n_cols = len(con.execute(f"PRAGMA table_info({table})").fetchall())
        rows.append({"table": table, "rows": n_rows, "columns": n_cols})
    inventory = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None):
        print(inventory.to_string(index=False))
    print(f"\n{len(inventory)} tables, {inventory['rows'].sum():,} rows, "
          f"{os.path.getsize(db_path) / 1e6:,.0f} MB ({db_path})")
    views = query(con, "SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name")["name"]
    print(f"Views: {', '.join(views) if len(views) else 'none'}")

    sources = query(con, """
        SELECT table_name, COUNT(*) AS csv_files, SUM(rows_read) AS rows_read,
               MAX(loaded_at) AS loaded_at
        FROM source_files GROUP BY table_name ORDER BY table_name""")
    print("\nRead from CSV files:")
    print(sources.to_string(index=False))


# One row per match with its tournament edition and both players' names.
MATCHES_SQL = """
    SELECT m.*, t.name AS tournament, t.tour, t.season, t.surface,
           p1.name AS player1, p1.hand AS player1_hand,
           p2.name AS player2, p2.hand AS player2_hand
    FROM matches AS m
    JOIN tournaments AS t USING (tournament_id)
    JOIN players AS p1 ON p1.player_id = m.player1_id
    JOIN players AS p2 ON p2.player_id = m.player2_id"""


def load_matches(con: sqlite3.Connection) -> pd.DataFrame:
    matches = query(con, MATCHES_SQL)
    matches["tour"] = matches["tour"].map(TOURS)
    matches["match_date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d")
    matches["year"] = matches["match_date"].dt.year
    matches["surface"] = matches["surface"].fillna("Unknown")
    for side in ("1", "2"):
        hand = matches[f"player{side}_hand"]
        matches[f"hand{side}"] = hand.where(hand.isin(["R", "L"]), "Unknown")
    return matches


def describe_matches(con: sqlite3.Connection, matches: pd.DataFrame) -> None:
    section("2. MATCH METADATA")
    print(f"Charted matches: {len(matches):,}  "
          f"({', '.join(f'{t}: {n:,}' for t, n in matches['tour'].value_counts().items())})")
    print(f"Date range: {matches['match_date'].min():%Y-%m-%d} to "
          f"{matches['match_date'].max():%Y-%m-%d}")

    editions = query(con, "SELECT COUNT(*) AS n, COUNT(DISTINCT name) AS names FROM tournaments").iloc[0]
    print(f"Tournament editions: {editions['n']:,} ({editions['names']:,} tournament names); "
          f"players: {con.execute('SELECT COUNT(*) FROM players').fetchone()[0]:,}")

    print("\nMissing values per column (% of matches):")
    cols = ["round", "time", "court", "umpire", "best_of", "final_tb", "charted_by",
            "player1_hand", "player2_hand"]
    missing = matches[cols].isna().mean() * 100
    missing["surface"] = (matches["surface"] == "Unknown").mean() * 100
    print(missing[missing > 0].round(2).sort_values(ascending=False).to_string())

    print("\nMatches by surface and tour:")
    print(pd.crosstab(matches["surface"], matches["tour"], margins=True).to_string())

    print("\nMatches by best-of format and tour:")
    print(pd.crosstab(matches["best_of"], matches["tour"]).to_string())

    print("\nMost-charted rounds:")
    print(query(con, """
        SELECT round, COUNT(*) AS matches FROM matches
        GROUP BY round ORDER BY matches DESC LIMIT 10""").to_string(index=False))

    print("\nMost-charted tournaments (all editions):")
    print(query(con, """
        SELECT t.name AS tournament, COUNT(DISTINCT t.tournament_id) AS editions, COUNT(*) AS matches
        FROM matches AS m JOIN tournaments AS t USING (tournament_id)
        GROUP BY t.name ORDER BY matches DESC LIMIT 10""").to_string(index=False))

    hands = pd.concat([matches["hand1"], matches["hand2"]])
    known = hands[hands != "Unknown"]
    print(f"\nLeft-handed player appearances: {(known == 'L').mean():.1%} of {len(known):,}")

    charters = query(con, """
        SELECT charted_by, COUNT(*) AS matches FROM matches
        WHERE charted_by IS NOT NULL
        GROUP BY charted_by ORDER BY matches DESC""")
    print(f"\nContributors: {len(charters):,}; top 10 charted "
          f"{charters['matches'].head(10).sum() / charters['matches'].sum():.0%} of matches")
    print(charters.head(10).to_string(index=False))


def top_players(con: sqlite3.Connection, tour: str, limit: int = 15) -> pd.Series:
    """Most-charted players for one tour ('M' or 'W'), counting both player slots."""
    return query(con, """
        SELECT p.name AS player, COUNT(*) AS matches
        FROM (SELECT m.player1_id AS player_id FROM matches AS m
              JOIN tournaments AS t USING (tournament_id) WHERE t.tour = :tour
              UNION ALL
              SELECT m.player2_id FROM matches AS m
              JOIN tournaments AS t USING (tournament_id) WHERE t.tour = :tour) AS a
        JOIN players AS p USING (player_id)
        GROUP BY p.player_id ORDER BY matches DESC LIMIT :limit""",
        params={"tour": tour, "limit": limit}).set_index("player")["matches"]


def derive_point_features(points: pd.DataFrame) -> pd.DataFrame:
    """Parse the MCP shot codes into per-point serve/rally/outcome features."""
    first = points["first_serve"].fillna("").astype(str).str.replace(r"^c+", "", regex=True)
    second = points["second_serve"].fillna("").astype(str).str.replace(r"^c+", "", regex=True)
    has_second = second.ne("")
    played = second.where(has_second, first)  # the serve that decided the point

    shots = played.str.count(SHOT_CHARS)
    mark = played.str.extract(r"([*#@])[^*#@]*$")[0]  # final outcome marker
    starts_with_serve = played.str.match(r"^[0456]")

    outcome = np.select(
        [
            (shots == 0) & (mark == "*"),
            (shots == 0) & (mark == "#"),
            has_second & (shots == 0) & mark.isna() & starts_with_serve,
            (shots > 0) & (mark == "*"),
            (shots > 0) & (mark == "#"),
            (shots > 0) & (mark == "@"),
        ],
        OUTCOME_ORDER,
        default="Other",
    )

    # Rally length by the usual convention: include the serve, exclude errors.
    ends_in_error = (shots > 0) & mark.isin(["#", "@"])
    rally_len = (1 + shots - ends_in_error.astype(int)).astype(float)
    rally_len[(outcome == "Double fault") | (outcome == "Other")] = np.nan

    valid_serve = first.str.match(r"^[0456]")

    return pd.DataFrame({
        "match_id": points["match_id"],
        "tour": points["tour"].map(TOURS),
        "pts": points["pts"].astype("category"),
        "valid_serve": valid_serve,
        "first_in": ~has_second,
        "serve1_dir": first.str[:1].map(SERVE_DIRECTIONS).astype("category"),
        "serve2_dir": second.str[:1].map(SERVE_DIRECTIONS).astype("category"),
        "outcome": pd.Categorical(outcome, categories=OUTCOME_ORDER + ["Other"]),
        "rally_len": rally_len,
        "server_won": points["svr"] == points["pt_winner"],
    })


def load_points(con: sqlite3.Connection, matches: pd.DataFrame, chunksize: int = 250_000) -> pd.DataFrame:
    """Stream the points table in chunks, keeping only the derived features."""
    frames: list[pd.DataFrame] = []
    chunks = pd.read_sql_query("""
        SELECT p.match_id, t.tour, p.pts, p.svr, p.first_serve, p.second_serve, p.pt_winner
        FROM points AS p
        JOIN matches AS m USING (match_id)
        JOIN tournaments AS t USING (tournament_id)""", con, chunksize=chunksize)
    for raw in chunks:
        frames.append(derive_point_features(raw))
        print(f"  parsed {sum(len(f) for f in frames):,} points", flush=True)
    points = pd.concat(frames, ignore_index=True)
    meta = matches.set_index("match_id")
    points["surface"] = points["match_id"].map(meta["surface"]).fillna("Unknown")
    points["year"] = points["match_id"].map(meta["year"])
    for col in ("match_id", "tour", "surface"):
        points[col] = points[col].astype("category")
    return points


def load_overview(con: sqlite3.Connection) -> pd.DataFrame:
    """Whole-match ('Total') rows of set_stats for players with >= 30 serve points."""
    ov = query(con, """
        SELECT s.*, p.name AS player, t.tour, COALESCE(t.surface, 'Unknown') AS surface
        FROM set_stats AS s
        JOIN players AS p USING (player_id)
        JOIN matches AS m USING (match_id)
        JOIN tournaments AS t USING (tournament_id)
        WHERE s.set_label = 'Total' AND s.serve_pts >= 30""")
    ov["tour"] = ov["tour"].map(TOURS)

    total_pts = ov["serve_pts"] + ov["return_pts"]
    ov["ace_pct"] = pct(ov["aces"], ov["serve_pts"])
    ov["df_pct"] = pct(ov["dfs"], ov["serve_pts"])
    ov["first_in_pct"] = pct(ov["first_in"], ov["serve_pts"])
    ov["first_won_pct"] = pct(ov["first_won"], ov["first_in"])
    ov["second_won_pct"] = pct(ov["second_won"], ov["second_in"])
    ov["serve_won_pct"] = pct(ov["first_won"] + ov["second_won"], ov["serve_pts"])
    ov["return_won_pct"] = pct(ov["return_pts_won"], ov["return_pts"])
    ov["bp_saved_pct"] = pct(ov["bp_saved"], ov["bk_pts"])
    ov["winners_per_100"] = pct(ov["winners"], total_pts)
    ov["unforced_per_100"] = pct(ov["unforced"], total_pts)
    ov["total_pts_won_pct"] = pct(ov["first_won"] + ov["second_won"] + ov["return_pts_won"],
                                  total_pts)
    return ov


# ----------------------------------------------------------------------------
# Analyses (console output + figures)
# ----------------------------------------------------------------------------

def plot_matches_over_time(matches: pd.DataFrame, out_dir: str, show: bool) -> None:
    counts = (matches.dropna(subset=["year"])
              .groupby(["year", "tour"]).size().unstack(fill_value=0).sort_index())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for tour in counts.columns:
        ax.plot(counts.index, counts[tour], color=TOUR_COLORS[tour], label=tour, marker="o",
                markersize=4, markeredgecolor=SURFACE_BG, markeredgewidth=1)
    peak = int(np.argmax(counts.sum(axis=1).to_numpy()))  # position of the busiest year
    peak_year = int(counts.index[peak])
    ax.set_title("Charted matches per year")
    ax.set_xlabel("Match year")
    ax.set_ylabel("Matches")
    ax.annotate(f"peak: {peak_year}", xy=(peak_year, float(counts.iloc[peak].max())),
                xytext=(0, 8), textcoords="offset points", ha="center",
                fontsize=8, color=INK_SECONDARY)
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, out_dir, "01_matches_per_year.png", show)


def plot_surface_and_rounds(matches: pd.DataFrame, out_dir: str, show: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    surf = pd.crosstab(matches["surface"], matches["tour"]).reindex(SURFACES)
    grouped_bars(axes[0], SURFACES, {t: surf[t] for t in TOURS.values()}, value_fmt="{:,.0f}")
    axes[0].set_title("Matches by surface")
    axes[0].set_ylabel("Matches")

    rounds = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "RR"]
    rnd = pd.crosstab(matches["round"], matches["tour"]).reindex(rounds).fillna(0)
    grouped_bars(axes[1], rounds, {t: rnd[t] for t in TOURS.values()}, value_fmt="{:,.0f}")
    axes[1].set_title("Matches by round")
    axes[1].tick_params(axis="x", labelsize=8)
    save(fig, out_dir, "02_matches_by_surface_and_round.png", show)


def plot_top_players(con: sqlite3.Connection, out_dir: str, show: bool) -> None:
    section("3. MOST-CHARTED PLAYERS")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (code, tour) in zip(axes, TOURS.items()):
        top = top_players(con, code)
        print(f"\n{tour}:")
        print(top.to_string())
        bars = ax.barh(top.index[::-1], top.values[::-1], height=0.6, color=TOUR_COLORS[tour])
        ax.bar_label(bars, padding=3, fontsize=8, color=INK_SECONDARY)
        ax.set_title(f"{tour}: most-charted players")
        ax.set_xlabel("Charted matches")
        ax.grid(axis="y", visible=False)
    save(fig, out_dir, "03_top_players.png", show)


def analyze_points(con: sqlite3.Connection, points: pd.DataFrame, matches: pd.DataFrame,
                   out_dir: str, show: bool) -> None:
    section("4. POINT-BY-POINT DATA")
    n_matches_with_points = points["match_id"].nunique()
    print(f"Points: {len(points):,} across {n_matches_with_points:,} matches")
    missing_meta = (points["surface"] == "Unknown").sum()
    print(f"Points whose match has no recorded surface: {missing_meta:,}")
    no_points = con.execute("""
        SELECT COUNT(*) FROM matches AS m
        WHERE NOT EXISTS (SELECT 1 FROM points AS p WHERE p.match_id = m.match_id)
    """).fetchone()[0]
    print(f"Matches with no point records: {no_points:,}")
    print(f"Points with unparseable/penalty codes ('Other'): "
          f"{(points['outcome'] == 'Other').mean():.2%}")

    # --- points per match ---------------------------------------------------
    per_match = points.groupby(["match_id", "tour"], observed=True).size().rename("points").reset_index()
    print("\nPoints per match:")
    print(per_match.groupby("tour", observed=True)["points"].describe().round(1).to_string())

    # Game and set counts come from the points_detail view, which numbers the games.
    games = query(con, """
        SELECT t.tour, d.match_id, MAX(d.game_num) AS games, MAX(d.set_num) AS sets
        FROM points_detail AS d
        JOIN matches AS m USING (match_id)
        JOIN tournaments AS t USING (tournament_id)
        GROUP BY d.match_id""")
    games["tour"] = games["tour"].map(TOURS)
    print("\nGames and sets per match (from the points_detail view):")
    print(games.groupby("tour")[["games", "sets"]].describe().loc[
        :, (slice(None), ["mean", "50%", "max"])].round(1).to_string())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.arange(0, per_match["points"].quantile(0.995) + 10, 10).tolist()
    for tour in TOURS.values():
        vals = per_match.loc[per_match["tour"] == tour, "points"]
        ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=TOUR_COLORS[tour],
                label=f"{tour} (median {vals.median():.0f})")
    ax.set_title("Points per charted match")
    ax.set_xlabel("Points in match")
    ax.set_ylabel("Matches")
    ax.legend()
    save(fig, out_dir, "04_points_per_match.png", show)

    # --- serve summary by tour and surface ----------------------------------
    serve = points[points["valid_serve"]]
    summary = serve[serve["surface"].isin(SURFACES)].groupby(["tour", "surface"], observed=True).agg(
        points=("server_won", "size"),
        first_in_pct=("first_in", "mean"),
        server_won_pct=("server_won", "mean"),
        mean_rally=("rally_len", "mean"),
        median_rally=("rally_len", "median"),
    )
    summary["first_in_pct"] *= 100
    summary["server_won_pct"] *= 100
    first_won = serve[serve["first_in"]].groupby(["tour", "surface"], observed=True)["server_won"].mean() * 100
    second_won = serve[~serve["first_in"]].groupby(["tour", "surface"], observed=True)["server_won"].mean() * 100
    summary["first_serve_won_pct"] = first_won
    summary["second_serve_won_pct"] = second_won
    print("\nServe and rally summary by tour and surface:")
    print(summary.round(1).to_string())

    # --- rally length distribution ------------------------------------------
    rally = points.dropna(subset=["rally_len"])
    cap = 15
    capped = rally["rally_len"].clip(upper=cap).astype(int)
    dist = (pd.crosstab(capped, rally["tour"], normalize="columns") * 100).reindex(range(1, cap + 1))
    labels = [str(i) for i in range(1, cap)] + [f"{cap}+"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    grouped_bars(ax, labels, {t: dist[t] for t in TOURS.values()}, value_fmt="{:.0f}")
    ax.set_title("Rally length distribution (shots incl. serve)")
    ax.set_xlabel("Rally length")
    ax.set_ylabel("% of points")
    save(fig, out_dir, "05_rally_length_distribution.png", show)

    share_short = (rally["rally_len"] <= 4).groupby(rally["tour"], observed=True).mean()
    print("\nShare of points decided within 4 shots: "
          + ", ".join(f"{t} {v:.1%}" for t, v in share_short.items()))

    # --- rally length & serve dominance by surface ---------------------------
    by_surf = rally[rally["surface"].isin(SURFACES)].groupby(
        ["surface", "tour"], observed=True)["rally_len"].mean().unstack().reindex(SURFACES)
    srv_surf = serve[serve["surface"].isin(SURFACES)].groupby(
        ["surface", "tour"], observed=True)["server_won"].mean().unstack().reindex(SURFACES) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    grouped_bars(axes[0], SURFACES, {t: by_surf[t] for t in TOURS.values()}, value_fmt="{:.2f}")
    axes[0].set_title("Mean rally length by surface")
    axes[0].set_ylabel("Shots per point")
    grouped_bars(axes[1], SURFACES, {t: srv_surf[t] for t in TOURS.values()})
    axes[1].set_title("Serve points won by surface")
    axes[1].set_ylabel("% of service points won")
    axes[1].set_ylim(40, max(70, np.nanmax(srv_surf.values) + 5))
    save(fig, out_dir, "06_surface_rally_and_serve.png", show)

    # --- server advantage vs rally length -----------------------------------
    # Rally length alone determines who won (odd = server hit the last ball in
    # play), so use two-shot buckets that each hold one odd and one even length.
    buckets = ["1-2", "3-4", "5-6", "7-8", "9-10", "11+"]
    bucket = pd.cut(rally["rally_len"], [0, 2, 4, 6, 8, 10, np.inf], labels=buckets)
    win_by_len = rally.groupby([bucket, rally["tour"]], observed=True)["server_won"].mean().unstack() * 100
    share_by_len = pd.crosstab(bucket, rally["tour"], normalize="columns") * 100
    print("\nServer win % (and share of points) by rally-length bucket:")
    print(pd.concat({"server_won_pct": win_by_len, "pct_of_points": share_by_len},
                    axis=1).round(1).to_string())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    grouped_bars(ax, buckets, {t: win_by_len[t] for t in TOURS.values()})
    ax.axhline(50, color=BASELINE, linewidth=1, zorder=0.5)
    ax.set_title("Server's win rate by rally length")
    ax.set_xlabel("Rally length (shots incl. serve)")
    ax.set_ylabel("% of points won by server")
    ax.set_ylim(0, 100)
    save(fig, out_dir, "07_server_win_by_rally_length.png", show)

    # --- how points end ------------------------------------------------------
    outcomes = (pd.crosstab(points["outcome"], points["tour"], normalize="columns") * 100
                ).reindex(OUTCOME_ORDER)
    print("\nHow points end (% of all points):")
    print(outcomes.round(1).to_string())
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped_bars(ax, OUTCOME_ORDER, {t: outcomes[t] for t in TOURS.values()}, horizontal=True)
    ax.set_title("How points end")
    ax.set_xlabel("% of points")
    save(fig, out_dir, "08_point_outcomes.png", show)

    # --- serve direction ----------------------------------------------------
    dirs = ["Wide", "Body", "T"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    print("\nServe direction (% of serves with known direction):")
    for ax, (col, label) in zip(axes, [("serve1_dir", "1st serve"), ("serve2_dir", "2nd serve")]):
        sub = points[points[col].isin(dirs)]
        share = (pd.crosstab(sub[col], sub["tour"], normalize="columns") * 100).reindex(dirs)
        print(f"  {label}:\n" + share.round(1).to_string())
        grouped_bars(ax, dirs, {t: share[t] for t in TOURS.values()})
        ax.set_title(f"{label} direction")
        ax.set_ylabel("% of serves")
    save(fig, out_dir, "09_serve_direction.png", show)

    # --- pressure points ----------------------------------------------------
    pts_str = serve["pts"].astype(str)
    situation = np.select(
        [pts_str.isin(BREAK_POINT_SCORES), pts_str.isin(GAME_POINT_SCORES), pts_str.eq("40-40")],
        ["Break point", "Game point", "Deuce"], default="Other points")
    order = ["Game point", "Other points", "Deuce", "Break point"]
    pressure = (serve.groupby([situation, serve["tour"]], observed=True)["server_won"]
                .mean().unstack().reindex(order) * 100)
    print("\nServer win % by game situation (standard games, server-first scores):")
    print(pressure.round(1).to_string())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    grouped_bars(ax, order, {t: pressure[t] for t in TOURS.values()})
    ax.set_title("Server's win rate by game situation")
    ax.set_ylabel("% of points won by server")
    ax.set_ylim(40, max(75, np.nanmax(pressure.values) + 5))
    save(fig, out_dir, "10_pressure_points.png", show)

    # --- trends over time ---------------------------------------------------
    match_counts = matches.groupby(["year", "tour"]).size()
    enough = match_counts[match_counts >= 30].index
    yearly = points.dropna(subset=["year"]).groupby(["year", "tour"], observed=True).agg(
        mean_rally=("rally_len", "mean"),
        ace_pct=("outcome", lambda s: (s == "Ace").mean() * 100),
        df_pct=("outcome", lambda s: (s == "Double fault").mean() * 100),
        server_won_pct=("server_won", lambda s: s.mean() * 100),
    )
    yearly = yearly[yearly.index.isin(enough)]
    print("\nYearly trends (years with >= 30 charted matches):")
    print(yearly.round(2).unstack("tour").tail(15).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, title, ylabel) in zip(axes, [
        ("mean_rally", "Mean rally length", "Shots per point"),
        ("ace_pct", "Aces", "% of points"),
        ("server_won_pct", "Server's win rate", "% of points won by server"),
    ]):
        wide = yearly[col].unstack("tour")
        for tour in wide.columns:
            ax.plot(wide.index, wide[tour], color=TOUR_COLORS[tour], label=tour)
        ax.set_title(title)
        ax.set_xlabel("Match year")
        ax.set_ylabel(ylabel)
        ax.legend()
    save(fig, out_dir, "11_trends_by_year.png", show)


def analyze_overview(ov: pd.DataFrame, out_dir: str, show: bool) -> None:
    section("5. PLAYER-MATCH STATS (set_stats, 'Total' rows)")
    rate_cols = ["ace_pct", "df_pct", "first_in_pct", "first_won_pct", "second_won_pct",
                 "return_won_pct", "bp_saved_pct", "winners_per_100", "unforced_per_100",
                 "total_pts_won_pct"]
    print(f"Player-match rows (>= 30 service points): {len(ov):,}")
    print("\nDescriptive statistics:")
    print(ov.groupby("tour")[rate_cols].describe().T.round(1).to_string())

    # Correlation heatmap computed with numpy (rows with any NaN dropped).
    data = ov[rate_cols].dropna().to_numpy()
    corr = np.corrcoef(data, rowvar=False)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(rate_cols)), rate_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rate_cols)), rate_cols)
    ax.grid(False)
    for i in range(len(rate_cols)):
        for j in range(len(rate_cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation between per-match player stats")
    save(fig, out_dir, "12_stat_correlations.png", show)

    target = "total_pts_won_pct"
    idx = rate_cols.index(target)
    ranked = pd.Series(corr[idx], index=rate_cols).drop(target).sort_values(key=np.abs, ascending=False)
    print(f"\nCorrelation of each stat with {target}:")
    print(ranked.round(3).to_string())

    # Winners vs unforced errors, one density panel per tour.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    lim = np.nanpercentile(ov[["winners", "unforced"]].to_numpy(), 99.5)
    for ax, tour in zip(axes, TOURS.values()):
        sub = ov[ov["tour"] == tour]
        hb = ax.hexbin(sub["unforced"], sub["winners"], gridsize=35, cmap=SEQUENTIAL_CMAP,
                       mincnt=1, extent=(0, lim, 0, lim))
        ax.plot([0, lim], [0, lim], color=INK_MUTED, linewidth=1)
        ax.text(lim * 0.97, lim * 0.92, "winners = UEs", ha="right", fontsize=8, color=INK_MUTED)
        r = np.corrcoef(sub["unforced"], sub["winners"])[0, 1]
        ax.set_title(f"{tour}: winners vs unforced errors (r = {r:.2f})")
        ax.set_xlabel("Unforced errors in match")
        ax.set_ylabel("Winners in match")
        ax.grid(False)
        fig.colorbar(hb, ax=ax, label="Player-matches")
    save(fig, out_dir, "13_winners_vs_unforced.png", show)

    # Surface effects on serve stats.
    surf = ov[ov["surface"].isin(SURFACES)]
    print("\nMedian serve/return stats by tour and surface:")
    print(surf.groupby(["tour", "surface"])[["ace_pct", "first_in_pct", "first_won_pct",
                                             "second_won_pct", "return_won_pct"]]
          .median().round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, tour in zip(axes, TOURS.values()):
        groups = [surf.loc[(surf["tour"] == tour) & (surf["surface"] == s), "ace_pct"].dropna()
                  for s in SURFACES]
        bp = ax.boxplot(groups, tick_labels=SURFACES, widths=0.4, showfliers=False,
                        patch_artist=True, medianprops={"color": INK, "linewidth": 1.5},
                        whiskerprops={"color": INK_MUTED}, capprops={"color": INK_MUTED})
        for box in bp["boxes"]:
            box.set(facecolor=TOUR_COLORS[tour], alpha=0.35, edgecolor=TOUR_COLORS[tour])
        ax.set_title(f"{tour}: ace rate per match by surface")
        ax.set_ylabel("Aces per 100 service points")
        ax.grid(axis="x", visible=False)
    save(fig, out_dir, "14_ace_rate_by_surface.png", show)

    # Career aggregates for frequently charted players.
    totals = ov.groupby(["tour", "player"]).agg(
        matches=("match_id", "nunique"),
        serve_pts=("serve_pts", "sum"), aces=("aces", "sum"), dfs=("dfs", "sum"),
        first_in=("first_in", "sum"), first_won=("first_won", "sum"),
        second_in=("second_in", "sum"), second_won=("second_won", "sum"),
        return_pts=("return_pts", "sum"), return_pts_won=("return_pts_won", "sum"),
    )
    totals = totals[totals["matches"] >= 40].copy()
    totals["ace_pct"] = pct(totals["aces"], totals["serve_pts"])
    totals["serve_won_pct"] = pct(totals["first_won"] + totals["second_won"], totals["serve_pts"])
    totals["return_won_pct"] = pct(totals["return_pts_won"], totals["return_pts"])
    totals["dominance"] = totals["return_won_pct"] / (100 - totals["serve_won_pct"])
    cols = ["matches", "ace_pct", "serve_won_pct", "return_won_pct", "dominance"]
    for tour in TOURS.values():
        sub = totals[totals.index.get_level_values("tour") == tour].droplevel("tour")
        print(f"\n{tour}: best servers (>= 40 charted matches, by % serve pts won)")
        print(sub.sort_values("serve_won_pct", ascending=False)[cols].head(10).round(2).to_string())
        print(f"\n{tour}: best returners (by % return pts won)")
        print(sub.sort_values("return_won_pct", ascending=False)[cols].head(10).round(2).to_string())


# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(REPO_DIR, "eda_output"),
                        help="directory for saved figures (default: ./eda_output)")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    parser.add_argument("--db", default=tennis_db.DEFAULT_DB_PATH,
                        help="SQLite database to read (default: ./tennis.db; built if missing)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the database from the CSV files before the analysis")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    setup_matplotlib(args.show)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 30)

    con = tennis_db.connect(args.db, rebuild=args.rebuild)
    try:
        database_inventory(con, args.db)

        matches = load_matches(con)
        describe_matches(con, matches)
        plot_matches_over_time(matches, args.out, args.show)
        plot_surface_and_rounds(matches, args.out, args.show)
        plot_top_players(con, args.out, args.show)

        section("Loading points from the database")
        points = load_points(con, matches)
        analyze_points(con, points, matches, args.out, args.show)
        del points

        analyze_overview(load_overview(con), args.out, args.show)
    finally:
        con.close()

    section("DONE")
    print(f"Figures saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
