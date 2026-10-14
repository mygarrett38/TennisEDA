"""
Build a SQLite database from the Tennis Abstract Match Charting Project CSVs.

The men's ("-m-") and women's ("-w-") files are combined into one relational
schema:

    players          one row per player (name, handedness)
    tournaments      one row per tournament edition: one tour's event in one
                     period, with its season, first/last charted match dates
                     and surface
    matches          one row per charted match -> tournaments, players
    points           one row per point, keyed by (match_id, pt) -> matches
    points_detail    VIEW: points plus set number, game number and whether
                     the set is decided by a tiebreak
    serve_stats, serve_direction, serve_influence, key_points, return_stats,
    shot_stats, net_stats, rally_stats, set_stats
                     per-match aggregate stats -> matches, players
    source_files     which CSV fed which table, and how many rows were read

The build reshapes the source files rather than copying them:
  * Player names and tournament details move out of matches into their own
    tables. A tournament edition is every match with the same (normalised)
    name and tour, split wherever charted matches are more than
    EDITION_GAP_DAYS apart. Its surface and each player's handedness are the
    most common value recorded.
  * points drops columns that can be recomputed (gm_num and TbSet, see the
    points_detail view) and keeps one chart per match.
  * The 16 charting-*-stats-*.csv pairs become 9 tables. Rows and columns that
    are sums or copies of other stats are dropped: SvBreakTotal and ServeBasics
    (sums of SvBreakSplit), ShotDirection (a pivot of ShotDirOutcomes), the
    returner's half of KeyPointsReturn (the opponent's KeyPointsServe rows), and
    total rows that add up other rows.
  * Malformed rows in the -matches- files are repaired (see repair_matches),
    and every repair or dropped duplicate is printed during the build.

Usage:
    python tennis_db.py                  # (re)build ./tennis.db
    python tennis_db.py --db other.db    # build somewhere else

Data: Jeff Sackmann / Tennis Abstract Match Charting Project, licensed
CC BY-NC-SA 4.0 (see README.md).
"""

import argparse
from collections.abc import Callable, Hashable
from functools import partial
import datetime as dt
import glob
import os
import re
import sqlite3
import time
from typing import TypedDict

import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "tennis.db")
VALID_SURFACES = {"Hard", "Clay", "Grass"}
EDITION_GAP_DAYS = 60

MATCH_COLUMNS = {
    "match_id": "match_id", "Player 1": "player1", "Player 2": "player2",
    "Pl 1 hand": "player1_hand", "Pl 2 hand": "player2_hand", "Date": "date",
    "Tournament": "tournament", "Round": "round", "Time": "time", "Court": "court",
    "Surface": "surface", "Umpire": "umpire", "Best of": "best_of",
    "Final TB?": "final_tb", "Charted by": "charted_by",
}

POINT_COLUMNS = {
    "match_id": "match_id", "Pt": "pt", "Set1": "set1", "Set2": "set2",
    "Gm1": "gm1", "Gm2": "gm2", "Pts": "pts", "Svr": "svr", "1st": "first_serve",
    "2nd": "second_serve", "Notes": "notes", "PtWinner": "pt_winner",
}
POINT_INT_COLUMNS = ["pt", "set1", "set2", "gm1", "gm2", "svr", "pt_winner"]

SCHEMA = """
CREATE TABLE players (
    player_id     INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    hand          TEXT CHECK (hand IN ('R', 'L'))   -- most common value across the player's matches
);

CREATE TABLE tournaments (            -- one row per edition: one tour's event in one period
    tournament_id INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    tour          TEXT NOT NULL CHECK (tour IN ('M', 'W')),
    season        INTEGER NOT NULL,   -- year of end_date
    start_date    TEXT NOT NULL,      -- 'YYYY-MM-DD' of the first charted match
    end_date      TEXT NOT NULL,      -- 'YYYY-MM-DD' of the last charted match
    surface       TEXT CHECK (surface IN ('Hard', 'Clay', 'Grass')),
    UNIQUE (name, tour, start_date)
);

CREATE TABLE matches (
    match_id      TEXT PRIMARY KEY,   -- 'YYYYMMDD-G-Tournament-Round-Player1-Player2'
    tournament_id INTEGER NOT NULL REFERENCES tournaments (tournament_id),
    player1_id    INTEGER NOT NULL REFERENCES players (player_id),   -- served first
    player2_id    INTEGER NOT NULL REFERENCES players (player_id),
    date          TEXT NOT NULL,      -- 'YYYY-MM-DD'
    round         TEXT,
    time          TEXT,
    court         TEXT,
    umpire        TEXT,
    best_of       INTEGER CHECK (best_of IN (3, 5)),
    final_tb      TEXT,               -- final-set tiebreak rule code
    charted_by    TEXT,
    CHECK (player1_id <> player2_id)
);
CREATE INDEX idx_matches_tournament ON matches (tournament_id);
CREATE INDEX idx_matches_player1 ON matches (player1_id);
CREATE INDEX idx_matches_player2 ON matches (player2_id);

CREATE TABLE points (
    match_id      TEXT NOT NULL REFERENCES matches (match_id),
    pt            INTEGER NOT NULL,   -- point number within the match, from 1
    set1          INTEGER,            -- sets won so far by player1 / player2
    set2          INTEGER,
    gm1           INTEGER,            -- games won so far in this set
    gm2           INTEGER,
    pts           TEXT,               -- game score, server's points first
    svr           INTEGER CHECK (svr IN (1, 2)),         -- 1 = player1 serving
    first_serve   TEXT,               -- MCP shot code for the 1st-serve point
    second_serve  TEXT,               -- MCP shot code for the 2nd serve, if any
    notes         TEXT,
    pt_winner     INTEGER CHECK (pt_winner IN (1, 2)),   -- 1 = player1 won
    PRIMARY KEY (match_id, pt)
) WITHOUT ROWID;

CREATE VIEW points_detail AS
SELECT p.*,
       p.set1 + p.set2 + 1 AS set_num,
       DENSE_RANK() OVER (PARTITION BY p.match_id
                          ORDER BY p.set1 + p.set2, p.gm1 + p.gm2) AS game_num,
       CASE WHEN m.final_tb = 'V' THEN 0
            WHEN p.set1 + p.set2 + 1 < m.best_of THEN 1
            WHEN m.final_tb IS NULL OR m.final_tb IN ('0', 'W') THEN 0
            ELSE 1 END AS set_has_tb
FROM points AS p
JOIN matches AS m USING (match_id);

CREATE TABLE source_files (
    file_name     TEXT PRIMARY KEY,
    table_name    TEXT NOT NULL,
    rows_read     INTEGER NOT NULL,
    loaded_at     TEXT NOT NULL
);
"""

# Stats tables. Each row covers one player (or server) in one match and one
# category; NULL in a nullable category column means "all".
#   key:      'player_id' or 'server_id' (both reference players)
#   dims:     (column, declaration, nullable)
#   measures: INTEGER counts, or REAL when the name ends in _pct
class StatsSpec(TypedDict):
    doc: str
    key: str
    dims: list[tuple[str, str, bool]]
    measures: list[str]


SHOT_DIRECTIONS = {"XC": "crosscourt", "DTM": "down_middle", "DTL": "down_the_line",
                   "IO": "inside_out", "II": "inside_in"}
STATS: dict[str, StatsSpec] = {
    "serve_stats": StatsSpec(
        doc="serve outcomes by serve, court side and direction",
        key="player_id",
        dims=[("serve_no", "INTEGER NOT NULL CHECK (serve_no IN (1, 2))", False),
              ("side", "TEXT CHECK (side IN ('deuce', 'ad'))", True),
              ("direction", "TEXT CHECK (direction IN ('wide', 'body', 't'))", True)],
        measures=["pts", "pts_won", "aces", "unret", "forced_err", "won_lte_3_shots", "dfs"]),
    "serve_direction": StatsSpec(
        doc="serves by side and direction, and faults by type",
        key="player_id",
        dims=[("serve_no", "INTEGER NOT NULL CHECK (serve_no IN (1, 2))", False)],
        measures=["deuce_wide", "deuce_middle", "deuce_t", "ad_wide", "ad_middle", "ad_t",
                  "err_net", "err_wide", "err_deep", "err_wide_deep", "err_foot", "err_unknown"]),
    "serve_influence": StatsSpec(
        doc="% of serve points won once the rally reaches N shots",
        key="player_id",
        dims=[("serve_no", "INTEGER NOT NULL CHECK (serve_no IN (1, 2))", False)],
        measures=["pts"] + [f"won_{n}plus_pct" for n in range(1, 11)]),
    "key_points": StatsSpec(
        doc="break points, game points and deuce points, from the server's side",
        key="server_id",
        dims=[("situation", "TEXT NOT NULL CHECK (situation IN ('BP', 'GP', 'Deuce', 'Total'))", False)],
        measures=["pts", "pts_won", "first_in", "aces", "svc_winners", "rally_winners", "rally_forced",
                  "unforced", "dfs", "ret_rally_winners", "ret_rally_forced", "ret_unforced"]),
    "return_stats": StatsSpec(
        doc="return outcomes and depth by category (v1st, v2nd, fh, bh, D, A, 4/5/6, ...)",
        key="player_id",
        dims=[("category", "TEXT NOT NULL", False)],
        measures=["pts", "pts_won", "returnable", "returnable_won", "in_play", "in_play_won",
                  "winners", "total_shots", "shallow", "deep", "very_deep", "unforced",
                  "err_net", "err_deep", "err_wide", "err_wide_deep"]),
    "shot_stats": StatsSpec(
        doc="shot outcomes by shot type, and by shot and direction",
        key="player_id",
        dims=[("shot", "TEXT NOT NULL", False),
              ("direction", "TEXT CHECK (direction IN ("
               + ", ".join(f"'{d}'" for d in SHOT_DIRECTIONS.values()) + "))", True)],
        measures=["shots", "pt_ending", "winners", "induced_forced", "unforced", "serve_return",
                  "shots_in_pts_won", "shots_in_pts_lost"]),
    "net_stats": StatsSpec(
        doc="net points, approaches and serve-and-volley points",
        key="player_id",
        dims=[("category", "TEXT NOT NULL", False)],
        measures=["pts", "pts_won", "net_winner", "induced_forced", "net_unforced", "passed_at_net",
                  "passing_shot_induced_forced", "total_shots", "aces", "unret", "return_forced"]),
    "rally_stats": StatsSpec(
        doc="points by rally length, from the server's side",
        key="server_id",
        dims=[("rally_len", "TEXT NOT NULL CHECK (rally_len IN ('1-3', '4-6', '7-9', '10+'))", False)],
        measures=["server_won", "server_winners", "server_forced", "server_unforced",
                  "returner_won", "returner_winners", "returner_forced", "returner_unforced"]),
    "set_stats": StatsSpec(
        doc="match overview per set ('1'-'5') and for the whole match ('Total')",
        key="player_id",
        dims=[("set_label", "TEXT NOT NULL", False)],
        measures=["serve_pts", "aces", "dfs", "first_in", "first_won", "second_in", "second_won",
                  "bk_pts", "bp_saved", "return_pts", "return_pts_won", "winners", "winners_fh",
                  "winners_bh", "unforced", "unforced_fh", "unforced_bh"]),
}


def stats_schema(name: str, spec: StatsSpec) -> str:
    cols = ["id INTEGER PRIMARY KEY",
            "match_id TEXT NOT NULL REFERENCES matches (match_id)",
            f"{spec['key']} INTEGER NOT NULL REFERENCES players (player_id)"]
    cols += [f"{col} {decl}" for col, decl, _ in spec["dims"]]
    cols += [f"{m} {'REAL' if m.endswith('_pct') else 'INTEGER'}" for m in spec["measures"]]
    unique = ["match_id", spec["key"]] + [f"COALESCE({col}, '')" if nullable else col
                                          for col, _, nullable in spec["dims"]]
    return (f"CREATE TABLE {name} (   -- {spec['doc']}\n    " + ",\n    ".join(cols) + "\n);\n"
            f"CREATE UNIQUE INDEX ux_{name} ON {name} ({', '.join(unique)});\n")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def log(msg: str):
    print(f"    {msg}")


def to_int(df: pd.DataFrame, cols: list):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def map_codes(series: pd.Series, mapping: dict, what: str):
    """Map source codes to values, failing loudly on a code the build doesn't know."""
    unknown = sorted(set(series.dropna()) - set(mapping))
    if unknown:
        raise ValueError(f"unknown {what} codes: {unknown}")
    return series.map(mapping)


def normalise_name(name: str):
    """'Australian_Open' and "Queen's Club" -> 'australianopen', 'queensclub'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def most_common(values: pd.Series):
    counts = values.dropna().value_counts()
    return counts.index[0] if len(counts) else None


class Sources:
    """Collects source_files rows as CSVs are read."""

    def __init__(self):
        self.rows = []

    def read(self, path: str, table: str, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(path, dtype=str, **kwargs)
        df = df.apply(lambda s: s.str.strip())
        self.rows.append((os.path.basename(path), table, len(df),
                          dt.datetime.now().isoformat(timespec="seconds")))
        return df

    def save(self, con: sqlite3.Connection):
        con.executemany("INSERT INTO source_files VALUES (?, ?, ?, ?)", self.rows)


# ----------------------------------------------------------------------------
# Matches, players, tournaments
# ----------------------------------------------------------------------------

def players_from_match_id(match_id: str):
    """'20221110-W-BJK_Cup_Finals-RR-Martina_Trevisan-X_Y' -> ('Martina Trevisan', 'X Y')."""
    parts = match_id.split("-")
    if len(parts) != 6:
        return None, None
    return parts[4].replace("_", " "), parts[5].replace("_", " ")


def shift_right(df: pd.DataFrame, row: Hashable, cols: list[str], by: int) -> None:
    """Move row's values in cols `by` places to the right, leaving the first `by` empty."""
    values = [df.at[row, c] for c in cols]
    for col, value in zip(cols, [None] * by + values[:-by]):
        df.at[row, col] = value


def repair_matches(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fix the known malformed rows in the -matches- files.

    Three patterns occur, all in Davis Cup / BJK Cup ties:
      1. Both player-name fields are missing, so every later value sits two
         columns to the left (player1_hand holds the date).
      2. The surface is missing, so the umpire's name sits in 'surface' and the
         later values are one column to the left (sometimes on top of 1.).
      3. Most rows with pattern 1 have a second "stub" row with the same
         match_id holding the names, surface and charter but little else.
    Returns the repaired frame and a list of human-readable notes.
    """
    df = df.copy()
    notes = []
    fields = [c for c in MATCH_COLUMNS.values() if c != "match_id"]  # in CSV order

    shifted = df["player1_hand"].str.fullmatch(r"\d{8}", na=False)
    for i in df.index[shifted]:
        shift_right(df, i, fields, 2)
        notes.append(f"{df.at[i, 'match_id']}: player names missing; shifted 2 columns right")

    surface_missing = ~df["surface"].isin(VALID_SURFACES) & df["umpire"].isin(["1", "3", "5"])
    tail = ["surface", "umpire", "best_of", "final_tb", "charted_by"]
    for i in df.index[surface_missing]:
        shift_right(df, i, tail, 1)
        notes.append(f"{df.at[i, 'match_id']}: umpire was in surface column; shifted 1 column right")

    # Merge each duplicated match_id into one row: keep the first row's values
    # and fill its gaps from the stub row.
    dup = df["match_id"].duplicated(keep=False)
    if dup.any():
        merged = df[dup].groupby("match_id", sort=False).first().reset_index()
        for match_id in merged["match_id"]:
            notes.append(f"{match_id}: merged duplicate stub row into the main row")
        df = pd.concat([df[~dup], merged[df.columns]], ignore_index=True)

    for i in df.index[df["player1"].isna() | df["player2"].isna()]:
        df.loc[i, ["player1", "player2"]] = players_from_match_id(str(df.at[i, "match_id"]))
        notes.append(f"{df.at[i, 'match_id']}: player names parsed from match_id")

    return df, notes


def read_matches(sources: Sources) -> pd.DataFrame:
    """Both -matches- files, repaired, one row per match_id, with a parsed date."""
    frames = []
    for tour in ("m", "w"):
        path = os.path.join(DATA_DIR, f"charting-{tour}-matches.csv")
        df = sources.read(path, "matches, players, tournaments")
        df = df.rename(columns=lambda c: MATCH_COLUMNS[c.strip()])
        df.insert(1, "tour", tour.upper())
        frames.append(df)
    matches, notes = repair_matches(pd.concat(frames, ignore_index=True))
    for note in notes:
        log(f"repaired {note}")

    dupes = matches["match_id"][matches["match_id"].duplicated()]
    if len(dupes):
        raise ValueError(f"duplicate match_id values remain after repair: {dupes.tolist()}")

    # The charter-entered date always agrees with the match_id date when present,
    # so the match_id date (never missing) is used.
    matches["match_date"] = pd.to_datetime(matches["match_id"].str[:8], format="%Y%m%d")
    disagree = matches["date"].notna() & (matches["date"] != matches["match_id"].str[:8])
    if disagree.any():
        log(f"{disagree.sum()} matches have a Date column that differs from match_id; using match_id")

    matches["surface"] = matches["surface"].where(matches["surface"].isin(VALID_SURFACES))
    best_of = pd.to_numeric(matches["best_of"], errors="coerce")
    matches["best_of"] = best_of.where(best_of.isin([3, 5])).astype("Int64")
    return matches


def build_players(matches: pd.DataFrame):
    apps = pd.concat([
        matches[["player1", "player1_hand"]].set_axis(["name", "hand"], axis=1),
        matches[["player2", "player2_hand"]].set_axis(["name", "hand"], axis=1)])
    apps["hand"] = apps["hand"].where(apps["hand"].isin(["R", "L"]))
    players = apps.groupby("name")["hand"].agg(most_common).reset_index().sort_values("name")
    players.insert(0, "player_id", range(1, len(players) + 1))
    conflicts = (apps.dropna().groupby("name")["hand"].nunique() > 1).sum()
    log(f"{len(players):,} players; {conflicts} with both R and L recorded use the most common")
    return players


def build_tournaments(matches: pd.DataFrame):
    """Return (tournaments table, tournament_id for each row of matches)."""
    name = matches["tournament"].fillna(
        matches["match_id"].str.split("-").str[2].str.replace("_", " "))
    key = name.map(normalise_name)
    # One spelling per normalised name: the most common one.
    spelling = name.groupby(key).agg(most_common)
    n_variants = (name.groupby(key).nunique() > 1).sum()

    m = pd.DataFrame({"key": key, "tour": matches["tour"], "date": matches["match_date"],
                      "surface": matches["surface"]}).sort_values(["key", "tour", "date"])
    gap = m.groupby(["key", "tour"])["date"].diff().dt.days
    m["edition"] = (gap.isna() | (gap > EDITION_GAP_DAYS)).cumsum()

    ed = m.groupby("edition").agg(key=("key", "first"), tour=("tour", "first"),
                                  start=("date", "min"), end=("date", "max"),
                                  surface=("surface", most_common),
                                  n_surfaces=("surface", "nunique"))
    ed = ed.sort_values(["start", "key", "tour"])
    ed["tournament_id"] = range(1, len(ed) + 1)
    tournaments = pd.DataFrame({
        "tournament_id": ed["tournament_id"],
        "name": ed["key"].map(spelling),
        "tour": ed["tour"],
        "season": ed["end"].dt.year,
        "start_date": ed["start"].dt.strftime("%Y-%m-%d"),
        "end_date": ed["end"].dt.strftime("%Y-%m-%d"),
        "surface": ed["surface"],
    })

    tournament_id = m["edition"].map(ed["tournament_id"]).reindex(matches.index)
    differs = (matches["surface"].notna()
               & (matches["surface"] != tournament_id.map(tournaments.set_index("tournament_id")["surface"])))
    log(f"{len(tournaments):,} tournament editions from {key.nunique()} names "
        f"({n_variants} names had more than one spelling)")
    log(f"{(ed['n_surfaces'] > 1).sum()} editions recorded more than one surface; "
        f"{differs.sum()} matches take their edition's most common surface")
    return tournaments, tournament_id


def load_matches(con: sqlite3.Connection, sources: Sources) -> tuple[pd.Series, pd.DataFrame]:
    matches = read_matches(sources)
    players = build_players(matches)
    tournaments, tournament_id = build_tournaments(matches)
    player_id = players.set_index("name")["player_id"]

    table = pd.DataFrame({
        "match_id": matches["match_id"],
        "tournament_id": tournament_id,
        "player1_id": matches["player1"].map(player_id),
        "player2_id": matches["player2"].map(player_id),
        "date": matches["match_date"].dt.strftime("%Y-%m-%d"),
        **{c: matches[c] for c in ("round", "time", "court", "umpire", "best_of",
                                   "final_tb", "charted_by")},
    })
    players.to_sql("players", con, if_exists="append", index=False)
    tournaments.to_sql("tournaments", con, if_exists="append", index=False)
    table.to_sql("matches", con, if_exists="append", index=False)
    log(f"{len(table):,} matches")
    # match_id -> (player1 name, player2 name), used to resolve stats rows
    return player_id, matches.set_index("match_id")[["player1", "player2"]]


# ----------------------------------------------------------------------------
# Points
# ----------------------------------------------------------------------------

def load_points(con, sources):
    total = 0
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "charting-?-points-*.csv"))):
        df = sources.read(path, "points")
        df = df.rename(columns=POINT_COLUMNS)[list(POINT_COLUMNS.values())]
        df = to_int(df, POINT_INT_COLUMNS)

        # A few matches were charted twice, the copies back to back: keep the first.
        dup = df.duplicated(["match_id", "pt"], keep=False)
        for match_id, rows in df[dup].groupby("match_id"):
            differing = (rows.drop_duplicates().groupby("pt").size() > 1).sum()
            kind = "identical copy" if differing == 0 else f"second chart differs on {differing} points"
            log(f"{match_id}: charted twice ({kind}); kept the first")
        df = df.drop_duplicates(["match_id", "pt"], keep="first")

        # Fill a missing game score from the next point (in each case so far, same game).
        df = df.sort_values(["match_id", "pt"])
        for col in ("gm1", "gm2"):
            missing = df[col].isna()
            if missing.any():
                df[col] = df[col].fillna(df.groupby("match_id")[col].bfill())
                for match_id, pt in df.loc[missing, ["match_id", "pt"]].values:
                    log(f"{match_id} point {pt}: {col} missing; taken from the next point")

        df.to_sql("points", con, if_exists="append", index=False, chunksize=100_000)
        log(f"{os.path.basename(path)}: {len(df):,} points")
        total += len(df)
    return total


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------

def read_stats(sources: Sources, name: str, table: str):
    """Both tours' charting-*-stats-<name>.csv, as one frame."""
    return pd.concat([sources.read(os.path.join(DATA_DIR, f"charting-{t}-stats-{name}.csv"), table)
                      for t in ("m", "w")], ignore_index=True)


def dedupe(df: pd.DataFrame, key: list[str], what: str):
    """Drop exact duplicate rows, then keep the first row for each natural key."""
    exact = len(df)
    df = df.drop_duplicates()
    exact -= len(df)
    conflicting = len(df)
    df = df.drop_duplicates(key, keep="first")
    conflicting -= len(df)
    if exact or conflicting:
        log(f"{what}: dropped {exact} duplicate rows and {conflicting} conflicting repeats (kept first)")
    return df


def players_of(df: pd.DataFrame, match_players: pd.DataFrame) -> pd.DataFrame:
    """player1 and player2 of each row's match, aligned to df's index."""
    return match_players.reindex(df["match_id"]).set_axis(df.index)


def opponent(df: pd.DataFrame, match_players: pd.DataFrame) -> pd.Series:
    p = players_of(df, match_players)
    return p["player1"].where(df["player"] != p["player1"], p["player2"])


SERVE_ROWS = {"d": ("deuce", None), "a": ("ad", None), "4": (None, "wide"), "5": (None, "body"),
              "6": (None, "t"), "4d": ("deuce", "wide"), "4a": ("ad", "wide"), "5d": ("deuce", "body"),
              "5a": ("ad", "body"), "6d": ("deuce", "t"), "6a": ("ad", "t")}


def serve_stats(sources: Sources) -> pd.DataFrame:
    split = dedupe(read_stats(sources, "SvBreakSplit", "serve_stats"), ["match_id", "player", "row"],
                   "SvBreakSplit")
    total = dedupe(read_stats(sources, "SvBreakTotal", "serve_stats"), ["match_id", "player", "row"],
                   "SvBreakTotal")
    split = split.merge(total[["match_id", "player", "row", "dfs"]], how="left")
    dims = map_codes(split["row"], SERVE_ROWS, "SvBreakSplit row")
    parts = []
    for serve_no, prefix in ((1, "first"), (2, "second")):
        part = pd.DataFrame({
            "match_id": split["match_id"], "player": split["player"], "serve_no": serve_no,
            "side": dims.str[0], "direction": dims.str[1],
            "pts": split[f"{prefix}_pts"], "pts_won": split[f"{prefix}_pts_won"],
            "aces": split[f"{prefix}_aces"], "unret": split[f"{prefix}_unret"],
            "forced_err": split[f"{prefix}_forced"],
            "won_lte_3_shots": split[f"{prefix}_won_lte_3_shots"],
            "dfs": split["dfs"] if serve_no == 2 else None})
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def serve_direction(sources: Sources) -> pd.DataFrame:
    df = read_stats(sources, "ServeDirection", "serve_direction")
    df = df[df["row"] != "Total"].copy()  # 1st + 2nd
    df["serve_no"] = map_codes(df["row"], {"1": 1, "2": 2}, "ServeDirection row")
    return dedupe(df.drop(columns="row"), ["match_id", "player", "serve_no"], "ServeDirection")


def serve_influence(sources: Sources) -> pd.DataFrame:
    df = read_stats(sources, "ServeInfluence", "serve_influence")
    df["serve_no"] = map_codes(df["row"], {"1": 1, "2": 2}, "ServeInfluence row")
    for n in range(1, 11):  # '68.8%' -> 68.8, '-' -> NULL
        df[f"won_{n}plus_pct"] = pd.to_numeric(df.pop(f"won_{n}+").str.rstrip("%"), errors="coerce")
    return dedupe(df.drop(columns="row"), ["match_id", "player", "serve_no"], "ServeInfluence")


def key_points(sources: Sources, match_players: pd.DataFrame) -> pd.DataFrame:
    serve = read_stats(sources, "KeyPointsServe", "key_points")
    serve["situation"] = map_codes(serve["row"], {"BP": "BP", "GP": "GP", "Deuce": "Deuce",
                                                  "STotal": "Total"}, "KeyPointsServe row")
    serve = dedupe(serve.drop(columns="row"), ["match_id", "player", "situation"], "KeyPointsServe")
    # A returner's row is the opponent's serve row seen from the other side; only
    # the returner's own winners and errors are new.
    ret = read_stats(sources, "KeyPointsReturn", "key_points")
    ret["situation"] = map_codes(ret["row"], {"BPO": "BP", "GPF": "GP", "DeuceR": "Deuce",
                                              "RTotal": "Total"}, "KeyPointsReturn row")
    ret = dedupe(ret, ["match_id", "player", "situation"], "KeyPointsReturn")
    ret = pd.DataFrame({"match_id": ret["match_id"], "player": opponent(ret, match_players),
                        "situation": ret["situation"], "ret_rally_winners": ret["rally_winners"],
                        "ret_rally_forced": ret["rally_forced"], "ret_unforced": ret["unforced"]})
    merged = serve.merge(ret, how="left", on=["match_id", "player", "situation"], indicator=True)
    unmatched = len(ret) - (merged["_merge"] == "both").sum()
    if unmatched:
        log(f"KeyPointsReturn: {unmatched} rows have no matching serve row and were dropped")
    return merged.drop(columns="_merge")


def return_stats(sources: Sources) -> pd.DataFrame:
    key = ["match_id", "player", "row"]
    outcomes = dedupe(read_stats(sources, "ReturnOutcomes", "return_stats"), key, "ReturnOutcomes")
    depth = dedupe(read_stats(sources, "ReturnDepth", "return_stats"), key, "ReturnDepth")
    df = outcomes.merge(depth, how="outer", on=key, suffixes=("", "_depth"))
    df["returnable"] = df["returnable"].fillna(df.pop("returnable_depth"))
    return df.rename(columns={"row": "category"})


def shot_stats(sources: Sources) -> pd.DataFrame:
    key = ["match_id", "player", "row"]
    types = dedupe(read_stats(sources, "ShotTypes", "shot_stats"), key, "ShotTypes")
    types = types.rename(columns={"row": "shot"}).assign(direction=None)
    dirs = dedupe(read_stats(sources, "ShotDirOutcomes", "shot_stats"), key, "ShotDirOutcomes")
    shot_dir = dirs.pop("row").str.split("-", n=1, expand=True)
    dirs["shot"] = shot_dir[0]
    dirs["direction"] = map_codes(shot_dir[1], SHOT_DIRECTIONS, "ShotDirOutcomes direction")
    return pd.concat([types, dirs], ignore_index=True)


def net_stats(sources: Sources) -> pd.DataFrame:
    key = ["match_id", "player", "row"]
    net = dedupe(read_stats(sources, "NetPoints", "net_stats"), key, "NetPoints")
    snv = dedupe(read_stats(sources, "SnV", "net_stats"), key, "SnV")
    snv = snv[~snv["row"].isin(["SnV", "nonSnV"])]  # 1st + 2nd serve rows
    df = pd.concat([net.rename(columns={"net_pts": "pts"}), snv.rename(columns={"snv_pts": "pts"})],
                   ignore_index=True)
    return df.rename(columns={"row": "category"})


def rally_stats(sources: Sources) -> pd.DataFrame:
    df = dedupe(read_stats(sources, "Rally", "rally_stats"), ["match_id", "row"], "Rally")
    # Rows ending -1 / -2 cover the points player1 / player2 served; on those rows
    # 'server' names that player and pl1_* are their numbers, pl2_* the returner's.
    # Rows without the suffix add both servers together and are dropped.
    df = df[df["row"].str.fullmatch(r".+-[12]")]
    out = pd.DataFrame({
        "match_id": df["match_id"],
        "player": df["server"],
        "rally_len": map_codes(df["row"].str[:-2], {"1-3": "1-3", "4-6": "4-6", "7-9": "7-9", "10": "10+"},
                               "Rally row")})
    for m in ("won", "winners", "forced", "unforced"):
        out[f"server_{m}"] = df[f"pl1_{m}"]
        out[f"returner_{m}"] = df[f"pl2_{m}"]
    return out


def set_stats(sources: Sources) -> pd.DataFrame:
    df = read_stats(sources, "Overview", "set_stats").rename(columns={"set": "set_label"})
    return dedupe(df, ["match_id", "player", "set_label"], "Overview")


# key_points also needs each match's players; load_stats binds them in.
STATS_LOADERS: dict[str, Callable[[Sources], pd.DataFrame]] = {
    "serve_stats": serve_stats, "serve_direction": serve_direction,
    "serve_influence": serve_influence, "return_stats": return_stats,
    "shot_stats": shot_stats, "net_stats": net_stats, "rally_stats": rally_stats,
    "set_stats": set_stats,
}


def load_stats(con: sqlite3.Connection, sources: Sources, player_id: pd.Series, match_players: pd.DataFrame):
    loaders = {**STATS_LOADERS, "key_points": partial(key_points, match_players=match_players)}
    for name, spec in STATS.items():
        df = loaders[name](sources)
        p = players_of(df, match_players)
        in_match = (df["player"] == p["player1"]) | (df["player"] == p["player2"])
        for match_id, player in df.loc[~in_match, ["match_id", "player"]].values:
            log(f"{name}: dropped a row for '{player}', who is not a player in {match_id}")
        df = df[in_match]
        ids: pd.Series = df["player"].map(player_id)
        if ids.isna().any():
            raise ValueError(f"{name}: unknown players {sorted(df.loc[ids.isna(), 'player'].unique())[:5]}")
        df[spec["key"]] = ids
        cols = ["match_id", spec["key"]] + [c for c, _, _ in spec["dims"]] + spec["measures"]
        df = df[cols].copy()
        df = to_int(df, [c for c in spec["measures"] if not c.endswith("_pct")])
        if any(c == "serve_no" for c, _, _ in spec["dims"]):
            df = to_int(df, ["serve_no"])
        df.to_sql(name, con, if_exists="append", index=False, chunksize=100_000)
        log(f"{name}: {len(df):,} rows")


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------

def source_csvs():
    return sorted(glob.glob(os.path.join(DATA_DIR, "charting-*.csv")))


def build_database(db_path=DEFAULT_DB_PATH):
    """Create db_path from the CSVs, replacing any existing file."""
    start = time.time()
    print(f"Building {db_path} from {len(source_csvs())} CSV files ...")
    tmp_path = db_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    con = sqlite3.connect(tmp_path)
    try:
        con.execute("PRAGMA journal_mode = OFF")
        con.execute("PRAGMA synchronous = OFF")
        con.executescript(SCHEMA + "".join(stats_schema(n, s) for n, s in STATS.items()))
        sources = Sources()

        print("  matches, players, tournaments")
        player_id, match_players = load_matches(con, sources)
        print("  points")
        load_points(con, sources)
        print("  stats")
        load_stats(con, sources, player_id, match_players)
        sources.save(con)

        orphans = con.execute("PRAGMA foreign_key_check").fetchall()
        if orphans:
            raise ValueError(f"{len(orphans)} rows reference a missing parent row, e.g. {orphans[:3]}")
        con.execute("ANALYZE")
        con.commit()
    finally:
        con.close()

    os.replace(tmp_path, db_path)
    print(f"Done in {time.time() - start:.0f}s: {os.path.getsize(db_path) / 1e6:,.0f} MB")


def connect(db_path=DEFAULT_DB_PATH, rebuild=False):
    """Open the database, building it first if it is missing (or rebuild=True)."""
    if rebuild or not os.path.exists(db_path):
        build_database(db_path)
    else:
        db_mtime = os.path.getmtime(db_path)
        stale = [os.path.basename(p) for p in source_csvs() if os.path.getmtime(p) > db_mtime]
        if stale:
            print(f"Warning: {len(stale)} CSV file(s) changed since {os.path.basename(db_path)} "
                  "was built; run `python tennis_db.py` to rebuild it.")
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=DEFAULT_DB_PATH,
                        help="path of the SQLite file to create (default: ./tennis.db)")
    args = parser.parse_args()
    build_database(args.db)


if __name__ == "__main__":
    main()
