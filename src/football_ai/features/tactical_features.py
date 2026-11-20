
"""Robust team-level tactical feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_team_tactical_features(
    tracking_df: pd.DataFrame,
    pitch_length: float,
    pitch_width: float,
    q_low: float = 0.10,
    q_high: float = 0.90,
    min_direct_players: int = 8,
) -> pd.DataFrame:
    """
    Build team-level tactical features for each frame.

    Original coordinates are preserved.

    Features include:
        - Raw team length and width
        - Quantile-based team length and width
        - Robust team center
        - Robust spatial spread
        - Detection rate
        - Outside-pitch count and rate
        - Direct-only quantile shape features

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Player tracking data.

    pitch_length : float
        Pitch length in meters.

    pitch_width : float
        Pitch width in meters.

    q_low : float
        Lower coordinate quantile.

    q_high : float
        Upper coordinate quantile.

    min_direct_players : int
        Minimum directly detected players required
        for direct-only tactical features.

    Returns
    -------
    pd.DataFrame
        One row per team, period and frame.
    """

    # 1. Validate input
    required = {
        "period",
        "frame",
        "player_id",
        "team_name",
        "x",
        "y",
        "is_detected",
    }

    missing = required - set(tracking_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if pitch_length <= 0 or pitch_width <= 0:
        raise ValueError("Pitch dimensions must be positive.")

    if not 0 <= q_low < q_high <= 1:
        raise ValueError("Invalid quantile range.")

    if min_direct_players < 1:
        raise ValueError(
            "min_direct_players must be positive."
        )

    df = tracking_df.copy()

    group_cols = [
        "period",
        "frame",
        "team_name",
    ]

    # 2. Validate player observations
    if df[list(required)].isna().any().any():
        raise ValueError(
            "Missing values in required tracking columns."
        )

    if df.duplicated(
        ["period", "frame", "player_id"]
    ).any():
        raise ValueError(
            "Duplicate player observations in a frame."
        )

    # 3. Coordinate quality indicators
    half_length = pitch_length / 2
    half_width = pitch_width / 2

    df["outside_x"] = (
        df["x"].abs() > half_length
    )

    df["outside_y"] = (
        df["y"].abs() > half_width
    )

    df["outside_pitch"] = (
        df["outside_x"] | df["outside_y"]
    )

    df["outside_distance_m"] = np.hypot(
        (df["x"].abs() - half_length).clip(lower=0),
        (df["y"].abs() - half_width).clip(lower=0)
    )

    df["direct_flag"] = (
        df["is_detected"].eq(True).astype(int)
    )

    # 4. Robust center per team and frame
    grouped = df.groupby(
        group_cols,
        sort=False
    )

    df["robust_center_x"] = (
        grouped["x"].transform("median")
    )

    df["robust_center_y"] = (
        grouped["y"].transform("median")
    )

    df["distance_to_robust_center"] = np.hypot(
        df["x"] - df["robust_center_x"],
        df["y"] - df["robust_center_y"]
    )

    # 5. All-tracking feature aggregation
    team = grouped.agg(
        n_players=("player_id", "nunique"),

        detected_players=("direct_flag", "sum"),

        outside_players=("outside_pitch", "sum"),

        mean_outside_distance_m=(
            "outside_distance_m", "mean"
        ),

        max_outside_distance_m=(
            "outside_distance_m", "max"
        ),

        min_x=("x", "min"),
        max_x=("x", "max"),

        min_y=("y", "min"),
        max_y=("y", "max"),

        q10_x=("x", lambda s: s.quantile(q_low)),
        q90_x=("x", lambda s: s.quantile(q_high)),

        q10_y=("y", lambda s: s.quantile(q_low)),
        q90_y=("y", lambda s: s.quantile(q_high)),

        robust_center_x=("x", "median"),
        robust_center_y=("y", "median"),

        robust_spread_m=(
            "distance_to_robust_center", "median"
        ),

        centroid_x=("x", "mean"),
        centroid_y=("y", "mean"),

        mean_squared_distance=(
            "distance_to_robust_center",
            lambda s: np.mean(s.to_numpy() ** 2)
        ),

        timestamp=("timestamp", "first")
        if "timestamp" in df.columns
        else ("frame", "first"),

    ).reset_index()

    # 6. Calculate raw and robust dimensions
    team["team_length_raw_m"] = (
        team["max_x"] - team["min_x"]
    )

    team["team_width_raw_m"] = (
        team["max_y"] - team["min_y"]
    )

    team["team_length_qrange_m"] = (
        team["q90_x"] - team["q10_x"]
    )

    team["team_width_qrange_m"] = (
        team["q90_y"] - team["q10_y"]
    )

    team["detection_rate"] = (
        team["detected_players"]
        / team["n_players"]
    )

    team["outside_rate"] = (
        team["outside_players"]
        / team["n_players"]
    )

    team["team_rms_spread_m"] = np.sqrt(
        team["mean_squared_distance"]
    )

    team["is_complete_team"] = (
        team["n_players"] == 11
    )

    # 7. Direct-only tactical measurements
    direct = df[
        df["is_detected"].eq(True)
    ].copy()

    direct_grouped = direct.groupby(
        group_cols,
        sort=False
    )

    direct_team = direct_grouped.agg(
        direct_n_players=("player_id", "nunique"),

        direct_q10_x=(
            "x", lambda s: s.quantile(q_low)
        ),

        direct_q90_x=(
            "x", lambda s: s.quantile(q_high)
        ),

        direct_q10_y=(
            "y", lambda s: s.quantile(q_low)
        ),

        direct_q90_y=(
            "y", lambda s: s.quantile(q_high)
        )

    ).reset_index()

    team = team.merge(
        direct_team,
        on=group_cols,
        how="left",
        validate="one_to_one"
    )

    team["direct_n_players"] = (
        team["direct_n_players"]
        .fillna(0)
        .astype(int)
    )

    team["direct_shape_available"] = (
        team["direct_n_players"]
        >= min_direct_players
    )

    team["direct_team_length_qrange_m"] = (
        team["direct_q90_x"]
        - team["direct_q10_x"]
    ).where(team["direct_shape_available"])

    team["direct_team_width_qrange_m"] = (
        team["direct_q90_y"]
        - team["direct_q10_y"]
    ).where(team["direct_shape_available"])

    team["direct_q10_x"] = team["direct_q10_x"].where(
        team["direct_shape_available"]
    )

    team["direct_q90_x"] = team["direct_q90_x"].where(
        team["direct_shape_available"]
    )

    team["direct_q10_y"] = team["direct_q10_y"].where(
        team["direct_shape_available"]
    )

    team["direct_q90_y"] = team["direct_q90_y"].where(
        team["direct_shape_available"]
    )

    # 8. Final ordering
    team = team.sort_values(
        ["period", "team_name", "frame"]
    ).reset_index(drop=True)

    return team
