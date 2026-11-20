
"""
Football Intelligence AI
========================

Motion Feature Engineering Module

Extracts player kinematic features from tracking data.

Features:
    - Raw velocity and acceleration
    - Savitzky-Golay smoothed position
    - Smoothed velocity and acceleration
    - Movement direction
    - Temporal segment quality indicators
    - SG boundary support indicators

The original tracking coordinates are preserved.

Note:
    Centered Savitzky-Golay filtering is intended for
    offline analysis. It uses observations on both
    sides of a timestamp and is not directly causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter


def add_motion_features(
    tracking_df: pd.DataFrame,
    fps: float = 10.0,
    window_length: int = 5,
    polyorder: int = 2,
) -> pd.DataFrame:
    """
    Extract raw and smoothed motion features.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Required columns:
        period, frame, player_id, x, y.

    fps : float
        Tracking frequency in frames per second.

    window_length : int
        Odd Savitzky-Golay window length.

    polyorder : int
        Polynomial order.

    Returns
    -------
    pd.DataFrame
        Original tracking data with motion features
        and quality indicators.
    """

    # ------------------------------------------
    # 1. Validate input
    # ------------------------------------------

    required = {
        "period",
        "frame",
        "player_id",
        "x",
        "y",
    }

    missing = required - set(tracking_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if fps <= 0:
        raise ValueError("fps must be positive.")

    if window_length < 3 or window_length % 2 == 0:
        raise ValueError(
            "window_length must be an odd integer >= 3."
        )

    if not 0 <= polyorder < window_length:
        raise ValueError(
            "polyorder must be smaller than window_length."
        )

    df = tracking_df.copy()

    # ------------------------------------------
    # 2. Validate and normalize data
    # ------------------------------------------

    if df[list(required)].isna().any().any():
        raise ValueError(
            "Missing values found in required columns."
        )

    df["frame"] = pd.to_numeric(
        df["frame"], errors="raise"
    )

    df["x"] = pd.to_numeric(
        df["x"], errors="raise"
    )

    df["y"] = pd.to_numeric(
        df["y"], errors="raise"
    )

    coords = df[["x", "y"]].to_numpy(dtype=float)

    if not np.isfinite(coords).all():
        raise ValueError(
            "Non-finite tracking coordinates found."
        )

    group_cols = ["period", "player_id"]

    df = df.sort_values(
        group_cols + ["frame"]
    ).reset_index(drop=True)

    if df.duplicated(
        subset=group_cols + ["frame"]
    ).any():

        raise ValueError(
            "Duplicate player observations detected."
        )

    dt = 1.0 / fps
    half_window = window_length // 2

    # ------------------------------------------
    # 3. Identify continuous trajectory segments
    # ------------------------------------------

    grouped = df.groupby(
        group_cols,
        sort=False,
        dropna=False
    )

    df["frame_gap"] = grouped["frame"].diff()

    df["segment_id"] = (
        df.groupby(
            group_cols,
            sort=False,
            dropna=False
        )["frame"]
        .transform(
            lambda s: s.diff().ne(1).cumsum()
        )
        .astype(int)
    )

    segments = df.groupby(
        group_cols + ["segment_id"],
        sort=False,
        dropna=False
    )

    df["segment_position"] = segments.cumcount()

    df["segment_length"] = segments["frame"].transform(
        "size"
    )

    valid_step = df["frame_gap"].eq(1)

    # ------------------------------------------
    # 4. Raw velocity
    # ------------------------------------------

    grouped = df.groupby(
        group_cols,
        sort=False,
        dropna=False
    )

    dx = grouped["x"].diff()
    dy = grouped["y"].diff()

    df["vx_raw_mps"] = (
        dx / dt
    ).where(valid_step)

    df["vy_raw_mps"] = (
        dy / dt
    ).where(valid_step)

    df["speed_raw_mps"] = np.hypot(
        df["vx_raw_mps"],
        df["vy_raw_mps"]
    )

    # ------------------------------------------
    # 5. Raw acceleration
    # ------------------------------------------

    previous_gap = (
        df.groupby(
            group_cols,
            sort=False,
            dropna=False
        )["frame_gap"].shift(1)
    )

    valid_acceleration = (
        valid_step & previous_gap.eq(1)
    )

    previous_vx = (
        df.groupby(
            group_cols,
            sort=False,
            dropna=False
        )["vx_raw_mps"].shift(1)
    )

    previous_vy = (
        df.groupby(
            group_cols,
            sort=False,
            dropna=False
        )["vy_raw_mps"].shift(1)
    )

    df["ax_raw_mps2"] = (
        (df["vx_raw_mps"] - previous_vx) / dt
    ).where(valid_acceleration)

    df["ay_raw_mps2"] = (
        (df["vy_raw_mps"] - previous_vy) / dt
    ).where(valid_acceleration)

    df["acceleration_raw_mps2"] = np.hypot(
        df["ax_raw_mps2"],
        df["ay_raw_mps2"]
    )

    # ------------------------------------------
    # 6. Initialize SG feature columns
    # ------------------------------------------

    sg_columns = [
        "x_sg_m",
        "y_sg_m",
        "vx_sg_mps",
        "vy_sg_mps",
        "ax_sg_mps2",
        "ay_sg_mps2",
    ]

    for column in sg_columns:
        df[column] = np.nan

    # ------------------------------------------
    # 7. Apply SG within continuous segments
    # ------------------------------------------

    segment_groups = df.groupby(
        group_cols + ["segment_id"],
        sort=False,
        dropna=False
    )

    for indices in segment_groups.indices.values():

        indices = np.asarray(indices, dtype=int)

        if len(indices) < window_length:
            continue

        x = df.loc[indices, "x"].to_numpy(dtype=float)
        y = df.loc[indices, "y"].to_numpy(dtype=float)

        # Smoothed coordinates
        df.loc[indices, "x_sg_m"] = savgol_filter(
            x,
            window_length,
            polyorder,
            deriv=0,
            mode="interp"
        )

        df.loc[indices, "y_sg_m"] = savgol_filter(
            y,
            window_length,
            polyorder,
            deriv=0,
            mode="interp"
        )

        # Velocity
        df.loc[indices, "vx_sg_mps"] = savgol_filter(
            x,
            window_length,
            polyorder,
            deriv=1,
            delta=dt,
            mode="interp"
        )

        df.loc[indices, "vy_sg_mps"] = savgol_filter(
            y,
            window_length,
            polyorder,
            deriv=1,
            delta=dt,
            mode="interp"
        )

        # Acceleration
        df.loc[indices, "ax_sg_mps2"] = savgol_filter(
            x,
            window_length,
            polyorder,
            deriv=2,
            delta=dt,
            mode="interp"
        )

        df.loc[indices, "ay_sg_mps2"] = savgol_filter(
            y,
            window_length,
            polyorder,
            deriv=2,
            delta=dt,
            mode="interp"
        )

    # ------------------------------------------
    # 8. Calculate SG motion features
    # ------------------------------------------

    df["speed_sg_mps"] = np.hypot(
        df["vx_sg_mps"],
        df["vy_sg_mps"]
    )

    df["acceleration_sg_mps2"] = np.hypot(
        df["ax_sg_mps2"],
        df["ay_sg_mps2"]
    )

    df["movement_direction_deg"] = np.degrees(
        np.arctan2(
            df["vy_sg_mps"],
            df["vx_sg_mps"]
        )
    )

    df.loc[
        df["speed_sg_mps"].isna()
        | df["speed_sg_mps"].eq(0),
        "movement_direction_deg"
    ] = np.nan

    # ------------------------------------------
    # 9. SG quality indicators
    # ------------------------------------------

    df["sg_window_available"] = (
        df["segment_length"] >= window_length
    )

    df["sg_edge_estimate"] = (
        df["sg_window_available"]
        & (
            (df["segment_position"] < half_window)
            |
            (
                df["segment_position"]
                >= df["segment_length"] - half_window
            )
        )
    )

    df["sg_full_window_support"] = (
        df["sg_window_available"]
        & ~df["sg_edge_estimate"]
        & df["acceleration_sg_mps2"].notna()
    )

    df["motion_features_valid"] = (
        df["speed_sg_mps"].notna()
        & df["acceleration_sg_mps2"].notna()
    )

    return df
