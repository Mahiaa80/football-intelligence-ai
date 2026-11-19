
"""
Football Intelligence AI
Tracking Data Loader

This module reads SkillCorner JSONL tracking files
and converts player positions into a Pandas DataFrame.
"""

import json
from pathlib import Path
from typing import Iterator

import pandas as pd


def iter_tracking_frames(
    file_path: str | Path,
) -> Iterator[dict]:
    """
    Read tracking data frame by frame from a JSONL file.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Tracking file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                frame_data = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from error

            if not isinstance(frame_data, dict):
                raise ValueError(
                    f"Expected a dictionary at line {line_number}"
                )

            yield frame_data


def load_player_tracking(
    file_path: str | Path,
    max_frames: int | None = None,
) -> pd.DataFrame:
    """
    Load player tracking information into a DataFrame.

    Parameters
    ----------
    file_path:
        Path to the SkillCorner JSONL file.

    max_frames:
        Maximum number of frames to process.
        None means process all frames.

    Returns
    -------
    pd.DataFrame
        A table containing player positions,
        timestamps, identities and possession information.
    """

    if max_frames is not None and max_frames < 1:
        raise ValueError(
            "max_frames must be a positive integer."
        )

    rows = []
    processed_frames = 0

    for frame_data in iter_tracking_frames(file_path):

        if (
            max_frames is not None
            and processed_frames >= max_frames
        ):
            break

        processed_frames += 1

        possession = frame_data.get("possession") or {}

        if not isinstance(possession, dict):
            possession = {}

        players = frame_data.get("player_data") or []

        if not isinstance(players, list):
            raise ValueError(
                "player_data must be a list."
            )

        for player in players:

            if not isinstance(player, dict):
                raise ValueError(
                    "Each player entry must be a dictionary."
                )

            required_fields = [
                "player_id",
                "x",
                "y",
            ]

            missing = [
                key for key in required_fields
                if key not in player
            ]

            if missing:
                raise ValueError(
                    f"Missing player fields: {missing}"
                )

            rows.append({
                "frame": frame_data.get("frame"),
                "timestamp": frame_data.get("timestamp"),
                "period": frame_data.get("period"),
                "player_id": player["player_id"],
                "x": player["x"],
                "y": player["y"],
                "is_detected": player.get("is_detected"),
                "possession_player_id": possession.get("player_id"),
                "possession_group": possession.get("group"),
            })

    columns = [
        "frame",
        "timestamp",
        "period",
        "player_id",
        "x",
        "y",
        "is_detected",
        "possession_player_id",
        "possession_group",
    ]

    return pd.DataFrame(rows, columns=columns)
