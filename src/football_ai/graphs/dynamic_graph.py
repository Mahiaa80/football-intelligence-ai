"""
Dynamic player graph construction for football tracking data.
"""

import numpy as np
import pandas as pd


def build_frame_graph(
    frame_df,
    k_same_team=3,
    k_opponents=3,
    possession_group_to_team=None,
):
    """
    Build a directed spatial graph for one match frame.

    Each player connects to:
        - k_same_team nearest teammates
        - k_opponents nearest opponents

    Spatial relationships are calculated using SG-smoothed
    positions. Raw coordinates are retained in node features.
    """

    required = [
        "period", "frame", "player_id", "team_name",
        "x", "y", "x_sg_m", "y_sg_m",
        "vx_sg_mps", "vy_sg_mps",
        "speed_sg_mps",
        "ax_sg_mps2", "ay_sg_mps2",
        "acceleration_sg_mps2",
        "movement_direction_deg",
        "is_detected",
        "sg_edge_estimate",
        "sg_full_window_support",
        "possession_player_id",
        "possession_group",
    ]

    missing = set(required) - set(frame_df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = frame_df.copy()

    # Validate one frame
    if df["period"].nunique() != 1 or df["frame"].nunique() != 1:
        raise ValueError("Input must contain exactly one frame.")

    if len(df) != 22 or df["player_id"].nunique() != 22:
        raise ValueError("Expected exactly 22 unique players.")

    team_counts = df.groupby("team_name").size()

    if len(team_counts) != 2 or not (team_counts == 11).all():
        raise ValueError("Expected two teams with 11 players each.")

    # Stable node ordering
    df = df.sort_values(
        ["team_name", "player_id"]
    ).reset_index(drop=True)

    teams = sorted(df["team_name"].unique())
    team_to_index = {team: i for i, team in enumerate(teams)}

    df["team_index"] = df["team_name"].map(team_to_index)

    # Check frame-level possession information
    owner_ids = (
        df["possession_player_id"]
        .dropna()
        .unique()
    )

    possession_groups = (
        df["possession_group"]
        .dropna()
        .unique()
    )

    if len(owner_ids) > 1:
        raise ValueError("Multiple possession owners in one frame.")

    if len(possession_groups) > 1:
        raise ValueError("Multiple possession groups in one frame.")

        # Resolve possession group using external mapping

    possession_team = None

    available_teams = set(df["team_name"].unique())

    if possession_group is not None:

        if possession_group_to_team is None:
            raise ValueError(
                "possession_group_to_team mapping is required."
            )

        if possession_group not in possession_group_to_team:
            raise ValueError(
                f"Unknown possession group: {possession_group}"
            )

        possession_team = (
            possession_group_to_team[possession_group]
        )

        if possession_team not in available_teams:
            raise ValueError(
                "Mapped possession team is not present in this frame."
            )

    # Validate possession player

    if possession_player is not None:

        owner_rows = df[
            df["player_id"] == possession_player
        ]

        if len(owner_rows) != 1:
            raise ValueError(
                "Possession owner is not present exactly once."
            )

        owner_team = owner_rows["team_name"].iloc[0]

        if possession_team is None:
            possession_team = owner_team

        elif owner_team != possession_team:
            raise ValueError(
                "Possession owner team conflicts with possession group."
            )

    possession_player = (
        owner_ids[0] if len(owner_ids) == 1 else None
    )

    possession_group = (
        possession_groups[0]
        if len(possession_groups) == 1
        else None
    )

    if (
        possession_group is not None
        and possession_group not in possession_mapping
    ):
        raise ValueError("Unknown possession group.")

    possession_team = (
        possession_mapping[possession_group]
        if possession_group is not None
        else None
    )

    if (
        possession_player is not None
        and possession_player not in set(df["player_id"])
    ):
        raise ValueError("Possession owner is not in this frame.")

    # Node features
    df["is_possession_player"] = (
        df["player_id"].eq(possession_player).astype(int)
    )

    df["possession_player_known"] = int(
        possession_player is not None
    )

    df["is_possession_team"] = (
        df["team_name"].eq(possession_team).astype(int)
    )

    df["possession_team_known"] = int(
        possession_team is not None
    )

    df["node_index"] = np.arange(len(df))

    node_feature_columns = [
        "x", "y",
        "x_sg_m", "y_sg_m",
        "vx_sg_mps", "vy_sg_mps",
        "speed_sg_mps",
        "ax_sg_mps2", "ay_sg_mps2",
        "acceleration_sg_mps2",
        "movement_direction_deg",
        "is_detected",
        "sg_edge_estimate",
        "sg_full_window_support",
        "team_index",
        "is_possession_player",
        "possession_player_known",
        "is_possession_team",
        "possession_team_known",
    ]

    nodes = df[
        ["node_index", "player_id", "team_name"]
        + node_feature_columns
    ].copy()

    node_features = (
        nodes[node_feature_columns]
        .to_numpy(dtype=np.float32)
    )

    # Pairwise spatial distances
    positions = df[["x_sg_m", "y_sg_m"]].to_numpy(
        dtype=np.float64
    )

    velocities = df[["vx_sg_mps", "vy_sg_mps"]].to_numpy(
        dtype=np.float64
    )

    speeds = df["speed_sg_mps"].to_numpy(dtype=np.float64)

    delta = positions[None, :, :] - positions[:, None, :]

    distances = np.linalg.norm(delta, axis=2)

    team_indices = df["team_index"].to_numpy()
    player_ids = df["player_id"].to_numpy()

    edge_records = []

    for source in range(len(df)):

        same_team_candidates = [
            j for j in range(len(df))
            if j != source
            and team_indices[j] == team_indices[source]
        ]

        opponent_candidates = [
            j for j in range(len(df))
            if team_indices[j] != team_indices[source]
        ]

        nearest_teammates = sorted(
            same_team_candidates,
            key=lambda j: (
                distances[source, j],
                str(player_ids[j])
            )
        )[:k_same_team]

        nearest_opponents = sorted(
            opponent_candidates,
            key=lambda j: (
                distances[source, j],
                str(player_ids[j])
            )
        )[:k_opponents]

        for target in nearest_teammates:
            relation = "teammate"

            dx = positions[target, 0] - positions[source, 0]
            dy = positions[target, 1] - positions[source, 1]

            dvx = velocities[target, 0] - velocities[source, 0]
            dvy = velocities[target, 1] - velocities[source, 1]

            edge_records.append({
                "source": source,
                "target": target,
                "source_player_id": player_ids[source],
                "target_player_id": player_ids[target],
                "distance_m": distances[source, target],
                "delta_x_m": dx,
                "delta_y_m": dy,
                "delta_vx_mps": dvx,
                "delta_vy_mps": dvy,
                "delta_speed_mps": float(
    speeds[target] - speeds[source]
),

                "relative_velocity_mps": float(
    np.hypot(dvx, dvy)
),
                "same_team": 1,
                "relation": relation,
            })

        for target in nearest_opponents:
            relation = "opponent"

            dx = positions[target, 0] - positions[source, 0]
            dy = positions[target, 1] - positions[source, 1]

            dvx = velocities[target, 0] - velocities[source, 0]
            dvy = velocities[target, 1] - velocities[source, 1]

            edge_records.append({
                "source": source,
                "target": target,
                "source_player_id": player_ids[source],
                "target_player_id": player_ids[target],
                "distance_m": distances[source, target],
                "delta_x_m": dx,
                "delta_y_m": dy,
                "delta_vx_mps": dvx,
                "delta_vy_mps": dvy,
                "delta_speed_mps": (speeds[target] - speeds[source]),

                "relative_velocity_mps": float(np.hypot(dvx, dvy)),
                "same_team": 0,
                "relation": relation,
            })

    edges = pd.DataFrame(edge_records)

    edge_index = edges[
        ["source", "target"]
    ].to_numpy(dtype=np.int64).T

    edge_feature_columns = [
    "distance_m",
    "delta_x_m",
    "delta_y_m",
    "delta_vx_mps",
    "delta_vy_mps",
    "delta_speed_mps",
    "relative_velocity_mps",
    "same_team",
]

    edge_features = edges[
        edge_feature_columns
    ].to_numpy(dtype=np.float32)

    metadata = {
        "period": int(df["period"].iloc[0]),
        "frame": int(df["frame"].iloc[0]),
        "teams": teams,
        "possession_group": possession_group,
        "possession_team": possession_team,
        "possession_player_id": possession_player,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "k_same_team": k_same_team,
        "k_opponents": k_opponents,
    }

    return {
        "nodes": nodes,
        "node_features": node_features,
        "node_feature_columns": node_feature_columns,
        "edge_index": edge_index,
        "edges": edges,
        "edge_features": edge_features,
        "edge_feature_columns": edge_feature_columns,
        "metadata": metadata,
    }

