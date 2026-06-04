# tid3/utils.py
import os
import numpy as np
import pandas as pd
import logging

from .constants import ENTITY_ID, VALUE, TEMPORAL_PROPERTY_ID, TIMESTAMP

logger = logging.getLogger(__name__)


def assign_state(value, boundaries):
    """
    Given a value and a sorted list of boundaries (cutoffs), assign a state id (starting at 1).

    For example, with 3 bins (boundaries = [b1, b2]):
      if value < b1:        state = 1
      if b1 <= value < b2:  state = 2
      if value >= b2:       state = 3
    """
    for i, b in enumerate(boundaries):
        if value < b:
            return i + 1
    return len(boundaries) + 1


def generate_candidate_cutpoints(df, nb_candidates):
    """
    Generate the initial candidate cutpoint pool by partitioning the value range of
    ``TemporalPropertyValue`` into ``nb_candidates`` equal-frequency bins.

    Parameters:
      df: A DataFrame that contains a column "TemporalPropertyValue".
      nb_candidates: Desired number of candidate cutpoints.

    Returns:
      A sorted list of candidate cutpoints.
    """
    values = df[VALUE].dropna().unique()
    values = np.sort(values)
    # If there are fewer than 2 unique values, return an empty list.
    if len(values) < 2:
        return []
    # Evenly space candidate indices between 1 and len(values)-1:
    indices = np.linspace(1, len(values) - 1, num=nb_candidates, dtype=int)
    candidates = values[indices]
    candidates = np.unique(candidates)
    return candidates.tolist()


def remove_na(data_to_use):
    """Drop rows with NaNs in the critical columns (EntityID / TemporalPropertyID / value)."""
    na_per_column = data_to_use.isna().sum()
    total_na = int(na_per_column.sum())

    if total_na > 0:
        logger.warning(
            f"Removing {total_na} rows containing NaNs in "
            f"{', '.join(na_per_column[na_per_column > 0].index)}"
        )

    data_to_use = data_to_use.dropna(subset=[ENTITY_ID, TEMPORAL_PROPERTY_ID, VALUE])
    return data_to_use


def generate_sti_series(symbolic_series: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """
    Build the Symbolic Time Interval (STI) series as an explicit table.

    Consecutive observations of the same (entity, property) are merged into one STI when
    they share the same StateID and the time gap between them is within ``max_gap``. Each
    observation spans ``[t, t+1)``, so a single observation at time ``t`` becomes the
    interval ``[t, t+1)``.

    This is a generic, algorithm-agnostic representation of the abstracted series: any
    Time Intervals-Related Patterns (TIRPs) mining algorithm can consume it.

    Parameters:
      symbolic_series (pd.DataFrame): columns EntityID, TemporalPropertyID, TimeStamp, StateID.
      max_gap (int): maximum gap threshold for merging consecutive observations into one STI.

    Returns:
      A DataFrame with one row per STI and columns:
        EntityID, TemporalPropertyID, StateID, StartTime, EndTime
      sorted by (EntityID, StartTime, EndTime, StateID, TemporalPropertyID).
    """
    df = symbolic_series.sort_values(by=[ENTITY_ID, TIMESTAMP, TEMPORAL_PROPERTY_ID])
    rows = []

    for entity in df[ENTITY_ID].unique():
        entity_df = df[df[ENTITY_ID] == entity].sort_values(by=[TIMESTAMP, TEMPORAL_PROPERTY_ID])
        for tpid, group in entity_df.groupby(TEMPORAL_PROPERTY_ID):
            group = group.sort_values(by=TIMESTAMP)
            current_interval = None
            for _, row in group.iterrows():
                ts = row[TIMESTAMP]
                state = row["StateID"]
                if current_interval is None:
                    current_interval = {"start": ts, "end": ts + 1, "StateID": state, TEMPORAL_PROPERTY_ID: tpid}
                else:
                    # Same state and the gap is within max_gap -> extend the interval.
                    if state == current_interval["StateID"] and (ts - current_interval["end"]) <= max_gap:
                        current_interval["end"] = ts + 1
                    else:
                        rows.append((entity, current_interval))
                        current_interval = {"start": ts, "end": ts + 1, "StateID": state, TEMPORAL_PROPERTY_ID: tpid}
            if current_interval is not None:
                rows.append((entity, current_interval))

    records = []
    for entity, interval in rows:
        state = interval["StateID"]
        if pd.isna(state):
            state = -1  # Replace NaN with -1
        records.append({
            "EntityID": int(entity),
            "TemporalPropertyID": int(interval[TEMPORAL_PROPERTY_ID]),
            "StateID": int(state),
            "StartTime": int(interval["start"]),
            "EndTime": int(interval["end"]),
        })

    sti_df = pd.DataFrame(records, columns=["EntityID", "TemporalPropertyID", "StateID", "StartTime", "EndTime"])
    sti_df = sti_df.sort_values(
        by=["EntityID", "StartTime", "EndTime", "StateID", "TemporalPropertyID"]
    ).reset_index(drop=True)
    return sti_df


def save_entity_ids(entity_relations: dict, output_path: str) -> None:
    """Write the EntityID -> ClassID mapping to entity-class-relations.csv."""
    df = pd.DataFrame(entity_relations.items(), columns=["EntityID", "ClassID"])
    df["EntityID"] = df["EntityID"].astype(int)
    df["ClassID"] = df["ClassID"].astype(int)
    df.to_csv(os.path.join(output_path, "entity-class-relations.csv"), index=False)
