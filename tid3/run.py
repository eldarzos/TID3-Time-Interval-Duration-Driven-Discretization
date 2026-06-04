# tid3/run.py
#
# End-to-end driver: take a long-format DataFrame (or a UEA .ts file), run TID3, and write
# the discretization states (states.csv) and the resulting Symbolic Time Interval (STI)
# series (sti_series.csv) — an explicit, algorithm-agnostic table that any Time
# Intervals-Related Patterns (TIRPs) mining algorithm can consume.
import os
import math
import argparse
import logging

import pandas as pd

from .constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE
from .tid3 import TID3
from .utils import generate_sti_series, save_entity_ids, remove_na

logger = logging.getLogger(__name__)


def _write_states_csv(boundaries: dict, output_dir: str):
    """
    Write states.csv from {TemporalPropertyID: [cutoffs]} and return a
    {(tpid, bin_id): StateID} mapping used to label the symbolic series.

    Columns: StateID, TemporalPropertyID, BinId, BinLow, BinHigh.
    """
    def sort_key(x):
        return (0, x) if isinstance(x, int) else (1, str(x))

    global_mapping = {}
    states_rows = []
    for tpid in sorted(boundaries.keys(), key=sort_key):
        bnds = boundaries[tpid]
        num_bins = len(bnds) + 1
        for local_bin in range(1, num_bins + 1):
            if local_bin == 1:
                bin_low = -math.inf
                bin_high = bnds[0] if bnds else math.inf
            elif local_bin == num_bins:
                bin_low = bnds[-1]
                bin_high = math.inf
            else:
                bin_low = bnds[local_bin - 2]
                bin_high = bnds[local_bin - 1]
            global_mapping[(tpid, local_bin)] = len(global_mapping) + 1
            states_rows.append({
                "StateID": global_mapping[(tpid, local_bin)],
                "TemporalPropertyID": tpid,
                "BinId": local_bin,
                "BinLow": round(bin_low, 5),
                "BinHigh": round(bin_high, 5),
            })
    pd.DataFrame(states_rows).to_csv(os.path.join(output_dir, "states.csv"), index=False)
    return global_mapping


def run_tid3(long_df: pd.DataFrame, bins: int, output_dir: str,
             duration_preference: str = "two_sided", max_gap: int = 1,
             nb_candidates: int = 100, min_duration_threshold: int = 2):
    """
    Run TID3 on a long-format DataFrame and save the outputs.

    Parameters:
      long_df: DataFrame with columns EntityID, TemporalPropertyID, TimeStamp,
        TemporalPropertyValue, including class rows (TemporalPropertyID == -1) that define
        the two populations.
      bins: number of states (TID3 selects bins-1 cutoffs per variable).
      output_dir: directory to write outputs into.
      duration_preference: "two_sided" (TID32), "class1_longer" (TID31) or "class0_longer" (TID30).
      max_gap: interpolation gap for merging consecutive observations into STIs (default 1).
      nb_candidates: number of initial equal-frequency candidate cutpoints.
      min_duration_threshold: minimum STI duration considered during fitting.

    Outputs written to ``output_dir``:
      states.csv                 - the discretization states (cutoffs) per variable.
      symbolic_time_series.csv   - per-observation StateID assignment.
      sti_series.csv             - the STI series: EntityID, TemporalPropertyID, StateID,
                                   StartTime, EndTime (one row per symbolic time interval).
      sti_series-class-<c>.csv   - the STI series restricted to each population.
      entity-class-relations.csv - the EntityID -> ClassID mapping.

    Returns:
      (symbolic_series, boundaries)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract the two populations from class-assignment rows, then drop them.
    class_rows = long_df[long_df[TEMPORAL_PROPERTY_ID] == -1]
    entity_class = {
        int(row[ENTITY_ID]): int(float(row[VALUE]))
        for _, row in class_rows.iterrows()
    }
    if not entity_class:
        raise ValueError(
            "No class-assignment rows (TemporalPropertyID == -1) found. TID3 needs two "
            "populations; see standardize.panel_to_long to build the long format."
        )
    data = long_df[long_df[TEMPORAL_PROPERTY_ID] != -1].copy()
    data = remove_na(data)

    # Fit + transform.
    model = TID3(
        bins,
        min_duration_threshold=min_duration_threshold,
        max_gap=max_gap,
        nb_candidates=nb_candidates,
        duration_preference=duration_preference,
    )
    model.entity_class = entity_class
    symbolic_series = model.fit_transform(data)
    boundaries = model.get_states()

    # states.csv + per-(tpid,bin) -> StateID map.
    state_mapping = _write_states_csv(boundaries, output_dir)

    # symbolic_time_series.csv (map local state -> global StateID).
    updated = symbolic_series.copy().reset_index(drop=True)
    updated["StateID"] = updated.apply(
        lambda row: state_mapping.get((row[TEMPORAL_PROPERTY_ID], int(row["state"])), int(row["state"])),
        axis=1,
    )
    updated = updated.drop(columns=["state"])
    updated.to_csv(os.path.join(output_dir, "symbolic_time_series.csv"), index=False)

    # sti_series.csv (the STI series) + per-population STI series files.
    generate_sti_series(updated, max_gap).to_csv(
        os.path.join(output_dir, "sti_series.csv"), index=False
    )
    updated_with_class = updated.copy()
    updated_with_class["EntityClass"] = updated_with_class[ENTITY_ID].map(entity_class)
    for cls in sorted(set(entity_class.values())):
        subset = updated_with_class[updated_with_class["EntityClass"] == cls]
        generate_sti_series(subset, max_gap).to_csv(
            os.path.join(output_dir, f"sti_series-class-{int(cls)}.csv"), index=False
        )

    # entity-class-relations.csv
    save_entity_ids(entity_class, output_dir)

    logger.info(f"Results saved in directory: {output_dir}")
    return symbolic_series, boundaries


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Run TID3 discretization end-to-end and write states.csv + the STI series."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-csv", help="Path to a long-format CSV (EntityID, TemporalPropertyID, "
                                          "TimeStamp, TemporalPropertyValue, incl. -1 class rows).")
    src.add_argument("--uea-train", help="Path to a UEA .ts (train) file (requires sktime).")
    p.add_argument("--uea-test", default=None, help="Optional UEA .ts test file to concatenate.")
    p.add_argument("--output-dir", required=True, help="Directory to write outputs into.")
    p.add_argument("--bins", type=int, default=3, help="Number of states (default: 3).")
    p.add_argument("--duration-preference", default="two_sided",
                   choices=["two_sided", "class1_longer", "class0_longer"],
                   help="TID32=two_sided, TID31=class1_longer, TID30=class0_longer.")
    p.add_argument("--max-gap", type=int, default=1, help="Interpolation gap for STIs (default: 1).")
    p.add_argument("--nb-candidates", type=int, default=100, help="Initial candidate cutpoints (default: 100).")
    p.add_argument("--min-duration-threshold", type=int, default=2,
                   help="Minimum STI duration considered during fitting (default: 2).")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if args.input_csv:
        long_df = pd.read_csv(args.input_csv, low_memory=False)
    else:
        from .standardize import load_uea_tsfile
        long_df = load_uea_tsfile(args.uea_train, test_path=args.uea_test)

    run_tid3(
        long_df,
        bins=args.bins,
        output_dir=args.output_dir,
        duration_preference=args.duration_preference,
        max_gap=args.max_gap,
        nb_candidates=args.nb_candidates,
        min_duration_threshold=args.min_duration_threshold,
    )


if __name__ == "__main__":
    main()
