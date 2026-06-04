# tid3/datasets.py
#
# Download the UEA multivariate classification datasets used in the paper, standardize them
# into the TID3 long format, and run TID3 end-to-end (states.csv + sti_series.csv).
#
# Datasets are fetched from the public UEA/UCR archive mirror at timeseriesclassification.com.
# Only the standard library is used (urllib + zipfile); no sktime/aeon dependency.
import os
import io
import sys
import zipfile
import argparse
import urllib.request

try:
    # Normal package import (e.g. `python -m tid3.datasets`).
    from .standardize import read_ts_file, panel_to_long
    from .run import run_tid3
except ImportError:
    # Fallback so the file also runs directly (e.g. `python tid3/datasets.py`):
    # put the package parent on sys.path and import absolutely.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tid3.standardize import read_ts_file, panel_to_long
    from tid3.run import run_tid3

# The six UEA multivariate datasets evaluated in the paper.
PAPER_UEA_DATASETS = [
    "FaceDetection",
    "FingerMovements",
    "Heartbeat",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]

_BASE_URL = "http://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"


def _find_ts(directory, suffix):
    for root, _, files in os.walk(directory):
        for fn in files:
            if fn.endswith(suffix):
                return os.path.join(root, fn)
    return None


def download_uea_dataset(name, cache_dir="uea_cache"):
    """
    Download and extract a UEA dataset's `.ts` files into ``cache_dir/<name>/`` (cached:
    skips the download if the files already exist).

    Returns:
      (train_ts_path, test_ts_path)
    """
    dest = os.path.join(cache_dir, name)
    os.makedirs(dest, exist_ok=True)

    train = _find_ts(dest, "_TRAIN.ts")
    test = _find_ts(dest, "_TEST.ts")
    if train and test:
        return train, test

    url = _BASE_URL.format(name=name)
    print(f"Downloading {name} from {url} ...")
    with urllib.request.urlopen(url) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(dest)
    print(f"  extracted to {dest}")

    train = _find_ts(dest, "_TRAIN.ts")
    test = _find_ts(dest, "_TEST.ts")
    if not train or not test:
        raise FileNotFoundError(
            f"Could not locate {name}_TRAIN.ts / {name}_TEST.ts after extraction in {dest}."
        )
    return train, test


def load_paper_dataset(name, cache_dir="uea_cache", positive_label=None):
    """
    Download (if needed) a paper UEA dataset and return it in the TID3 long format
    (train + test concatenated). See ``standardize.panel_to_long`` for the schema.
    """
    train_path, test_path = download_uea_dataset(name, cache_dir=cache_dir)
    X_train, y_train = read_ts_file(train_path)
    X_test, y_test = read_ts_file(test_path)

    import numpy as np
    lt, le = X_train.shape[2], X_test.shape[2]
    if lt != le:
        target = max(lt, le)

        def _pad(a):
            if a.shape[2] == target:
                return a
            pad = np.full((a.shape[0], a.shape[1], target - a.shape[2]), np.nan)
            return np.concatenate([a, pad], axis=2)

        X_train, X_test = _pad(X_train), _pad(X_test)

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([np.asarray(y_train), np.asarray(y_test)], axis=0)
    print(f"{name}: {X.shape[0]} instances, {X.shape[1]} channels, {X.shape[2]} timepoints "
          f"(labels: {sorted(set(map(str, y)))})")
    return panel_to_long(X, y, positive_label=positive_label)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Download the paper's UEA datasets, standardize them, and run TID3 "
                    "(writes states.csv + sti_series.csv per dataset)."
    )
    p.add_argument("--dataset", default="FingerMovements",
                   choices=PAPER_UEA_DATASETS + ["all"],
                   help="Which UEA dataset to process. Default: FingerMovements (the smallest "
                        "paper dataset, for a quick first run). Use '--dataset all' for all six.")
    p.add_argument("--output-dir", default="tid3_runs",
                   help="Root output directory; each dataset goes to <output-dir>/<name>/.")
    p.add_argument("--cache-dir", default="uea_cache", help="Where downloaded .ts files are cached.")
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
    names = PAPER_UEA_DATASETS if args.dataset == "all" else [args.dataset]
    if args.dataset != "all":
        print(f"Processing dataset '{args.dataset}'. Use '--dataset all' to run all six "
              f"paper datasets (the large ones, e.g. FaceDetection/MotorImagery, take a while).")

    for name in names:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        long_df = load_paper_dataset(name, cache_dir=args.cache_dir)
        out_dir = os.path.join(args.output_dir, name)
        run_tid3(
            long_df,
            bins=args.bins,
            output_dir=out_dir,
            duration_preference=args.duration_preference,
            max_gap=args.max_gap,
            nb_candidates=args.nb_candidates,
            min_duration_threshold=args.min_duration_threshold,
        )
        print(f"{name}: outputs written to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
