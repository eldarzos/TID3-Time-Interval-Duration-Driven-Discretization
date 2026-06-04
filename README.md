# TID3: Time Interval Duration Driven Discretization

Reference implementation of **TID3**, a supervised state-abstraction (discretization) method
that selects cutoffs by maximizing the **cross-population divergence of the resulting Symbolic
Time Interval (STI) time-duration distributions**.

Given two populations (the two classes in classification, or the cases vs. controls in
continuous event prediction), TID3 greedily selects, one at a time, the cutoffs that make the
per-state STI time durations most different between the populations. The resulting STI series
is written as an explicit table (`sti_series.csv`) that any Time Intervals-Related Patterns
(TIRPs) mining algorithm can consume.

## Variants

All three variants score candidate cutoffs by the sum, over states, of Welch's t-statistic
between the two populations' STI time-duration distributions. They differ only in direction,
set via the `duration_preference` argument:

| Variant | `duration_preference` | Objective |
|---------|-----------------------|-----------|
| **TID32** | `"two_sided"`     | maximize the **absolute** t-statistic (any duration difference) |
| **TID31** | `"class1_longer"` | favor states where **class 1** (D1) STIs are longer than class 0 |
| **TID30** | `"class0_longer"` | favor states where **class 0** (D0) STIs are longer than class 1 |

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `tqdm`. The UEA `.ts` loader additionally needs
`sktime` (optional — only if you load `.ts` files rather than supplying your own data).

## Input format

TID3 consumes a **long-format** table with four columns:

| Column | Meaning |
|--------|---------|
| `EntityID` | entity/instance identifier (int) |
| `TemporalPropertyID` | variable/channel identifier (int) |
| `TimeStamp` | integer time index (each observation spans `[t, t+1)`) |
| `TemporalPropertyValue` | the measured value |

The two populations are encoded with **class-assignment rows**: one row per entity with
`TemporalPropertyID = -1`, `TimeStamp = 0`, and `TemporalPropertyValue` set to the entity's
class (`0` = D0, `1` = D1, the target population). The evaluation in the paper is binary.

### Building the input

- **UEA archive (`.ts`)** — `load_uea_tsfile(train_path, test_path=None)` parses a UEA
  multivariate dataset (via `sktime`) into the long format.
- **Arrays / panel data** — `panel_to_long(X, y, positive_label=None)` converts a 3D array of
  shape `[n_instances, n_channels, n_timepoints]` (or an sktime/aeon nested DataFrame) plus a
  label vector `y` into the long format. `EntityID`, `TemporalPropertyID` and `TimeStamp` are
  assigned as 1-based indices.

## Usage

### End-to-end, from Python

```python
from tid3 import panel_to_long, run_tid3

# X: [n_instances, n_channels, n_timepoints]; y: per-instance labels
long_df = panel_to_long(X, y, positive_label=1)

symbolic_series, boundaries = run_tid3(
    long_df,
    bins=3,                       # number of states; TID3 selects bins-1 cutoffs per variable
    output_dir="out/",
    duration_preference="two_sided",   # TID32 (use class1_longer / class0_longer for TID31 / TID30)
    max_gap=1,                    # interpolation gap for merging observations into STIs
)
```

### End-to-end, from the command line

```bash
# From a UEA .ts dataset (requires sktime):
python -m tid3.run --uea-train path/to/Dataset_TRAIN.ts --uea-test path/to/Dataset_TEST.ts \
    --output-dir out/ --bins 3 --duration-preference two_sided --max-gap 1

# From an already long-format CSV:
python -m tid3.run --input-csv path/to/long.csv --output-dir out/ --bins 3
```

### Library API (cutoffs only)

```python
from tid3 import tid3
symbolic_series, boundaries = tid3(long_df, bins=3, duration_preference="class0_longer")
# boundaries: {TemporalPropertyID: [cutoff_1, cutoff_2, ...]}
```

## Outputs

`run_tid3` (and the CLI) write into `output_dir`:

| File | Contents |
|------|----------|
| `states.csv` | the discretization states per variable: `StateID, TemporalPropertyID, BinId, BinLow, BinHigh` |
| `symbolic_time_series.csv` | each observation labeled with its global `StateID` |
| `sti_series.csv` | the **STI series**: `EntityID, TemporalPropertyID, StateID, StartTime, EndTime` (one row per symbolic time interval) |
| `sti_series-class-<c>.csv` | the STI series restricted to each population |
| `entity-class-relations.csv` | the `EntityID -> ClassID` mapping |

The STI series is a plain table — one row per symbolic time interval — so it is independent of
any particular TIRP mining tool. Each interval covers the half-open span `[StartTime, EndTime)`.

The default **interpolation gap** (`max_gap`) is **1**: consecutive same-state observations of
a variable are merged into one STI when the time gap between them is within `max_gap`.

## Algorithm (summary)

For each variable, starting from an empty cutoff set, TID3 builds a pool of `nb_candidates`
equal-frequency candidate cutoffs and performs greedy forward selection: at each of the `bins-1`
iterations it adds the candidate that maximizes the total per-state Welch t-statistic between the
two populations' STI time-duration distributions, given the cutoffs chosen so far. Complexity is
`O(bins · nb_candidates · N)` per variable, where `N` is the number of time points.

## Repository layout

```
tid3/
├── constants.py     # long-format column names
├── standardize.py   # UEA loader + panel→long converter
├── tid3.py          # the TID3 algorithm (greedy engine + 3 variants)
├── utils.py         # state assignment, candidate generation, STI series generation, I/O helpers
└── run.py           # end-to-end driver (states.csv + STI series); also a CLI (python -m tid3.run)
```
