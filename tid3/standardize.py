# tid3/standardize.py
#
# Convert raw multivariate temporal data into the long-format that TID3 consumes:
#
#   EntityID, TemporalPropertyID, TimeStamp, TemporalPropertyValue
#
# plus one class-assignment row per entity (TemporalPropertyID == -1, TimeStamp == 0)
# whose TemporalPropertyValue is the entity's class label (0 or 1). These rows define the
# two populations (D0 / D1) that TID3 contrasts.
import logging

import numpy as np
import pandas as pd

from .constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

logger = logging.getLogger(__name__)


def _to_3d_array(X):
    """
    Coerce supported panel containers into a 3D numpy array of shape
    [n_instances, n_channels, n_timepoints].

    Supported inputs:
      - 3D numpy array (returned as-is)
      - sktime/aeon "nested" DataFrame: rows = instances, columns = channels, each cell a
        pd.Series of length n_timepoints.
    """
    if isinstance(X, np.ndarray):
        if X.ndim != 3:
            raise ValueError(f"Expected a 3D array [instances, channels, timepoints], got ndim={X.ndim}")
        return X

    if isinstance(X, pd.DataFrame):
        n_instances, n_channels = X.shape
        # Each cell is expected to be a sequence (pd.Series / list / 1D array).
        series_lengths = {
            len(np.asarray(X.iat[i, c]))
            for i in range(n_instances)
            for c in range(n_channels)
        }
        if len(series_lengths) != 1:
            raise ValueError(
                "Nested DataFrame has unequal series lengths across instances/channels; "
                f"found lengths {sorted(series_lengths)}. TID3's long format supports ragged "
                "series, so build the long DataFrame directly with panel_to_long(array, ...) "
                "after padding, or pass arrays per entity."
            )
        n_timepoints = series_lengths.pop()
        arr = np.empty((n_instances, n_channels, n_timepoints), dtype=float)
        for i in range(n_instances):
            for c in range(n_channels):
                arr[i, c, :] = np.asarray(X.iat[i, c], dtype=float)
        return arr

    raise TypeError(
        f"Unsupported panel type {type(X)!r}. Pass a 3D numpy array "
        "[instances, channels, timepoints] or an sktime/aeon nested DataFrame."
    )


def panel_to_long(X, y, positive_label=None):
    """
    Convert panel (multivariate time series) data + labels into the TID3 long format.

    Parameters:
      X: panel data — a 3D numpy array [n_instances, n_channels, n_timepoints], or an
         sktime/aeon nested DataFrame (rows = instances, columns = channels).
      y: 1D array-like of class labels, one per instance (any hashable labels).
      positive_label: the label to map to class 1 (the target population, D1). The other
         label maps to class 0 (D0). If None, labels are sorted and the larger one is taken
         as positive (so a binary {0,1} task keeps 1 as positive).

    Indexing conventions (1-based, integer):
      EntityID            = instance index + 1
      TemporalPropertyID  = channel index + 1
      TimeStamp           = time index + 1
      TemporalPropertyValue = the measurement

    NaNs are kept (TID3 drops them downstream). Only binary tasks are supported, matching the
    paper's evaluation; for multiclass, reduce to binary (e.g. one-vs-rest) before calling.

    Returns:
      A long-format DataFrame with the four schema columns, including one class row per entity.
    """
    arr = _to_3d_array(X)
    n_instances, n_channels, n_timepoints = arr.shape

    y = np.asarray(y)
    if len(y) != n_instances:
        raise ValueError(f"y has length {len(y)} but X has {n_instances} instances.")

    unique_labels = sorted(pd.unique(y).tolist())
    if len(unique_labels) != 2:
        raise ValueError(
            f"panel_to_long supports binary tasks only; found {len(unique_labels)} "
            f"classes: {unique_labels}. Reduce to a binary task (e.g. one-vs-rest) first."
        )
    if positive_label is None:
        positive_label = unique_labels[-1]
    elif positive_label not in unique_labels:
        raise ValueError(f"positive_label={positive_label!r} not in labels {unique_labels}.")
    negative_label = [lbl for lbl in unique_labels if lbl != positive_label][0]
    logger.info(
        f"Class mapping: {positive_label!r} -> 1 (D1, target), {negative_label!r} -> 0 (D0)."
    )

    # Vectorized construction of the measurement rows.
    entity_idx = np.repeat(np.arange(1, n_instances + 1), n_channels * n_timepoints)
    prop_idx = np.tile(np.repeat(np.arange(1, n_channels + 1), n_timepoints), n_instances)
    time_idx = np.tile(np.arange(1, n_timepoints + 1), n_instances * n_channels)
    values = arr.reshape(-1)

    measurements = pd.DataFrame({
        ENTITY_ID: entity_idx,
        TEMPORAL_PROPERTY_ID: prop_idx,
        TIMESTAMP: time_idx,
        VALUE: values,
    })

    # One class-assignment row per entity.
    classes = (y == positive_label).astype(int)
    class_rows = pd.DataFrame({
        ENTITY_ID: np.arange(1, n_instances + 1),
        TEMPORAL_PROPERTY_ID: -1,
        TIMESTAMP: 0,
        VALUE: classes,
    })

    long_df = pd.concat([measurements, class_rows], ignore_index=True)
    long_df[ENTITY_ID] = long_df[ENTITY_ID].astype(int)
    long_df[TEMPORAL_PROPERTY_ID] = long_df[TEMPORAL_PROPERTY_ID].astype(int)
    long_df[TIMESTAMP] = long_df[TIMESTAMP].astype(int)
    return long_df


def load_uea_tsfile(train_path, test_path=None, positive_label=None):
    """
    Load a UEA multivariate `.ts` file (and optionally its test split) into the TID3 long
    format. Requires the optional ``sktime`` (or ``aeon``) dependency to parse `.ts` files.

    Parameters:
      train_path: path to the (train) `.ts` file.
      test_path: optional path to the test `.ts` file. When given, train and test instances
        are concatenated into one long DataFrame (TID3 / downstream code can re-split by
        EntityID as needed).
      positive_label: forwarded to panel_to_long (which label becomes class 1).

    Returns:
      A long-format DataFrame (see panel_to_long).
    """
    try:
        from sktime.datasets import load_from_tsfile  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without sktime
        raise ImportError(
            "Reading UEA .ts files requires 'sktime' (pip install sktime). "
            "Alternatively, build the long DataFrame yourself via "
            "panel_to_long(X, y) from a 3D numpy array [instances, channels, timepoints]."
        ) from exc

    X_train, y_train = load_from_tsfile(train_path, return_data_type="numpy3d")
    if test_path is None:
        return panel_to_long(X_train, y_train, positive_label=positive_label)

    X_test, y_test = load_from_tsfile(test_path, return_data_type="numpy3d")
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([np.asarray(y_train), np.asarray(y_test)], axis=0)
    return panel_to_long(X, y, positive_label=positive_label)
