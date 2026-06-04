# tid3/tid3.py
#
# TID3 (Time Interval Duration Driven Discretization): a supervised state-abstraction
# (discretization) method that selects cutoffs by maximizing the cross-population divergence
# of the resulting Symbolic Time Interval (STI) time-duration distributions.
#
# Three variants (selected via ``duration_preference``), all based on Welch's t-statistic:
#   - "two_sided"     (TID32): maximize the absolute t-statistic (any duration difference)
#   - "class1_longer" (TID31): favor states where class-1 STIs are longer than class-0
#   - "class0_longer" (TID30): favor states where class-0 STIs are longer than class-1
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import logging
import warnings
from tqdm import tqdm

from .base import TAMethod
from .utils import assign_state, generate_candidate_cutpoints
from .constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

# Suppress scipy warnings for nearly identical data (expected in constant variables)
warnings.filterwarnings(
    'ignore',
    message='Precision loss occurred in moment calculation due to catastrophic cancellation',
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TID3(TAMethod):
    """
    TID3 (Time Interval Duration Driven Discretization) - A supervised
    discretization method that selects cutoffs by maximizing the divergence
    of symbolic time interval (STI) time-duration distributions between
    two populations (e.g. two classes, or cases vs. controls).

    Scoring is the sum of per-state Welch's t-statistics computed over the STI
    time-duration distributions, with an optional one-sided directional preference
    (class 0 longer / class 1 longer).

    ZERO-VARIANCE HANDLING:
    When a specific cutoff configuration produces identical durations (zero variance),
    the scoring functions skip that state (contribute 0). The greedy algorithm naturally
    prefers configurations producing duration variance, allowing meaningful t-test comparisons.
    """

    # Only the t-statistic scoring method is supported.
    AVAILABLE_SCORING_METHODS = {
        "max_t_stat_sum": "Maximize sum of t-statistics for each state separately"
    }

    # Available duration preference options.
    AVAILABLE_DURATION_PREFERENCES = {
        "two_sided": "Favor any difference in duration distributions between classes (default)",
        "class1_longer": "Favor cutoffs where class 1 has longer durations than class 0",
        "class0_longer": "Favor cutoffs where class 0 has longer durations than class 1",
    }

    def __init__(self, bins: int, min_duration_threshold: int = 2, max_gap: int = 1,
                 nb_candidates: int = 100, duration_preference: str = "two_sided"):
        """
        Initialize TID3.

        Parameters:
            bins (int): Desired number of states (bins). TID3 selects ``bins - 1`` cutoffs.
            min_duration_threshold (int): Minimum STI time duration to consider for the
                duration analysis. STIs shorter than this are filtered out. If None, defaults to 0.
            max_gap (int): Maximum gap between consecutive timestamps still treated as one STI
                (same convention used in STI generation). Each observation spans [ts, ts+1).
            nb_candidates (int): Number of initial equal-frequency candidate cutpoints. Default 100.
            duration_preference (str): Direction of the duration comparison. One of
                {"two_sided" (default), "class1_longer", "class0_longer"}.
        """
        self.bins = bins
        self.min_duration_threshold = min_duration_threshold if min_duration_threshold is not None else 0
        self.max_gap = max_gap
        self.boundaries = None
        self.entity_class = {}  # Mapping {EntityID: class}; must be set before fit().
        self.nb_candidates = nb_candidates
        if duration_preference not in self.AVAILABLE_DURATION_PREFERENCES:
            raise ValueError(
                f"Unknown duration_preference '{duration_preference}'. "
                f"Available options: {list(self.AVAILABLE_DURATION_PREFERENCES.keys())}"
            )
        self.duration_preference = duration_preference
        self.final_scores_per_tpid = {}   # {tpid: final_t_stat_score}
        self.final_cutoffs_per_tpid = {}  # {tpid: list_of_cutoff_values}
        # Per-state t/p stats for the final chosen cutoffs of each TPID.
        # {tpid: {state_id: {'t': float, 'p': float}}}
        self.per_state_stats = {}

    def _get_scipy_alternative(self) -> str:
        """
        Map duration_preference to scipy's ``alternative`` parameter for the t-test.

        Returns:
            str: 'two-sided', 'greater', or 'less'.
        """
        if self.duration_preference == "two_sided":
            return "two-sided"
        elif self.duration_preference == "class1_longer":
            # class1 > class0: when comparing (class1, class0), we want 'greater'
            return "greater"
        elif self.duration_preference == "class0_longer":
            # class0 > class1: when comparing (class1, class0), we want 'less'
            return "less"
        else:
            return "two-sided"

    def _score_from_durations_direct(self, durations_by_class_and_state: dict):
        """
        Compute the t-stat-sum score directly from a precomputed
        {class_id: {state_id: [durations]}} structure.

        Returns:
            tuple[float, dict[int, dict[str, float]]]: (score, per_state_stats),
                propagated directly from `_score_t_stat_sum`. The dict maps
                state_id -> {'t': float, 'p': float} for states that
                contributed to the score; empty when no classes have data.
        """
        classes_with_data = list(durations_by_class_and_state.keys())
        if len(classes_with_data) < 2:
            return 0.0, {}

        states = set()
        for class_id in classes_with_data:
            states.update(durations_by_class_and_state[class_id].keys())

        return self._score_t_stat_sum(
            durations_by_class_and_state,
            classes_with_data,
            states,
        )

    def _describe_zero_score_reason(self, durations, classes, states, duration_filter_stats=None) -> str:
        """Summarize why a candidate produced no positive t-stat contribution."""
        if len(classes) < 2:
            return "fewer than 2 classes have valid durations"

        is_directional = self.duration_preference != "two_sided"
        if is_directional and (0 not in durations or 1 not in durations):
            return "directional scoring requires both class 0 and class 1 durations"

        insufficient_samples = 0
        zero_variance_both = 0
        ttest_exceptions = 0
        wrong_direction = 0
        valid_zero_t = 0
        evaluated_pairs = 0

        for state in states:
            if is_directional:
                d1 = durations.get(1, {}).get(state, [])
                d0 = durations.get(0, {}).get(state, [])
                evaluated_pairs += 1
                if len(d1) < 2 or len(d0) < 2:
                    insufficient_samples += 1
                    continue
                if np.std(d1) == 0 and np.std(d0) == 0:
                    zero_variance_both += 1
                    continue
                try:
                    t, _ = ttest_ind(d1, d0, alternative=self._get_scipy_alternative(), equal_var=False)
                except Exception:
                    ttest_exceptions += 1
                    continue
                if self.duration_preference == "class1_longer" and t <= 0:
                    wrong_direction += 1
                elif self.duration_preference == "class0_longer" and t >= 0:
                    wrong_direction += 1
                elif t == 0:
                    valid_zero_t += 1
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        d1 = durations.get(c1, {}).get(state, [])
                        d2 = durations.get(c2, {}).get(state, [])
                        evaluated_pairs += 1
                        if len(d1) < 2 or len(d2) < 2:
                            insufficient_samples += 1
                            continue
                        if np.std(d1) == 0 and np.std(d2) == 0:
                            zero_variance_both += 1
                            continue
                        try:
                            t, _ = ttest_ind(d1, d2, equal_var=False)
                        except Exception:
                            ttest_exceptions += 1
                            continue
                        if t == 0:
                            valid_zero_t += 1

        parts = []
        if duration_filter_stats is not None:
            parts.extend([
                f"raw_stis={duration_filter_stats['raw_stis']}",
                f"dropped_by_min_duration={duration_filter_stats['dropped_by_min_duration']}",
                f"dropped_by_nan_state={duration_filter_stats['dropped_by_nan_state']}",
                f"kept_after_duration_filter={duration_filter_stats['kept_after_duration_filter']}",
            ])
        parts.extend([
            f"states={len(states)}",
            f"class_pairs_checked={evaluated_pairs}",
            f"insufficient_samples={insufficient_samples}",
            f"zero_variance_both={zero_variance_both}",
            f"ttest_exceptions={ttest_exceptions}",
        ])
        if is_directional:
            parts.append(f"wrong_direction={wrong_direction}")
        if valid_zero_t:
            parts.append(f"valid_zero_t={valid_zero_t}")
        return ", ".join(parts)

    def _score_t_stat_sum(self, durations, classes, states):
        """Score using the sum of per-state t-statistics. Respects duration_preference.

        Returns:
            tuple[float, dict[int, dict[str, float]]]: (score, per_state_stats),
                where per_state_stats maps state_id -> {'t': float, 'p': float}
                for every state that contributed to the score. States with <2
                samples in either class or zero variance in both are absent.
        """
        is_directional = self.duration_preference != "two_sided"

        # For directional, we need class 0 and 1
        if is_directional and (0 not in durations or 1 not in durations):
            return 0.0, {}

        per_pattern_stats = []
        per_state_stats = {}  # {state_id: {'t': float, 'p': float}}

        for state in states:
            if is_directional:
                d1 = durations.get(1, {}).get(state, [])
                d0 = durations.get(0, {}).get(state, [])
                if len(d1) >= 2 and len(d0) >= 2:
                    if np.std(d1) == 0 and np.std(d0) == 0:
                        continue
                    try:
                        t, p = ttest_ind(d1, d0, alternative=self._get_scipy_alternative(), equal_var=False)
                        if self.duration_preference == "class1_longer" and t > 0:
                            per_pattern_stats.append(t)
                            per_state_stats[state] = {'t': float(t), 'p': float(p)}
                        elif self.duration_preference == "class0_longer" and t < 0:
                            per_pattern_stats.append(abs(t))
                            per_state_stats[state] = {'t': float(t), 'p': float(p)}
                    except Exception:
                        pass
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        d1 = durations.get(c1, {}).get(state, [])
                        d2 = durations.get(c2, {}).get(state, [])
                        if len(d1) >= 2 and len(d2) >= 2:
                            if np.std(d1) == 0 and np.std(d2) == 0:
                                continue
                            try:
                                t, p = ttest_ind(d1, d2, equal_var=False)
                                per_pattern_stats.append(abs(t))
                                per_state_stats[state] = {'t': float(t), 'p': float(p)}
                            except Exception:
                                pass

        return float(sum(per_pattern_stats)), per_state_stats

    def _build_candidate_search_context(self, df: pd.DataFrame, temporal_property_id: int = None):
        """
        Pre-compute the numpy arrays and break masks consumed by _evaluate_candidate_pool.

        Sorts data once, computes per-entity group boundaries (one group per
        entity/class/tpid), the time-gap-break mask (positions where a STI cannot span
        regardless of state assignment), and the NaN mask.
        """
        df_sorted = df.sort_values([ENTITY_ID, 'Class', TEMPORAL_PROPERTY_ID, TIMESTAMP])

        timestamps = df_sorted[TIMESTAMP].values.astype(np.float64)
        values = df_sorted[VALUE].values.astype(np.float64)
        classes_arr = df_sorted['Class'].values

        n_total = len(df_sorted)

        if n_total <= 1:
            logger.error(f"Insufficient data for TemporalPropertyID {temporal_property_id}: n_total={n_total}. At least 2 observations are required.")
            raise ValueError(f"_build_candidate_search_context requires at least 2 observations for TemporalPropertyID {temporal_property_id}, got {n_total}")

        group_labels = df_sorted.groupby(
            [ENTITY_ID, 'Class', TEMPORAL_PROPERTY_ID], sort=False
        ).ngroup().values

        group_boundary_mask = np.diff(group_labels) != 0
        group_boundaries = np.where(group_boundary_mask)[0] + 1
        group_starts = np.concatenate([[0], group_boundaries])
        group_ends = np.concatenate([group_boundaries, [n_total]])

        num_groups = len(group_starts)
        group_classes = classes_arr[group_starts]

        # Pre-compute positions where no candidate interval can ever span across, regardless of cutpoint choice.
        # A hard boundary occurs when the time gap between consecutive observations exceeds the allowed gap (max_gap).
        # This is computed once here — time gaps depend only on timestamps, not on state assignment.
        # Each observation spans [ts, ts+1), so the gap is ts_next - ts_prev - 1.
        # A hard boundary occurs when ts_next - ts_prev > max_gap + 1.
        time_gap_breaks = np.zeros(n_total, dtype=bool)
        for g in range(num_groups):
            s, e = group_starts[g], group_ends[g]
            if e - s > 1:
                time_gap_breaks[s + 1:e] = np.diff(timestamps[s:e]) > self.max_gap + 1

        nan_mask = np.isnan(values)
        has_nans = np.any(nan_mask)

        return {
            'timestamps': timestamps,
            'values': values,
            'classes_arr': classes_arr,
            'group_starts': group_starts,
            'group_ends': group_ends,
            'group_classes': group_classes,
            'time_gap_breaks': time_gap_breaks,
            'nan_mask': nan_mask,
            'has_nans': has_nans,
            'num_groups': num_groups,
            'n_total': n_total,
            'tpid': temporal_property_id,
        }

    def _evaluate_candidate_pool(self, ctx, existing_cutoffs, pool, iteration_label):
        """
        Score each candidate in `pool` as an additional cutoff added to `existing_cutoffs`.

        Builds STI durations per (class, state) over the candidate's full cutoff list and
        scores via _score_from_durations_direct. Returns parallel arrays/lists over `pool`.
        """
        timestamps = ctx['timestamps']
        values = ctx['values']
        group_starts = ctx['group_starts']
        group_ends = ctx['group_ends']
        group_classes = ctx['group_classes']
        time_gap_breaks = ctx['time_gap_breaks']
        nan_mask = ctx['nan_mask']
        has_nans = ctx['has_nans']
        num_groups = ctx['num_groups']

        scores = np.full(len(pool), -np.inf)
        per_state_stats_list = [None] * len(pool)
        zero_score_reasons = [None] * len(pool)
        show_progress = logger.isEnabledFor(logging.DEBUG)
        pbar = tqdm(total=len(pool),
                    desc=f"    Evaluating candidates ({iteration_label})",
                    disable=not show_progress,
                    leave=False)

        for i, candidate in enumerate(pool):
            if len(existing_cutoffs) > 0 and any(np.isclose(candidate, existing_cutoffs)):
                pbar.update(1)
                continue

            cutoffs = np.sort(np.append(existing_cutoffs, candidate))
            states = np.searchsorted(cutoffs, values, side='left')

            if has_nans:
                states = states.astype(np.int64)
                states[nan_mask] = -1

            durations_by_class_and_state = {}
            duration_filter_stats = {
                "raw_stis": 0,
                "dropped_by_min_duration": 0,
                "dropped_by_nan_state": 0,
                "kept_after_duration_filter": 0,
            }

            # Build STI durations per class and state across all entities
            for g in range(num_groups):
                # g = one entity's observations for this variable
                s, e = group_starts[g], group_ends[g]
                class_id = group_classes[g]
                n = e - s  # number of observations for this entity

                if n == 0:
                    continue

                if class_id not in durations_by_class_and_state:
                    durations_by_class_and_state[class_id] = {}

                st_group = states[s:e]
                ts_group = timestamps[s:e]

                # Single observation: STI duration is always 1 by convention
                if n == 1:
                    dur = 1
                    state = int(st_group[0])
                    duration_filter_stats["raw_stis"] += 1
                    if dur >= self.min_duration_threshold and state != -1:
                        duration_filter_stats["kept_after_duration_filter"] += 1
                        durations_by_class_and_state[class_id].setdefault(state, []).append(dur)
                    elif dur < self.min_duration_threshold:
                        duration_filter_stats["dropped_by_min_duration"] += 1
                    elif state == -1:
                        duration_filter_stats["dropped_by_nan_state"] += 1
                    continue

                # Detect STI boundaries: state change or time gap exceeds allowed gap
                state_changes = np.diff(st_group) != 0
                breaks = state_changes | time_gap_breaks[s + 1:e]

                # Convert break positions to STI start/end index pairs
                break_idx = np.where(breaks)[0]
                start_indices = np.concatenate([[0], break_idx + 1])
                end_indices = np.concatenate([break_idx, [n - 1]])

                # Compute STI durations: end timestamp + 1 - start timestamp
                durations_arr = ts_group[end_indices] + 1 - ts_group[start_indices]
                interval_states = st_group[start_indices]
                duration_filter_stats["raw_stis"] += len(durations_arr)

                # Filter out STIs below min duration and NaN-valued intervals
                min_duration_valid = durations_arr >= self.min_duration_threshold
                duration_filter_stats["dropped_by_min_duration"] += int(np.sum(~min_duration_valid))
                valid = min_duration_valid
                if has_nans:
                    non_nan_state = interval_states != -1
                    duration_filter_stats["dropped_by_nan_state"] += int(
                        np.sum(min_duration_valid & ~non_nan_state)
                    )
                    valid = valid & non_nan_state

                durations_valid = durations_arr[valid]
                states_valid = interval_states[valid]
                duration_filter_stats["kept_after_duration_filter"] += len(durations_valid)

                # Accumulate durations into class -> state -> [durations] structure
                class_dict = durations_by_class_and_state[class_id]
                for j in range(len(durations_valid)):
                    state = int(states_valid[j])
                    if state not in class_dict:
                        class_dict[state] = []
                    class_dict[state].append(int(durations_valid[j]))

            # Score this candidate by how well STI durations separate the classes
            try:
                scores[i], per_state_stats_list[i] = self._score_from_durations_direct(durations_by_class_and_state)
                if scores[i] == 0.0:
                    if per_state_stats_list[i]:
                        zero_score_reasons[i] = "valid tests existed, but all t-stat contributions were 0"
                    else:
                        classes_with_data = list(durations_by_class_and_state.keys())
                        candidate_states = set()
                        for class_dict in durations_by_class_and_state.values():
                            candidate_states.update(class_dict.keys())
                        zero_score_reasons[i] = self._describe_zero_score_reason(
                            durations_by_class_and_state,
                            classes_with_data,
                            candidate_states,
                            duration_filter_stats,
                        )
            except Exception:
                scores[i] = -np.inf

            pbar.update(1)

        pbar.close()
        return scores, per_state_stats_list, zero_score_reasons

    def _run_greedy_search(self, ctx, initial_cutpoints, candidate_pool, target_bins,
                            temporal_property_id):
        """
        Greedy iterative cutpoint selection starting from `initial_cutpoints`.

        At each iteration calls _evaluate_candidate_pool over the remaining pool,
        appends the argmax candidate, removes it and near-duplicates from the pool,
        and continues until target_bins - 1 cutpoints exist or candidates run out.

        Returns (cutpoints, scores, per_state_stats). per_state_stats corresponds to
        the final cutpoints (by construction the last successful iteration's winner
        holds the stats for the final set).
        """
        chosen_cutpoints = list(initial_cutpoints)
        chosen_scores = []
        # Local copy so we don't mutate the caller's pool; pre-filter anything already chosen.
        candidate_pool = list(candidate_pool)
        if chosen_cutpoints:
            candidate_pool = [c for c in candidate_pool
                              if not any(np.isclose(c, chosen_cutpoints))]

        latest_per_state_stats = None
        start_iter = len(chosen_cutpoints) + 1
        for iteration in range(start_iter, target_bins):
            scores, per_state_stats_list, zero_score_reasons = self._evaluate_candidate_pool(
                ctx, chosen_cutpoints, candidate_pool,
                f"bin {iteration}/{target_bins-1}",
            )

            # Check for valid candidates
            if len(scores) == 0:
                logger.warning(f"Early termination at iteration {iteration}: No candidate scores available")
                break
            elif np.all(np.isneginf(scores)):
                logger.warning(f"Early termination at iteration {iteration}: All {len(scores)} candidates produced invalid scores")
                break

            # Select best candidate
            best_idx = int(np.argmax(scores))
            best_candidate = candidate_pool[best_idx]
            best_score = scores[best_idx]
            latest_per_state_stats = per_state_stats_list[best_idx]
            if best_score == 0.0:
                print(
                    f"  [TID3]   Zero-score reason for TPID {temporal_property_id}, "
                    f"candidate {best_candidate:.4f}: {zero_score_reasons[best_idx]}"
                )

            print(f"  [TID3]   Iter {iteration}/{target_bins-1}: winner={best_candidate:.4f} (score={best_score:.4f})")

            chosen_cutpoints.append(best_candidate)
            chosen_cutpoints.sort()
            chosen_scores.append(best_score)

            # Remove chosen candidate and near-duplicates from pool
            candidate_pool.pop(best_idx)
            candidate_pool = [c for c in candidate_pool
                              if not any(np.isclose(c, chosen_cutpoints))]

            # Check if pool is exhausted
            if len(candidate_pool) == 0 and iteration < target_bins - 1:
                logger.warning(f"Candidate pool exhausted at iteration {iteration}: achieved {len(chosen_cutpoints) + 1}/{target_bins} bins")

        return chosen_cutpoints, chosen_scores, latest_per_state_stats

    def _optimized_candidate_selection(self, df: pd.DataFrame, nb_bins: int, temporal_property_id: int = None):
        """
        Greedy univariate candidate selection for the t-stat-sum scoring.

        Builds the precomputed numpy context, generates the equal-frequency candidate pool,
        and delegates to _run_greedy_search for the iterative cutpoint selection.

        Returns:
            tuple: (chosen_cutpoints, chosen_scores).
        """
        ctx = self._build_candidate_search_context(df, temporal_property_id)
        candidate_pool = generate_candidate_cutpoints(df, self.nb_candidates)

        print(f"\n  [TID3] Variable {temporal_property_id}: {len(candidate_pool)} initial candidates "
              f"(equal-frequency from {len(df[VALUE].dropna().unique())} unique values)")
        if candidate_pool:
            print(f"  [TID3]   Range: [{candidate_pool[0]:.4f}, {candidate_pool[-1]:.4f}]")

        chosen_cutpoints, chosen_scores, latest_per_state_stats = self._run_greedy_search(
            ctx, initial_cutpoints=[], candidate_pool=candidate_pool,
            target_bins=nb_bins, temporal_property_id=temporal_property_id,
        )

        cutpoints_str = ", ".join([f"{c:.4f}" for c in chosen_cutpoints])
        print(f"  [TID3]   Final cutpoints: [{cutpoints_str}] → {len(chosen_cutpoints)+1} bins")

        if latest_per_state_stats is not None:
            self.per_state_stats[temporal_property_id] = latest_per_state_stats

        return chosen_cutpoints, chosen_scores

    def _generate_cutpoints(self, df: pd.DataFrame, temporal_property_id: int = None):
        """
        For a given DataFrame (corresponding to one variable), choose candidate cutpoints
        that maximize the cross-population STI duration divergence.
        """
        if temporal_property_id is None:
            logger.error("_generate_cutpoints called without a temporal_property_id; this argument is required.")
            raise ValueError("_generate_cutpoints requires a temporal_property_id")

        # Handle case where all values are the same
        if df[VALUE].nunique() == 1:
            logger.warning(f"Insufficient variability for TemporalPropertyID {temporal_property_id}: only 1 unique value ({df[VALUE].iloc[0]})")
            candidates = [df[VALUE].min()] * (self.bins - 1)
            return candidates

        candidates, scores = self._optimized_candidate_selection(df, self.bins, temporal_property_id)

        # Log final bin analysis
        achieved_bins = len(candidates) + 1
        if achieved_bins < self.bins:
            logger.warning(f"TemporalPropertyID {temporal_property_id}: achieved {achieved_bins}/{self.bins} bins - missing {self.bins - achieved_bins} bins due to optimization constraints")

        # Store final score for diagnostics
        final_score = scores[-1] if scores and len(scores) > 0 else 0.0
        self.final_scores_per_tpid[temporal_property_id] = final_score
        self.final_cutoffs_per_tpid[temporal_property_id] = list(candidates)

        return candidates

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit TID3 by selecting cutpoints for each variable based on the cross-population
        STI time-duration divergence. Requires ``self.entity_class`` to be set first.
        """
        boundaries = {}
        temporal_properties = list(data.groupby(TEMPORAL_PROPERTY_ID).groups.keys())

        logger.info(f"Processing {len(temporal_properties)} temporal properties sequentially")

        pbar = tqdm(total=len(temporal_properties),
                    desc="Processing temporal properties",
                    unit="property",
                    ncols=120)

        for tpid, group in data.groupby(TEMPORAL_PROPERTY_ID):
            try:
                if not self.entity_class:
                    raise ValueError('No entity class mapping found')
                group = group.assign(Class=group[ENTITY_ID].map(self.entity_class))

                cutpoints = self._generate_cutpoints(group, tpid)
                boundaries[tpid] = cutpoints

                achieved_bins = len(boundaries[tpid]) + 1
                pbar.set_postfix({
                    'current_tpid': tpid,
                    'bins': f'{achieved_bins}/{self.bins}',
                    'completed': f'{pbar.n + 1}/{len(temporal_properties)}'
                })
                pbar.update(1)

            except Exception as e:
                logger.error(f"Failed to process TemporalPropertyID {tpid}: {e}")
                raise ValueError(f"Error processing TemporalPropertyID {tpid}: {e}")

        pbar.close()
        self.boundaries = boundaries

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the learned TID3 cutpoints. Each observation is assigned a
        state (1..bins) via assign_state() against its variable's boundaries.
        """
        data = data.copy()
        data["state"] = data.apply(
            lambda row: assign_state(row[VALUE], self.boundaries.get(row[TEMPORAL_PROPERTY_ID], [])),
            axis=1,
        )
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_states(self):
        """Return the computed TID3 boundaries: {TemporalPropertyID: [cutoffs]}."""
        return self.boundaries


def tid3(data: pd.DataFrame, bins: int,
         min_duration_threshold: int = 2, max_gap: int = 1,
         nb_candidates: int = 100, duration_preference: str = "two_sided"):
    """
    Convenience wrapper to run TID3 on a long-format dataset.

    The two populations are read from class-assignment rows (TemporalPropertyID == -1),
    whose TemporalPropertyValue holds the class (0 or 1) of the entity.

    Parameters:
      data: input long-format DataFrame.
      bins: number of states desired.
      min_duration_threshold: minimum STI duration considered for the duration analysis.
      max_gap: maximum gap between timestamps treated as one STI.
      nb_candidates: number of initial equal-frequency candidate cutpoints.
      duration_preference: one of {"two_sided", "class1_longer", "class0_longer"}.

    Returns:
      (symbolic_series, states): the transformed DataFrame (with a "state" column) and the
      cutpoints per variable.
    """
    class_rows = data[data[TEMPORAL_PROPERTY_ID] == -1]
    entity_class = {
        int(row[ENTITY_ID]): int(float(row[VALUE]))
        for _, row in class_rows.iterrows()
    }
    data = data[data[TEMPORAL_PROPERTY_ID] != -1]

    model = TID3(
        bins,
        min_duration_threshold=min_duration_threshold,
        max_gap=max_gap,
        nb_candidates=nb_candidates,
        duration_preference=duration_preference,
    )
    model.entity_class = entity_class
    symbolic_series = model.fit_transform(data)
    return symbolic_series, model.get_states()
