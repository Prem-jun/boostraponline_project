"""
RBULT Statistical Process Control (SPC) Module
===============================================

Implements a Memory-Bounded Adaptive Control Chart (RBULTControlChart)
for univariate and multivariate non-Gaussian IoT data streams.

Classes:
    RBULTControlChart — Real-time stream monitor establishing dynamic LCL/UCL bounds.
"""

from typing import Dict, List, Union, Tuple, Optional
import math
import time
import sys
import numpy as np
import pandas as pd
from scipy.stats import binom
from online_bootstrap.bootstrap_online import BootstrapOnline
from online_bootstrap.res_bootstrap_v2 import ResBootstrap


class RBULTControlChart:
    """Memory-Bounded Adaptive Control Chart (RBULT-SPC).

    Monitors single or multi-dimensional data streams by establishing dynamic
    Lower Control Limits (LCL = L_d) and Upper Control Limits (UCL = R_d)
    for each feature channel.

    Memory Complexity: O(D) where D is the number of monitored features.
    Time Complexity: Constant per chunk per feature.

    Attributes:
        features: List of feature names to monitor.
        minmax_flag: Whether min-max bootstrap is enabled for boundary estimation.
        outlier_filter: Whether Z-score outlier filtering (Algorithm 4) is applied.
        alpha_sys: Target overall system false alarm rate (default: 0.05).
        fwer_correction: 'bonferroni', 'sidak', or 'none'.
        chunk_alarm_rate: Default chunk alarm threshold as a fraction q of chunk size k,
            giving C = ceil(q * k). Replaces the former fixed count of 3.
        alpha_chunk: Target per-chunk system false alarm probability, used by the
            optional Phase I binomial calibration.
        engines: Dict mapping feature_name -> BootstrapOnline instance.
        results: Dict mapping feature_name -> ResBootstrap instance.
        feature_thresholds: Per-dimension thresholds C_d set by calibrate_phase1();
            None until Phase I calibration is performed.
        history: Event log of streaming updates and Out-Of-Control (OOC) detections.

    Chunk alarm threshold
    ---------------------
    A chunk is flagged OOC when ANY monitored feature accumulates at least C
    out-of-bound samples within it. C is resolved per chunk in this order:

      1. An explicit ``ooc_threshold_count`` passed to update_chunk().
      2. Per-dimension C_d from calibrate_phase1(), when Phase I data is available.
      3. The default rate rule  C = ceil(chunk_alarm_rate * k).

    Rule 3 is the default because an absolute count is not scale-free: the number of
    violations a chunk carries while in control grows with k, so a constant such as 3
    is far above the noise floor for small chunks and far below it for large ones.
    """

    def __init__(self, features: List[str], minmax_flag: bool = False,
                 outlier_filter: bool = True, alpha_sys: float = 0.05,
                 fwer_correction: str = 'bonferroni',
                 chunk_alarm_rate: float = 0.05, alpha_chunk: float = 0.05,
                 difference: bool = False):
        """Initialize the RBULT Control Chart engine.

        Args:
            features: List of column/sensor feature names.
            minmax_flag: If True, use min-max bootstrap for tail bounds.
            outlier_filter: If True, apply Z-score spike filtering before bound updates.
            alpha_sys: Desired overall system false alarm rate.
            fwer_correction: Adjustment type ('bonferroni', 'sidak', 'none').
            chunk_alarm_rate: Fraction q of chunk size used for the default
                threshold C = ceil(q * k).
            alpha_chunk: Target per-chunk false alarm probability for Phase I calibration.
            difference: If True, first-order difference each dimension as chunks arrive,
                carrying one scalar of state per feature so the transform stays O(D).
                Pass new_sequence=True to update_chunk() at boundaries where the process
                restarts, so no difference is taken across them.
        """
        self.features = features
        self.minmax_flag = minmax_flag
        self.outlier_filter = outlier_filter
        self.alpha_sys = alpha_sys
        self.fwer_correction = fwer_correction
        self.chunk_alarm_rate = chunk_alarm_rate
        self.alpha_chunk = alpha_chunk

        # Optional first-order differencing, applied per dimension as the chunk arrives.
        # State is one scalar per feature — the last value of the previous chunk — so the
        # transform is genuinely streaming and keeps the chart at O(D). Set
        # new_sequence=True on update_chunk() when a chunk begins an independent segment
        # (e.g. a TEP simulation run), so no difference is taken across that boundary.
        self.difference = difference
        self._diff_state: Dict[str, Optional[float]] = {feat: None for feat in features}

        # Phase I state (optional per-dimension binomial calibration)
        self.feature_thresholds: Optional[Dict[str, int]] = None
        self._phase1_active = False
        self._phase1_counts: List[List[int]] = []
        self._phase1_sizes: List[int] = []

        # Running interval-width statistics, accumulated with Welford so the chart
        # keeps O(D) memory (8 scalars per feature, independent of stream length).
        # Coverage alone cannot judge an interval: a wide enough interval attains
        # 100% coverage trivially. Width must be reported alongside it.
        self._width_stats: Dict[str, dict] = {
            feat: {'n': 0, 'w_mean': 0.0, 'w_m2': 0.0, 'l_mean': 0.0, 'l_m2': 0.0,
                   'r_mean': 0.0, 'r_m2': 0.0, 'range_sum': 0.0, 'range_n': 0}
            for feat in features
        }

        # Prequential (one-step-ahead) scoring counters, O(D).
        # Each chunk is scored against the limits in force BEFORE it is allowed to
        # update them, which is how a deployed control chart actually operates.
        # compute_spc_metrics()'s coverage is in-sample by contrast: it applies the
        # FINAL limits retrospectively to the whole stream.
        self._preq: Dict[str, dict] = {feat: {'covered': 0, 'total': 0} for feat in features}
        self._preq_joint: Dict[str, int] = {'covered': 0, 'total': 0}

        # Calculate adjusted alpha per dimension for FWER control
        num_dim = len(features)
        if fwer_correction == 'bonferroni':
            self.alpha_dim = alpha_sys / max(1, num_dim)
        elif fwer_correction == 'sidak':
            self.alpha_dim = 1.0 - (1.0 - alpha_sys) ** (1.0 / max(1, num_dim))
        else:
            self.alpha_dim = alpha_sys

        # Initialize engines & collectors per feature
        self.engines: Dict[str, BootstrapOnline] = {}
        self.results: Dict[str, ResBootstrap] = {}

        for feat in features:
            eng = BootstrapOnline()
            eng.set_online(minmax_flag=minmax_flag)
            self.engines[feat] = eng

            res = ResBootstrap()
            res.add_init_params(eng)
            self.results[feat] = res

        self.history: List[dict] = []
        self.total_chunks_processed = 0
        self.total_samples_processed = 0

    def default_threshold(self, chunk_len: int) -> int:
        """Rate-based chunk alarm threshold C = ceil(chunk_alarm_rate * k).

        Args:
            chunk_len: Number of samples k in the current chunk.

        Returns:
            Threshold C, at least 1.
        """
        return max(1, int(math.ceil(self.chunk_alarm_rate * chunk_len)))

    def resolve_threshold(self, feat: str, chunk_len: int,
                          explicit: Optional[int] = None) -> int:
        """Resolve the alarm threshold for one feature on the current chunk.

        Precedence: explicit argument > Phase I per-dimension C_d > rate rule.
        """
        if explicit is not None:
            return explicit
        if self.feature_thresholds is not None and feat in self.feature_thresholds:
            return self.feature_thresholds[feat]
        return self.default_threshold(chunk_len)

    def _apply_difference(self, df_chunk: pd.DataFrame,
                          new_sequence: bool = False) -> pd.DataFrame:
        """First-order difference the chunk per dimension, carrying O(D) state.

        For sample t > 1 within the chunk this is x[t] - x[t-1]. For t = 1 the difference
        is taken against the last value of the previous chunk, held in `_diff_state` —
        one scalar per feature, so the transform adds no memory that grows with the stream.

        Args:
            df_chunk: Raw chunk.
            new_sequence: True when this chunk starts an independent segment. The carried
                state is discarded and the chunk's first row is dropped, so no difference
                is ever taken across a boundary where the process restarts. This is
                required for TEP, whose stream is 2,900 independent simulation runs
                concatenated; differencing across such a boundary produces a meaningless
                jump, which is exactly what made AI4I's 'Tool wear Rate' an artefact.

        Returns:
            The differenced chunk. One row shorter than the input on the first chunk of
            each sequence, otherwise the same length.
        """
        if new_sequence:
            for feat in self.features:
                self._diff_state[feat] = None

        out = {}
        drop_first = False
        for feat in self.features:
            if feat not in df_chunk.columns:
                continue
            v = df_chunk[feat].to_numpy(dtype=float)
            prev = self._diff_state[feat]
            if prev is None:
                d = np.diff(v)                      # loses the first row
                drop_first = True
            else:
                d = np.diff(np.concatenate([[prev], v]))
            out[feat] = d
            if len(v):
                self._diff_state[feat] = float(v[-1])

        if not out:
            return df_chunk.iloc[0:0]
        idx = df_chunk.index[1:] if drop_first else df_chunk.index
        return pd.DataFrame(out, index=idx)

    def _score_prequential(self, df_chunk: pd.DataFrame, chunk_len: int) -> None:
        """Score a chunk against the limits held *before* it updates them.

        Run as a pre-pass, so every dimension is evaluated against the state carried in
        from chunks 1..m-1. Dimensions whose estimator has not yet seen data are skipped
        (their initial interval is the empty [9999.99, -9999.99]), which excludes the
        first chunk from the denominator exactly as a forecaster would.

        Joint coverage is only accumulated on chunks where every monitored dimension
        already has limits, so the marginal and joint denominators stay consistent.
        """
        joint = None
        for feat in self.features:
            if feat not in df_chunk.columns:
                continue
            eng = self.engines[feat]
            if eng.total_size <= 0:
                continue
            lcl, ucl = eng.exp_l, eng.exp_r

            vals = df_chunk[feat].dropna().values
            if len(vals):
                s = self._preq[feat]
                s['covered'] += int(np.sum((vals >= lcl) & (vals <= ucl)))
                s['total'] += len(vals)

            raw = df_chunk[feat].values
            in_bounds = (raw >= lcl) & (raw <= ucl)
            joint = in_bounds if joint is None else (joint & in_bounds)

        if joint is not None:
            self._preq_joint['covered'] += int(np.sum(joint))
            self._preq_joint['total'] += chunk_len

    def _update_width_stats(self, feat: str, lcl: float, ucl: float,
                            vals: List[float]) -> None:
        """Accumulate interval-width and boundary-stability statistics for one chunk.

        Uses Welford's online algorithm so mean and variance of the interval width and
        of each boundary are maintained in constant space per feature. Also tracks the
        mean within-chunk data range, which is the reference the final interval width is
        compared against: an interval far wider than the data's local variation attains
        high coverage without being informative.
        """
        s = self._width_stats[feat]
        s['n'] += 1
        n = s['n']
        for key, x in (('w', ucl - lcl), ('l', lcl), ('r', ucl)):
            mean_key, m2_key = f'{key}_mean', f'{key}_m2'
            delta = x - s[mean_key]
            s[mean_key] += delta / n
            s[m2_key] += delta * (x - s[mean_key])
        if vals:
            s['range_sum'] += (max(vals) - min(vals))
            s['range_n'] += 1

    def start_phase1(self) -> None:
        """Begin collecting in-control violation counts for Phase I calibration.

        Feed only chunks known (or assumed) to be in-control while this is active.
        Bounds keep adapting exactly as normal; only the counts are recorded.
        """
        self._phase1_active = True
        self._phase1_counts = []
        self._phase1_sizes = []

    def calibrate_phase1(self, warmup_chunks: int = 0) -> Dict[str, int]:
        """Derive per-dimension thresholds C_d from the collected Phase I counts.

        For each feature d the number of violations in an in-control chunk of size k
        is modelled as V_d ~ Binomial(k, p_d), with p_d estimated from Phase I. A chunk
        alarms when ANY of D features exceeds its threshold, so the per-feature tail
        probability is Bonferroni-corrected to alpha_chunk / D:

            C_d = F^-1_Binom(k, p_d)(1 - alpha_chunk / D) + 1

        Args:
            warmup_chunks: Number of leading Phase I chunks to discard. RBULT bounds
                expand monotonically, so the violation rate is strongly inflated while
                they converge; including that transient inflates p_d several-fold and
                yields thresholds too high to detect anything.

        Returns:
            Dict mapping feature_name -> C_d. Also stored on self.feature_thresholds.

        Raises:
            ValueError: If no Phase I chunks remain after discarding the warm-up.
        """
        counts = np.array(self._phase1_counts[warmup_chunks:], dtype=float)
        sizes = np.array(self._phase1_sizes[warmup_chunks:], dtype=float)
        if counts.size == 0:
            raise ValueError(
                f"No Phase I chunks left after discarding {warmup_chunks} warm-up chunks "
                f"(collected {len(self._phase1_counts)})."
            )

        k = int(round(sizes.mean()))
        D = max(1, len(self.features))
        p_hat = np.clip(counts.mean(axis=0) / max(1, k), 1e-9, 1 - 1e-9)
        C = binom.ppf(1.0 - self.alpha_chunk / D, k, p_hat) + 1

        self.feature_thresholds = {
            feat: int(max(1, C[i])) for i, feat in enumerate(self.features)
        }
        self._phase1_active = False
        return self.feature_thresholds

    def update_chunk(self, chunk_data: Union[pd.DataFrame, Dict[str, List[float]]],
                     ooc_threshold_count: Optional[int] = None,
                     new_sequence: bool = False) -> dict:
        """Process a streaming data chunk across all monitored features.

        Args:
            chunk_data: DataFrame or Dict mapping feature_name -> list of values.
            ooc_threshold_count: Explicit minimum number of out-of-bounds samples
                required to flag a chunk OOC. When None (default), the threshold is
                taken from Phase I calibration if available, otherwise from the rate
                rule C = ceil(chunk_alarm_rate * k).
            new_sequence: Only meaningful when difference=True. Marks this chunk as the
                start of an independent segment, so no difference is taken against the
                previous chunk. Use it when the stream is a concatenation of separate
                records rather than one continuous process.

        Returns:
            Dict containing chunk processing statistics, dynamic LCL/UCL bounds,
            OOC flags per feature, the applied threshold, total violations across
            features (``sample_ooc_count``), processing latency (ms) and RAM (KB).
        """
        start_time = time.perf_counter()
        self.total_chunks_processed += 1

        # Convert Dict to DataFrame if needed
        if isinstance(chunk_data, dict):
            df_chunk = pd.DataFrame(chunk_data)
        else:
            df_chunk = chunk_data

        if self.difference:
            df_chunk = self._apply_difference(df_chunk, new_sequence=new_sequence)

        chunk_len = len(df_chunk)
        self.total_samples_processed += chunk_len

        chunk_summary = {
            'chunk_id': self.total_chunks_processed,
            'chunk_size': chunk_len,
            'bounds': {},
            'ooc_flags': {},
            'any_ooc': False,
            'ooc_features': [],
            'sample_ooc_count': 0,
            'thresholds': {},
            'latency_ms': 0.0,
            'memory_kb': 0.0
        }
        phase1_row = []

        # Pre-pass: score this chunk against the limits carried in from chunks 1..m-1,
        # before any of them are widened by this chunk's data.
        self._score_prequential(df_chunk, chunk_len)

        # Update RBULT bounds per feature channel
        for feat in self.features:
            if feat not in df_chunk.columns:
                continue

            vals = df_chunk[feat].dropna().tolist()
            if not vals:
                continue

            eng = self.engines[feat]
            res = self.results[feat]

            # Run online bootstrap expansion
            is_expanded = eng.expand_bt_online(vals, outlier=self.outlier_filter)
            res.add_params(eng)

            lcl = eng.exp_l
            ucl = eng.exp_r

            # Check Out-Of-Control (OOC) violations in the current chunk
            ooc_mask = [(x < lcl or x > ucl) for x in vals]
            ooc_count = sum(ooc_mask)
            threshold = self.resolve_threshold(feat, chunk_len, ooc_threshold_count)
            has_ooc = ooc_count >= threshold

            chunk_summary['bounds'][feat] = {
                'lcl': lcl,
                'ucl': ucl,
                'range': eng.range,
                'expanded': is_expanded
            }
            chunk_summary['ooc_flags'][feat] = {
                'has_ooc': has_ooc,
                'ooc_count': ooc_count,
                'ooc_rate': ooc_count / max(1, len(vals))
            }
            chunk_summary['thresholds'][feat] = threshold
            chunk_summary['sample_ooc_count'] += ooc_count
            phase1_row.append(ooc_count)
            self._update_width_stats(feat, lcl, ucl, vals)

            if has_ooc:
                chunk_summary['any_ooc'] = True
                chunk_summary['ooc_features'].append(feat)

        if self._phase1_active and phase1_row:
            self._phase1_counts.append(phase1_row)
            self._phase1_sizes.append(chunk_len)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        chunk_summary['latency_ms'] = elapsed_ms
        chunk_summary['memory_kb'] = self.estimate_memory_kb()

        self.history.append(chunk_summary)
        return chunk_summary


    def get_control_limits(self) -> Dict[str, Tuple[float, float]]:
        """Get current dynamic [LCL, UCL] bounds for all features.

        Returns:
            Dict mapping feature_name -> (LCL, UCL).
        """
        return {
            feat: (self.engines[feat].exp_l, self.engines[feat].exp_r)
            for feat in self.features
        }

    def estimate_memory_kb(self) -> float:
        """Estimate current memory usage of the control chart in KB.

        Returns:
            Estimated peak RAM usage in KB (strictly O(D)).
        """
        total_bytes = sys.getsizeof(self)
        for feat in self.features:
            total_bytes += sys.getsizeof(self.engines[feat])
            total_bytes += sys.getsizeof(self.results[feat])
        return total_bytes / 1024.0

    def compute_prequential_metrics(self) -> dict:
        """One-step-ahead (prequential) coverage — the deployment-realistic protocol.

        Every chunk is scored against the limits in force before it arrived. This is what
        a control chart does in production: limits must exist before the data they judge.

        It differs from the in-sample coverage in `compute_spc_metrics()` in two ways,
        both of which make the in-sample figure optimistic:

        1. `compute_spc_metrics()` applies the FINAL limits retrospectively to the entire
           stream. Since RBULT limits only ever widen, the final interval is the widest
           the chart ever held, and early observations are judged by limits fitted to
           data that had not yet arrived.
        2. Even the per-chunk violation counts recorded in `update_chunk` are in-sample:
           the chunk widens the limits first, then is measured against the widened ones,
           so the very samples that pushed a boundary out are then scored as inside it.

        Tier 1 reports both *In-Sample Adaptation* and *One-Step-Ahead Pre-Sequential*;
        these metrics give Tier 2 the same pair.

        Returns:
            prequential_coverage_pct / prequential_far_pct: marginal, averaged over
                dimensions; prequential_joint_coverage_pct / prequential_joint_far_pct:
                all dimensions simultaneously in bounds; prequential_coverage_per_feature;
                prequential_n_samples: denominator (excludes each dimension's first chunk).
        """
        total = sum(s['total'] for s in self._preq.values())
        if total == 0:
            return {}

        covered = sum(s['covered'] for s in self._preq.values())
        cov_pct = covered / total * 100.0
        out = {
            'prequential_coverage_pct': cov_pct,
            'prequential_far_pct': 100.0 - cov_pct,
            'prequential_n_samples': total,
            'prequential_coverage_per_feature': {
                f'preq_coverage_{feat}': (s['covered'] / s['total'] if s['total'] else float('nan'))
                for feat, s in self._preq.items()
            },
        }
        if self._preq_joint['total'] > 0:
            j = self._preq_joint['covered'] / self._preq_joint['total'] * 100.0
            out['prequential_joint_coverage_pct'] = j
            out['prequential_joint_far_pct'] = 100.0 - j
        return out

    def compute_interval_metrics(self, sample_df: Optional[pd.DataFrame] = None) -> dict:
        """Interval width and boundary stability — the counterpart to coverage.

        Coverage cannot be interpreted on its own: an arbitrarily wide interval attains
        100% coverage while carrying no information, and because RBULT boundaries expand
        monotonically they inflate to absorb any non-stationarity in the stream. These
        metrics quantify the price paid for a given coverage. Tier 1 already reports
        Mean_Interval_Width and Sigma_L/Sigma_R; this is the Tier 2 equivalent.

        Args:
            sample_df: Optional full stream, used for the global reference width.

        Returns:
            mean_interval_width: Mean over dimensions of the per-chunk mean width
                (comparable to Tier 1's Mean_Interval_Width).
            final_interval_width: Mean over dimensions of the final width R_d - L_d.
            sigma_L, sigma_R: Mean over dimensions of the standard deviation of each
                boundary across chunks — boundary stability, lower is more stable.
            width_ratio_local: Final width divided by the mean within-chunk data range,
                averaged over dimensions. ~1 means the interval tracks local variation;
                >> 1 means it is inflated well beyond it, so high coverage is cheap.
            width_ratio_global: Final width divided by the 0.5-99.5 percentile span of
                the whole stream. ~1 means the interval has converged to the empirical
                support, i.e. to what a full-history percentile baseline would compute.
            interval_width_per_feature: Per-dimension breakdown.
        """
        per_feat, w_means, w_finals, l_sds, r_sds, local_ratios, global_ratios = {}, [], [], [], [], [], []
        bounds = self.get_control_limits()

        for feat in self.features:
            s = self._width_stats[feat]
            if s['n'] == 0:
                continue
            lcl, ucl = bounds[feat]
            final_w = ucl - lcl
            denom = max(1, s['n'] - 1)
            l_sd = float(np.sqrt(s['l_m2'] / denom))
            r_sd = float(np.sqrt(s['r_m2'] / denom))
            local = s['range_sum'] / s['range_n'] if s['range_n'] else float('nan')

            entry = {'mean_width': s['w_mean'], 'final_width': final_w,
                     'sigma_L': l_sd, 'sigma_R': r_sd, 'mean_chunk_range': local}
            if local and local > 0 and np.isfinite(local):
                entry['width_ratio_local'] = final_w / local
                local_ratios.append(entry['width_ratio_local'])

            if sample_df is not None and feat in sample_df.columns:
                v = sample_df[feat].dropna().values
                if len(v) > 1:
                    p_lo, p_hi = np.percentile(v, [0.5, 99.5])
                    span = p_hi - p_lo
                    if span > 0:
                        entry['width_ratio_global'] = final_w / span
                        global_ratios.append(entry['width_ratio_global'])

            per_feat[feat] = entry
            w_means.append(s['w_mean'])
            w_finals.append(final_w)
            l_sds.append(l_sd)
            r_sds.append(r_sd)

        if not per_feat:
            return {}

        out = {
            'mean_interval_width': float(np.mean(w_means)),
            'final_interval_width': float(np.mean(w_finals)),
            'sigma_L': float(np.mean(l_sds)),
            'sigma_R': float(np.mean(r_sds)),
            'interval_width_per_feature': per_feat,
        }
        if local_ratios:
            out['width_ratio_local'] = float(np.mean(local_ratios))
            out['width_ratio_local_max'] = float(np.max(local_ratios))
        if global_ratios:
            out['width_ratio_global'] = float(np.mean(global_ratios))
        return out

    def compute_spc_metrics(self, true_labels: Optional[List[int]] = None,
                            sample_df: Optional[pd.DataFrame] = None) -> dict:
        """Compute comprehensive Statistical Process Control metrics.

        Args:
            true_labels: Optional list of true state labels (0=In-Control, 1=Out-Of-Control)
                         per chunk.
            sample_df: Optional full streaming DataFrame for sample-level Coverage and ARL evaluation.

        Returns:
            Dict of SPC metrics: total_chunks, total_samples, avg_latency_ms,
            peak_memory_kb, ooc_chunk_rate, overall_coverage_pct (marginal),
            joint_coverage_pct, arl_0, arl_1, and FAR.

        Notes on Coverage Rate:
            - overall_coverage_pct: Marginal coverage averaged across dimensions.
              Corresponds to per-dimension Bonferroni-corrected alpha_dim target.
              Formula: (sum_d sum_t 1[x_{t,d} in [L_d, R_d]]) / (D * N)
            - joint_coverage_pct: Joint hyper-rectangle coverage.
              An observation is in-control only when ALL D dimensions are within
              their respective bounds simultaneously.
              Formula: (sum_t 1[forall d: x_{t,d} in [L_d, R_d]]) / N
              Guaranteed by Bonferroni to be >= 1 - alpha_sys.
        """
        total_ooc = sum(1 for h in self.history if h['any_ooc'])
        avg_latency = np.mean([h['latency_ms'] for h in self.history]) if self.history else 0.0

        metrics = {
            'total_chunks': self.total_chunks_processed,
            'total_samples': self.total_samples_processed,
            'avg_latency_ms': avg_latency,
            'peak_memory_kb': self.estimate_memory_kb(),
            'total_ooc_chunks': total_ooc,
            'ooc_chunk_rate': total_ooc / max(1, self.total_chunks_processed),
            'fwer_adjusted_alpha_dim': self.alpha_dim,
            'chunk_alarm_rule': ('phase1_binomial' if self.feature_thresholds is not None
                                 else f'rate_{self.chunk_alarm_rate:g}k'),
            'chunk_alarm_rate': self.chunk_alarm_rate
        }

        metrics.update(self.compute_interval_metrics(sample_df=sample_df))
        metrics.update(self.compute_prequential_metrics())

        # Calculate sample-level coverage rate if sample_df is provided
        if sample_df is not None:
            feature_coverage = {}
            total_in_control_samples = 0
            total_covered_samples = 0

            # Get final or average bounds per feature
            bounds = self.get_control_limits()
            for feat in self.features:
                if feat in sample_df.columns:
                    vals = sample_df[feat].dropna().values
                    lcl, ucl = bounds[feat]
                    covered = np.sum((vals >= lcl) & (vals <= ucl))
                    total_in_control_samples += len(vals)
                    total_covered_samples += covered
                    feature_coverage[f'coverage_{feat}'] = covered / max(1, len(vals))

            metrics['overall_coverage_pct'] = (total_covered_samples / max(1, total_in_control_samples)) * 100.0
            metrics['sample_far_pct'] = 100.0 - metrics['overall_coverage_pct']
            metrics['feature_coverage'] = feature_coverage

            # Joint hyper-rectangle coverage: in-control iff ALL dimensions within bounds simultaneously
            n_samples = len(sample_df)
            if n_samples > 0:
                in_bounds_per_dim = np.stack([
                    (sample_df[feat].values >= bounds[feat][0]) & (sample_df[feat].values <= bounds[feat][1])
                    for feat in self.features
                    if feat in sample_df.columns
                ], axis=0)  # shape: (D, N)
                joint_covered = int(np.sum(np.all(in_bounds_per_dim, axis=0)))
                metrics['joint_coverage_pct'] = (joint_covered / n_samples) * 100.0
                metrics['joint_far_pct'] = 100.0 - metrics['joint_coverage_pct']
            else:
                metrics['joint_coverage_pct'] = 0.0
                metrics['joint_far_pct'] = 100.0


        # Compute ARL0 (In-Control Run Length) & ARL1 (Shift Detection Delay) if true_labels are provided
        if true_labels and len(true_labels) == len(self.history):
            false_alarms = sum(
                1 for h, label in zip(self.history, true_labels)
                if h['any_ooc'] and label == 0
            )
            in_control_chunks = sum(1 for label in true_labels if label == 0)
            # NaN, not 0, when there is nothing to false-alarm on: a 0/0 rate is undefined
            # and reporting it as 0.00 reads as flawless in-control behaviour.
            metrics['false_alarm_rate'] = (false_alarms / in_control_chunks
                                          if in_control_chunks > 0 else float('nan'))

            # ARL0: Average run length between false alarms during in-control periods
            in_control_run_lengths = []
            current_run = 0
            for h, label in zip(self.history, true_labels):
                if label == 0:  # In-control chunk
                    if h['any_ooc']:  # False Alarm
                        in_control_run_lengths.append(current_run)
                        current_run = 0
                    else:
                        current_run += 1
            if current_run > 0:
                in_control_run_lengths.append(current_run)

            # With no false alarm, ARL0 is right-censored at the observation window, so
            # report the in-control chunk count as a lower bound -- and NaN when there is
            # no in-control data at all.
            if in_control_chunks == 0:
                metrics['arl_0'] = float('nan')
            else:
                metrics['arl_0'] = (float(np.mean(in_control_run_lengths))
                                    if in_control_run_lengths else float(in_control_chunks))
            metrics['arl_0_censored'] = not bool(in_control_run_lengths)

            # ARL1: Average detection delay (chunks) from actual failure onset to alarm
            ooc_detection_delays = []
            detecting = False
            delay = 0
            for h, label in zip(self.history, true_labels):
                if label == 1:  # Actual Out-Of-Control chunk
                    delay += 1
                    if h['any_ooc']:  # Successfully detected
                        ooc_detection_delays.append(delay)
                        delay = 0
                else:
                    delay = 0

            # NaN, not 1.0, when nothing was ever detected. The old fallback was
            # indistinguishable from instant detection, the best possible score.
            metrics['arl_1'] = (float(np.mean(ooc_detection_delays))
                                if ooc_detection_delays else float('nan'))
            metrics['n_detected_episodes'] = len(ooc_detection_delays)

        return metrics

