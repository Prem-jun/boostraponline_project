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
                 chunk_alarm_rate: float = 0.05, alpha_chunk: float = 0.05):
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
        """
        self.features = features
        self.minmax_flag = minmax_flag
        self.outlier_filter = outlier_filter
        self.alpha_sys = alpha_sys
        self.fwer_correction = fwer_correction
        self.chunk_alarm_rate = chunk_alarm_rate
        self.alpha_chunk = alpha_chunk

        # Phase I state (optional per-dimension binomial calibration)
        self.feature_thresholds: Optional[Dict[str, int]] = None
        self._phase1_active = False
        self._phase1_counts: List[List[int]] = []
        self._phase1_sizes: List[int] = []

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
                     ooc_threshold_count: Optional[int] = None) -> dict:
        """Process a streaming data chunk across all monitored features.

        Args:
            chunk_data: DataFrame or Dict mapping feature_name -> list of values.
            ooc_threshold_count: Explicit minimum number of out-of-bounds samples
                required to flag a chunk OOC. When None (default), the threshold is
                taken from Phase I calibration if available, otherwise from the rate
                rule C = ceil(chunk_alarm_rate * k).

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
            metrics['false_alarm_rate'] = false_alarms / max(1, in_control_chunks)

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

            metrics['arl_0'] = float(np.mean(in_control_run_lengths)) if in_control_run_lengths else float(in_control_chunks)

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

            metrics['arl_1'] = float(np.mean(ooc_detection_delays)) if ooc_detection_delays else 1.0

        return metrics

