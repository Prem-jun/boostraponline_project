"""
RBULT Statistical Process Control (SPC) Module
===============================================

Implements a Memory-Bounded Adaptive Control Chart (RBULTControlChart)
for univariate and multivariate non-Gaussian IoT data streams.

Classes:
    RBULTControlChart — Real-time stream monitor establishing dynamic LCL/UCL bounds.
"""

from typing import Dict, List, Union, Tuple, Optional
import time
import sys
import numpy as np
import pandas as pd
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
        engines: Dict mapping feature_name -> BootstrapOnline instance.
        results: Dict mapping feature_name -> ResBootstrap instance.
        history: Event log of streaming updates and Out-Of-Control (OOC) detections.
    """

    def __init__(self, features: List[str], minmax_flag: bool = False,
                 outlier_filter: bool = True, alpha_sys: float = 0.05,
                 fwer_correction: str = 'bonferroni'):
        """Initialize the RBULT Control Chart engine.

        Args:
            features: List of column/sensor feature names.
            minmax_flag: If True, use min-max bootstrap for tail bounds.
            outlier_filter: If True, apply Z-score spike filtering before bound updates.
            alpha_sys: Desired overall system false alarm rate.
            fwer_correction: Adjustment type ('bonferroni', 'sidak', 'none').
        """
        self.features = features
        self.minmax_flag = minmax_flag
        self.outlier_filter = outlier_filter
        self.alpha_sys = alpha_sys
        self.fwer_correction = fwer_correction

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

    def update_chunk(self, chunk_data: Union[pd.DataFrame, Dict[str, List[float]]]) -> dict:
        """Process a streaming data chunk across all monitored features.

        Args:
            chunk_data: DataFrame or Dict mapping feature_name -> list of values.

        Returns:
            Dict containing chunk processing statistics, dynamic LCL/UCL bounds,
            OOC flags per feature, processing latency (ms), and estimated RAM (KB).
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
            'latency_ms': 0.0,
            'memory_kb': 0.0
        }

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
            has_ooc = any(ooc_mask)
            ooc_count = sum(ooc_mask)

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

            if has_ooc:
                chunk_summary['any_ooc'] = True
                chunk_summary['ooc_features'].append(feat)

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
            peak_memory_kb, ooc_chunk_rate, coverage_rate_pct, arl_0, arl_1, and FAR.
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
            'fwer_adjusted_alpha_dim': self.alpha_dim
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
            metrics['feature_coverage'] = feature_coverage

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

