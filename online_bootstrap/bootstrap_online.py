"""
Online Bootstrap Module (Modular Version)
==========================================

Refactored from boot_stream.py for improved readability and maintainability.
All logic and behavior are preserved exactly as in the original boot_stream.booststream class.

Class:
    BootstrapOnline — Online bootstrap engine with modular method decomposition.

Methods (Public):
    set_online()          — Configure for online bootstrap mode
    compute_error()       — Compute error vs target boundaries
    update_center_range() — Update center (avg) and spread (std)
    expand_bt_online()    — Run online bootstrap on a new data chunk (orchestrator)
    expand_bt_trad()      — Run traditional (offline) bootstrap
    expand_whole()        — Run traditional bootstrap for mean/std estimation

Methods (Private):
    _validate_online_mode()     — Check online mode is active
    _update_sample_count()      — Update total_size and chunk_size
    _apply_outlier_detection()  — Clean outliers via Z-score
    _update_global_minmax()     — Track global min/max across chunks
    _try_expand_left()          — Expand left boundary
    _try_expand_right()         — Expand right boundary
    _compute_histogram()        — Compute data and theoretical histograms
    _recompute_bins()           — Recompute histogram bins (inside expansion loop)
    _run_expansion_loop()       — Iterative bootstrap expansion until convergence
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from online_bootstrap import bootstrap_v1, BatchOutlierDetection
import math, copy, statistics
import numpy as np


@dataclass
class BootstrapOnline:
    """Online Bootstrap engine for streaming data.

    Supports three modes:
    - Online: Process data chunk-by-chunk, expanding boundaries incrementally
    - Online with Min-Max: Online with min-max bootstrap for boundary estimation
    - Traditional: Standard bootstrap on accumulated data

    Attributes:
        online: Whether online mode is active.
        online_cum: Whether cumulative mode is active (replace total_size instead of accumulate).
        minmax_boost: Whether min-max bootstrap is used for boundary expansion.
        numbin: Number of theoretical histogram bins (default: 8).
        number_bt_iter: Number of bootstrap iterations (default: 600).
        nboost: Minimum number of data points required for bootstrap (default: 3).
        dist_list: List of distribution names for theoretical histogram fitting.
        total_size: Total number of learned data points.
        chunk_size: Size of the current data chunk.
        min_chs: Global minimum value across all chunks.
        max_chs: Global maximum value across all chunks.
        min_list: Data points in the leftmost histogram bin.
        max_list: Data points in the rightmost histogram bin.
        avg: List of center values (mean of exp_l and exp_r) per epoch.
        std: List of spread values (range / 8) per epoch.
        exp_l: Left boundary (expand left).
        exp_r: Right boundary (expand right).
        range: Current range (exp_r - exp_l).
        flag_learning: Whether learning has been triggered.
        nlearn_l: List of left-side learning counts per epoch.
        nlearn_r: List of right-side learning counts per epoch.
    """

    # ------------------------------------------------------------------ #
    #                           Configuration                              #
    # ------------------------------------------------------------------ #
    online_cum: bool = False
    online: bool = False
    minmax_boost: bool = False
    filesampl: str = ''
    numbin: int = 0
    number_bt_iter: int = 600
    nboost: int = 0
    dist_list: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #                             State                                    #
    # ------------------------------------------------------------------ #
    total_size: int = 0
    chunk_size: int = 0
    max_chs: float = -9999.99
    min_chs: float = 9999.99
    min_list: List[float] = field(default_factory=list)
    max_list: List[float] = field(default_factory=list)
    avg: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    exp_l: float = 9999.99
    exp_r: float = -9999.99
    range: float = 0.0
    flag_learning: bool = False
    nlearn_l: List[int] = field(default_factory=list)
    nlearn_r: List[int] = field(default_factory=list)

    # ================================================================== #
    #                       Configuration Methods                          #
    # ================================================================== #

    def set_online(self, minmax_flag: bool = False) -> None:
        """Configure the engine for online bootstrap mode.

        Args:
            minmax_flag: If True, use min-max bootstrap for boundary estimation.
                         If False, use raw chunk min/max for boundaries.
        """
        self.online = True
        self.minmax_boost = minmax_flag
        self.numbin = 8
        self.dist_list = [
            'exponweib', 'wald', 'gamma', 'norm',
            'expon', 'powerlaw', 'lognorm', 'chi2',
            'weibull_min', 'weibull_max'
        ]
        self.nboost = 3

    # ================================================================== #
    #                         Public Methods                               #
    # ================================================================== #

    def compute_error(self, target_l: float, target_r: float) -> Tuple[float, float, float]:
        """Compute error between current boundaries and target values.

        Args:
            target_l: Target left boundary (e.g., population minimum).
            target_r: Target right boundary (e.g., population maximum).

        Returns:
            Tuple of (left_error, right_error, range_error).
        """
        target_range = target_r - target_l
        return (target_l - self.exp_l), (target_r - self.exp_r), (target_range - self.range)

    def update_center_range(self, leftmost: float, rightmost: float) -> None:
        """Update center (avg) and spread (std) from boundary values.

        Appends new values to the avg and std history lists.
        Center = midpoint of boundaries, Spread = range / 8.

        Args:
            leftmost: Left boundary value.
            rightmost: Right boundary value.
        """
        self.avg.append((rightmost + leftmost) / 2)
        self.std.append((rightmost - leftmost) / 8)

    def expand_bt_online(self, new_data_chunk: list, outlier: bool = False,
                         cum: bool = False, cum_left_right: bool = False) -> bool:
        """Run online bootstrap on a new data chunk.

        Main orchestrator method that executes the full pipeline:
            1. Validate online mode
            2. Update sample counts
            3. Apply outlier detection (optional)
            4. Update global min/max
            5. Try expanding left/right boundaries
            6. Compute histogram and run iterative expansion loop

        Args:
            new_data_chunk: List of new data values.
            outlier: If True, apply Z-score outlier detection before processing.
            cum: If True, replace total_size (cumulative mode) instead of accumulating.
            cum_left_right: If True, include previous min/max lists in histogram computation.

        Returns:
            True if any boundary expansion occurred, False otherwise.
        """
        # Step 1: Validate online mode
        if not self._validate_online_mode():
            return False

        # Step 2: Update sample counts
        self._update_sample_count(new_data_chunk, cum)

        # Step 3: Outlier detection (optional)
        if outlier:
            new_data_chunk = self._apply_outlier_detection(new_data_chunk)

        # Step 4: Compute chunk min/max and update global tracking
        chunk_min = min(new_data_chunk)
        chunk_max = max(new_data_chunk)
        self._update_global_minmax(chunk_min, chunk_max)

        # Step 5: Try expanding boundaries
        expand_min = self._try_expand_left(chunk_min)
        expand_max = self._try_expand_right(chunk_max)

        # Step 6: If any boundary changed → compute histogram & run expansion loop
        expansion = False
        if expand_min or expand_max:
            # Augment data with previous tail data if cumulative left-right mode
            if cum_left_right:
                new_data_chunk = new_data_chunk + self.min_list + self.max_list

            hist_data, hist_theo = self._compute_histogram(new_data_chunk)
            expansion = self._run_expansion_loop(new_data_chunk, hist_data, hist_theo)

        # Step 7: Finalize — update range if any expansion occurred
        if expansion or expand_max or expand_min:
            self.range = self.exp_r - self.exp_l
            expansion = True

        return expansion

    def expand_bt_trad(self, input_data: list) -> None:
        """Run traditional (offline) bootstrap on accumulated data.

        Resamples the entire dataset with replacement for number_bt_iter
        iterations, then estimates boundaries as the mean of bootstrap
        min/max values.

        Args:
            input_data: Complete accumulated dataset.
        """
        try:
            if self.online is True:
                raise ValueError("The network in online mode. Can not perform whole mode.")
        except ValueError as e:
            return print(f"Error: {e}")

        self.number_bt_iter = 600
        data_set = copy.deepcopy(input_data)
        nsample = len(data_set)
        bootstrap_min = []
        bootstrap_max = []
        size_boost = len(data_set)

        # Create bootstrap samples and collect min/max from each
        bootstrap_sample_list = [
            list(np.random.choice(data_set, size_boost, replace=True))
            for _ in range(self.number_bt_iter)
        ]
        for samples in bootstrap_sample_list:
            bootstrap_min.append(np.min(samples))
            bootstrap_max.append(np.max(samples))

        # Update state with bootstrap estimates
        self.chunk_size = size_boost
        self.max_chs = np.max(data_set)
        self.min_chs = np.min(data_set)
        self.exp_l = np.mean(bootstrap_min)
        self.exp_r = np.mean(bootstrap_max)
        self.range = self.exp_r - self.exp_l
        self.nlearn_l.append(nsample)
        self.nlearn_r.append(nsample)

    def expand_whole(self, input_data: list) -> None:
        """Run traditional bootstrap for mean and standard deviation estimation.

        Uses an incremental averaging approach where each bootstrap mean is
        averaged with the previous one. Computes estimated mean and std,
        and their differences from the input statistics.

        Args:
            input_data: Complete dataset for bootstrap estimation.
        """
        try:
            if self.online is True:
                raise ValueError("The network in online mode. Can not perform whole mode.")
        except ValueError as e:
            return print(f"Error: {e}")

        data_set = copy.deepcopy(input_data)
        bootstrap_means = []
        bootstrap_std = []
        size_boost = len(data_set)
        input_mean = np.mean(data_set)

        # Create bootstrap samples
        bootstrap_sample_list = [
            list(np.random.choice(data_set, size_boost, replace=True))
            for _ in range(self.number_bt_iter)
        ]

        # Compute incremental bootstrap means
        for idx, samples in enumerate(bootstrap_sample_list):
            if idx == 0:
                bootstrap_means.append(np.mean(samples))
                previous_bootstrap_mean = bootstrap_means[0]
            else:
                bootstrap_means.append(
                    0.5 * (np.mean(samples) + previous_bootstrap_mean)
                )
                previous_bootstrap_mean = bootstrap_means[-1]

        # Compute bootstrap mean estimation and difference
        estimated_mean = np.mean(bootstrap_means)
        different_input_mean_bootstrap_mean = abs(input_mean - estimated_mean)

        # Compute bootstrap std estimation and difference
        est_mean_list = list(estimated_mean) * size_boost
        for data in bootstrap_sample_list:
            variance = list(map(lambda a, b: (a - b) ** 2, data, est_mean_list))
            std_val = math.sqrt(sum(variance) / (size_boost - 1))
            bootstrap_std.append(std_val)
        estimated_std = np.mean(bootstrap_std)
        different_input_std_bootstrap_std = abs(statistics.stdev(data_set) - estimated_std)

    # ================================================================== #
    #                       Private Helper Methods                         #
    # ================================================================== #

    def _validate_online_mode(self) -> bool:
        """Check if the engine is configured for online mode.

        Returns:
            True if online mode is active, False otherwise (prints error).
        """
        if self.online is False:
            print("Error: The network in traditional mode. Can not perform online mode.")
            return False
        return True

    def _update_sample_count(self, new_data_chunk: list, cum: bool) -> None:
        """Update total sample count and chunk size.

        Args:
            new_data_chunk: Current data chunk (used for len()).
            cum: If True, replace total_size; if False, accumulate.
        """
        if not cum:
            self.total_size += len(new_data_chunk)
        else:
            self.total_size = len(new_data_chunk)
        self.chunk_size = len(new_data_chunk)

    def _apply_outlier_detection(self, new_data_chunk: list) -> list:
        """Apply Z-score based outlier detection to clean the data chunk.

        Uses running avg/std for detection. On the first chunk (when avg/std
        lists are empty), initializes them from the data.

        Args:
            new_data_chunk: Raw data chunk.

        Returns:
            Cleaned data chunk with outliers removed.
        """
        # Initialize avg/std from data if this is the first chunk
        if self.avg == []:
            self.avg.append(statistics.mean(new_data_chunk))
        if self.std == []:
            self.std.append(statistics.stdev(new_data_chunk))

        detector = BatchOutlierDetection.ZBatchOutlierDetector()
        detector.add_init_params(threshold=3.0, mean=self.avg[-1], sd=self.std[-1])
        return detector.get_clean_data(new_data_chunk)

    def _update_global_minmax(self, chunk_min: float, chunk_max: float) -> None:
        """Update global minimum and maximum values across all chunks.

        Args:
            chunk_min: Minimum value of current chunk.
            chunk_max: Maximum value of current chunk.
        """
        if chunk_min < self.min_chs:
            self.min_chs = chunk_min
        if chunk_max > self.max_chs:
            self.max_chs = chunk_max

    def _try_expand_left(self, chunk_min: float) -> bool:
        """Try to expand the left boundary based on the chunk minimum.

        If minmax_boost is True and enough data points exist in min_list,
        uses bootstrap to estimate the new boundary. Otherwise, directly
        sets the boundary to chunk_min.

        Args:
            chunk_min: Minimum value of current chunk.

        Returns:
            True if the left boundary was expanded, False otherwise.
        """
        if chunk_min < self.exp_l:
            if len(self.min_list) >= self.nboost and self.minmax_boost is True:
                self.min_list.append(chunk_min)
                adjust_left_std = bootstrap_v1.bootstrap_online(
                    self.min_list, "left",
                    number_bootstrap_iteration=self.number_bt_iter,
                    minmax_boost=self.minmax_boost,
                    prob=False
                )
                if self.exp_l >= adjust_left_std:
                    self.exp_l = adjust_left_std
            else:
                self.exp_l = chunk_min
            return True
        return False

    def _try_expand_right(self, chunk_max: float) -> bool:
        """Try to expand the right boundary based on the chunk maximum.

        If minmax_boost is True and enough data points exist in max_list,
        uses bootstrap to estimate the new boundary. Otherwise, directly
        sets the boundary to chunk_max.

        Args:
            chunk_max: Maximum value of current chunk.

        Returns:
            True if the right boundary was expanded, False otherwise.
        """
        if chunk_max > self.exp_r:
            if len(self.max_list) >= self.nboost and self.minmax_boost is True:
                self.max_list.append(chunk_max)
                adjust_right_std = bootstrap_v1.bootstrap_online(
                    self.max_list, "right",
                    number_bootstrap_iteration=self.number_bt_iter,
                    minmax_boost=self.minmax_boost,
                    prob=False
                )
                if self.exp_r <= adjust_right_std:
                    self.exp_r = adjust_right_std
            else:
                self.exp_r = chunk_max
            return True
        return False

    def _compute_histogram(self, data_chunk: list) -> Tuple[List[int], List[int]]:
        """Compute data histogram and theoretical histogram.

        Updates center range, bins data into histogram based on +/-sigma ranges,
        and updates self.min_list and self.max_list with tail bin data.

        Bin layout (8 bins):
            bin[0]  : data in [-4σ, -3σ]  (leftmost tail)
            bin[1]  : data in [-3σ, -2σ]
            bin[2-5]: data in [-2σ, +2σ]   (not explicitly counted)
            bin[-2] : data in [+2σ, +3σ]
            bin[-1] : data in [+3σ, +4σ]  (rightmost tail)

        Args:
            data_chunk: Current data chunk.

        Returns:
            Tuple of (hist_data, hist_theo) — observed and theoretical histograms.
        """
        # Update center and spread from current boundaries
        self.update_center_range(self.exp_l, self.exp_r)
        avg = self.avg[-1]
        std = self.std[-1]

        # Bin data into tail bins and update min_list / max_list
        self.min_list = [k for k in data_chunk if (avg - 4 * std <= k <= avg - 3 * std)]
        self.max_list = [k for k in data_chunk if (avg + 3 * std <= k <= avg + 4 * std)]

        # Build observed histogram
        hist_data = [0] * int(self.numbin)
        hist_data[0] = len(self.min_list)
        hist_data[-1] = len(self.max_list)
        hist_data[1] = len([i for i in data_chunk if (avg - 3 * std <= i <= avg - 2 * std)])
        hist_data[-2] = len([i for i in data_chunk if (avg + 2 * std <= i <= avg + 3 * std)])

        # Build theoretical histogram from best-fit distribution
        percent_data = bootstrap_v1.get_percent_std_data_from_best_distribution(
            self.total_size, self.min_list, self.max_list, self.dist_list
        )
        hist_theo = [math.ceil(i * self.total_size / 100.0) for i in percent_data]

        return hist_data, hist_theo

    def _recompute_bins(self, data_chunk: list) -> List[int]:
        """Recompute histogram bins after boundary expansion (inside the while loop).

        Similar to _compute_histogram but does NOT recompute hist_theo.
        Updates center range, min_list, max_list, and returns new hist_data.

        Args:
            data_chunk: Current data chunk.

        Returns:
            Updated hist_data (observed histogram).
        """
        self.update_center_range(self.exp_l, self.exp_r)
        avg = self.avg[-1]
        std = self.std[-1]

        end_bin_left = [i for i in data_chunk if (avg - 4 * std <= i <= avg - 3 * std)]
        end_bin_right = [i for i in data_chunk if (avg + 3 * std <= i <= avg + 4 * std)]

        hist_data = [0] * int(self.numbin)
        hist_data[0] = len(end_bin_left)
        hist_data[-1] = len(end_bin_right)
        hist_data[1] = len([i for i in data_chunk if (avg - 3 * std <= i <= avg - 2 * std)])
        hist_data[-2] = len([i for i in data_chunk if (avg + 2 * std <= i <= avg + 3 * std)])

        self.min_list = end_bin_left
        self.max_list = end_bin_right

        return hist_data

    def _run_expansion_loop(self, data_chunk: list, hist_data: List[int],
                            hist_theo: List[int]) -> bool:
        """Run the iterative boundary expansion loop.

        Compares observed vs theoretical histograms at both tails. If the
        observed count exceeds the theoretical count, uses bootstrap to
        expand boundaries iteratively until convergence (boundaries stop changing).

        Args:
            data_chunk: Current data chunk.
            hist_data: Observed histogram from _compute_histogram().
            hist_theo: Theoretical histogram from _compute_histogram().

        Returns:
            True if any expansion occurred during the loop, False otherwise.
        """
        expand = False
        expansion = False

        # Compute differences at tail bins
        difference_max = hist_data[-1] - hist_theo[-1]
        difference_min = hist_data[0] - hist_theo[0]

        # Check if expansion is needed and track learning counts
        if difference_max > 0 or difference_min > 0:
            dif_expand = True
            self.nlearn_r.append(hist_data[-1] if difference_max > 0 else 0)
            self.nlearn_l.append(hist_data[0] if difference_min > 0 else 0)
        else:
            dif_expand = False

        # -------------------------------------------------------------- #
        #  Iterative expansion: bootstrap boundaries until convergence     #
        # -------------------------------------------------------------- #
        while dif_expand is True:
            expandL = self.exp_l
            expandR = self.exp_r

            # --- Right-side expansion ---
            if difference_max > 0:
                if hist_data[-1] >= self.nboost:
                    # First attempt: bootstrap if not yet learning
                    if self.flag_learning is False:
                        tmp_exp_r = bootstrap_v1.bootstrap_online(
                            self.max_list, "right",
                            number_bootstrap_iteration=self.number_bt_iter,
                            minmax_boost=self.minmax_boost,
                            prob=False
                        )
                        if tmp_exp_r > expandR:
                            self.exp_r = tmp_exp_r
                            expand = True
                            expansion = True

                    # Correction: if boundary is still within data, re-bootstrap
                    if self.exp_r <= max(self.max_list):
                        self.exp_r = bootstrap_v1.bootstrap_online(
                            self.max_list, "right",
                            number_bootstrap_iteration=self.number_bt_iter,
                            minmax_boost=self.minmax_boost,
                            prob=False
                        )
                        # Prevent shrinking
                        if self.exp_r < expandR:
                            self.exp_r = expandR
                        expand = True
                        expansion = True
            else:
                expand = False

            # --- Left-side expansion ---
            if difference_min > 0:
                if hist_data[0] >= self.nboost:
                    # First attempt: bootstrap if not yet learning
                    if self.flag_learning is False:
                        self.flag_learning = True
                        tmp_exp_l = bootstrap_v1.bootstrap_online(
                            self.min_list, "left",
                            number_bootstrap_iteration=self.number_bt_iter,
                            minmax_boost=self.minmax_boost,
                            prob=False
                        )
                        if tmp_exp_l < expandL:
                            self.exp_l = tmp_exp_l
                            expand = True
                            expansion = True

                    # Correction: if boundary is still within data, re-bootstrap
                    if self.exp_l >= min(self.min_list):
                        self.exp_l = bootstrap_v1.bootstrap_online(
                            self.min_list, "left",
                            number_bootstrap_iteration=self.number_bt_iter,
                            minmax_boost=self.minmax_boost,
                            prob=False
                        )
                        # Prevent expanding in wrong direction
                        if self.exp_l > expandL:
                            self.exp_l = expandL
                        expand = True
                        expansion = True

            # --- Recompute bins if any expansion occurred ---
            if expand is True:
                hist_data = self._recompute_bins(data_chunk)
                expand = False

                # Check convergence: did boundaries actually change?
                if expandL == self.exp_l and expandR == self.exp_r:
                    dif_expand = False
                else:
                    # Recompute differences for next iteration
                    difference_max = hist_data[-1] - hist_theo[-1]
                    difference_min = hist_data[0] - hist_theo[0]
            else:
                dif_expand = False

        return expansion
