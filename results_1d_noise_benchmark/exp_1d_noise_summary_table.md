# 1D Synthetic Benchmark Results: 7-Scenario Gold Standard Suite

This report presents the 1D Synthetic Benchmark results evaluated under **both protocols** across 7 Noise & Outlier Scenarios:
1. **Scenario A (Clean Stream)**: Baseline pure stream without noise.
2. **Scenario B1 (GAWN Noise 0.1\sigma)**: Light continuous Gaussian noise.
3. **Scenario B2 (GAWN Noise 0.2\sigma)**: Moderate continuous Gaussian noise.
4. **Scenario B3 (GAWN Noise 0.3\sigma)**: Heavy continuous Gaussian noise.
5. **Scenario C1 (Impulse Spikes 1%)**: Light outlier spike contamination.
6. **Scenario C2 (Impulse Spikes 5%)**: Severe outlier spike contamination.
7. **Scenario C3 (Impulse Spikes 10%)**: Extreme stress test outlier contamination.

Subset: Chunk Size = 100, Target Alpha = 0.05 (Target Coverage = 95.00%, Target Tail FAR = 2.50%)

## Protocol: In-Sample Adaptation

| Distribution | Noise_Scenario | Eval_Mode | Chunk_Size | Target_Alpha | Target_Coverage_Pct | Target_Tail_FAR_Pct | Method | Empirical_Coverage_Pct | Left_FAR_Pct | Right_FAR_Pct | Mean_Interval_Width | Sigma_L_Stability | Sigma_R_Stability | Noise_Sensitivity_Ratio_NSR | Latency_per_Chunk_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-Distribution | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.96 | 2.57 | 2.47 | 4.0433 | 0.0059 | 0.1148 | 1.0 | 0.3664 |
| F-Distribution | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 11.1253 | 0.0056 | 1.5164 | 1.0 | 8.035 |
| F-Distribution | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 98.12 | 0.04 | 1.84 | 4.5687 | 0.0 | 0.0 | 1.0003 | 2.0551 |
| F-Distribution | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.9 | 2.63 | 2.47 | 4.1012 | 0.0058 | 0.1077 | 1.0143 | 0.358 |
| F-Distribution | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 11.3751 | 0.005 | 1.4683 | 1.0225 | 6.8802 |
| F-Distribution | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 98.16 | 0.01 | 1.83 | 4.83 | 0.0 | 0.0 | 1.0575 | 2.0332 |
| F-Distribution | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.8 | 2.68 | 2.52 | 4.211 | 0.0143 | 0.1137 | 1.0415 | 0.3717 |
| F-Distribution | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 11.6996 | 0.0604 | 1.4161 | 1.0516 | 7.1815 |
| F-Distribution | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 98.54 | 0.02 | 1.44 | 5.3975 | 0.0 | 0.0 | 1.1817 | 1.9795 |
| F-Distribution | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.81 | 2.65 | 2.54 | 4.3254 | 0.0193 | 0.1283 | 1.0698 | 0.3733 |
| F-Distribution | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 12.0382 | 0.1047 | 1.3596 | 1.0821 | 9.0925 |
| F-Distribution | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 98.4 | 0.08 | 1.52 | 5.5904 | 0.0 | 0.0 | 1.2239 | 2.0504 |
| F-Distribution | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.94 | 2.6 | 2.46 | 4.3444 | 0.009 | 0.1692 | 1.0745 | 0.3608 |
| F-Distribution | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 16.1457 | 0.7552 | 1.5164 | 1.4513 | 11.6598 |
| F-Distribution | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.61 | 0.53 | 1.86 | 5.0128 | 0.0 | 0.0 | 1.0975 | 2.0147 |
| F-Distribution | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.14 | 2.4 | 2.46 | 9.1071 | 1.0518 | 0.1391 | 2.2524 | 0.3895 |
| F-Distribution | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 16.4254 | 0.0373 | 1.5164 | 1.4764 | 11.1428 |
| F-Distribution | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.56 | 0.19 | 0.25 | 12.9435 | 0.0 | 0.0 | 2.8338 | 2.0321 |
| F-Distribution | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.06 | 2.34 | 2.6 | 11.0992 | 0.108 | 0.1541 | 2.7451 | 0.3662 |
| F-Distribution | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 16.4457 | 0.0483 | 1.4952 | 1.4782 | 11.1908 |
| F-Distribution | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.56 | 0.34 | 0.1 | 14.2985 | 0.0 | 0.0 | 3.1305 | 2.1916 |
| Uniform | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.01 | 2.55 | 2.44 | 94.7299 | 0.0877 | 0.1823 | 1.0 | 0.3476 |
| Uniform | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 99.8431 | 0.1883 | 0.2265 | 0.9997 | 14.9968 |
| Uniform | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 98.47 | 0.44 | 1.09 | 98.1558 | 0.0 | 0.0 | 0.9998 | 2.0486 |
| Uniform | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.09 | 2.46 | 2.45 | 95.9485 | 0.1701 | 0.4072 | 1.0129 | 0.3463 |
| Uniform | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 112.8691 | 3.1032 | 1.3236 | 1.1302 | 12.0694 |
| Uniform | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.96 | 1.36 | 1.68 | 98.5448 | 0.0 | 0.0 | 1.0037 | 2.0382 |
| Uniform | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.94 | 2.47 | 2.59 | 99.2949 | 0.08 | 0.7168 | 1.0482 | 0.3603 |
| Uniform | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 127.8786 | 4.9093 | 2.1523 | 1.2804 | 10.8041 |
| Uniform | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.73 | 1.57 | 1.7 | 102.8711 | 0.0 | 0.0 | 1.0478 | 2.0771 |
| Uniform | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.75 | 2.64 | 2.61 | 103.5946 | 0.2825 | 0.8168 | 1.0936 | 0.4195 |
| Uniform | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 144.7124 | 5.9626 | 3.1894 | 1.449 | 33.8514 |
| Uniform | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.01 | 0.91 | 2.08 | 110.8621 | 0.0 | 0.0 | 1.1292 | 2.3528 |
| Uniform | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.12 | 2.61 | 2.27 | 95.7346 | 0.205 | 0.2682 | 1.0106 | 0.4916 |
| Uniform | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 334.1701 | 17.6372 | 4.7282 | 3.346 | 19.5448 |
| Uniform | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.55 | 0.9 | 1.55 | 98.223 | 0.0 | 0.0 | 1.0005 | 3.5595 |
| Uniform | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.18 | 2.42 | 2.4 | 179.3499 | 22.3945 | 30.8968 | 1.8933 | 0.3683 |
| Uniform | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 343.9149 | 0.9113 | 0.268 | 3.4436 | 6.2687 |
| Uniform | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.11 | 0.19 | 0.7 | 325.383 | 0.0 | 0.0 | 3.3142 | 2.0166 |
| Uniform | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.08 | 2.34 | 2.58 | 288.5184 | 2.8472 | 3.7301 | 3.0457 | 0.3648 |
| Uniform | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 344.3801 | 1.3396 | 0.2901 | 3.4483 | 6.3801 |
| Uniform | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.58 | 0.36 | 0.06 | 340.1256 | 0.0 | 0.0 | 3.4644 | 2.2669 |
| Wald | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.88 | 2.57 | 2.55 | 2.6961 | 0.0045 | 0.1292 | 1.0 | 0.679 |
| Wald | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 6.3698 | 0.0127 | 0.8437 | 0.9999 | 16.4458 |
| Wald | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 91.31 | 1.05 | 7.64 | 1.8749 | 0.0 | 0.0 | 1.0016 | 2.6965 |
| Wald | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.89 | 2.62 | 2.49 | 2.7382 | 0.01 | 0.137 | 1.0156 | 0.4098 |
| Wald | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 6.5047 | 0.0488 | 0.7909 | 1.021 | 16.3908 |
| Wald | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 91.48 | 0.97 | 7.55 | 1.9456 | 0.0 | 0.0 | 1.0394 | 2.4396 |
| Wald | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.89 | 2.65 | 2.46 | 2.8129 | 0.0159 | 0.1329 | 1.0433 | 0.3549 |
| Wald | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 6.741 | 0.0574 | 0.7791 | 1.0581 | 15.3748 |
| Wald | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 92.44 | 0.39 | 7.17 | 2.1584 | 0.0 | 0.0 | 1.153 | 2.4494 |
| Wald | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.78 | 2.69 | 2.53 | 2.924 | 0.0219 | 0.1246 | 1.0845 | 0.3555 |
| Wald | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 6.9779 | 0.0713 | 0.7708 | 1.0953 | 15.4322 |
| Wald | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 95.07 | 0.29 | 4.64 | 2.6841 | 0.0 | 0.0 | 1.4338 | 2.3901 |
| Wald | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.87 | 2.65 | 2.48 | 2.8977 | 0.0061 | 0.1232 | 1.0748 | 0.3561 |
| Wald | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 9.6634 | 0.4961 | 0.6997 | 1.5169 | 17.3799 |
| Wald | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 93.2 | 1.55 | 5.25 | 2.2714 | 0.0 | 0.0 | 1.2134 | 2.2423 |
| Wald | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.14 | 2.4 | 2.46 | 6.0043 | 0.6735 | 0.0966 | 2.227 | 0.3683 |
| Wald | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 9.8714 | 0.0212 | 0.6631 | 1.5495 | 9.1615 |
| Wald | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.34 | 1.68 | 0.98 | 7.3192 | 0.0 | 0.0 | 3.9099 | 2.1421 |
| Wald | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.03 | 2.34 | 2.63 | 7.3799 | 0.072 | 0.0784 | 2.7373 | 0.443 |
| Wald | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 9.8815 | 0.0322 | 0.6689 | 1.5511 | 11.7286 |
| Wald | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.47 | 0.34 | 0.19 | 8.6015 | 0.0 | 0.0 | 4.5949 | 8.2658 |
| Gamma | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.87 | 2.52 | 2.61 | 10.5032 | 0.0498 | 0.2439 | 1.0 | 0.3657 |
| Gamma | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 25.3324 | 0.0657 | 4.206 | 1.0 | 17.8982 |
| Gamma | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 95.24 | 1.96 | 2.8 | 10.4194 | 0.0 | 0.0 | 0.9996 | 2.0534 |
| Gamma | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.82 | 2.6 | 2.58 | 10.5663 | 0.0482 | 0.2471 | 1.006 | 0.3621 |
| Gamma | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 25.7652 | 0.2457 | 4.1012 | 1.0171 | 19.0276 |
| Gamma | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 95.15 | 2.11 | 2.74 | 10.5242 | 0.0 | 0.0 | 1.0097 | 2.1209 |
| Gamma | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.68 | 2.77 | 2.55 | 10.7713 | 0.0581 | 0.2326 | 1.0255 | 0.3711 |
| Gamma | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 26.3251 | 0.5072 | 3.9984 | 1.0392 | 17.0086 |
| Gamma | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 93.9 | 3.39 | 2.71 | 10.6233 | 0.0 | 0.0 | 1.0192 | 1.9952 |
| Gamma | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.7 | 2.67 | 2.63 | 11.2162 | 0.1038 | 0.2411 | 1.0679 | 0.3671 |
| Gamma | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 26.9509 | 0.7216 | 3.8789 | 1.0639 | 18.2921 |
| Gamma | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 94.26 | 3.02 | 2.72 | 11.0076 | 0.0 | 0.0 | 1.056 | 2.0067 |
| Gamma | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.93 | 2.58 | 2.49 | 11.1327 | 0.0502 | 0.2337 | 1.0599 | 0.4011 |
| Gamma | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 38.126 | 1.8785 | 2.7103 | 1.505 | 13.7092 |
| Gamma | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 94.31 | 2.39 | 3.3 | 10.425 | 0.0 | 0.0 | 1.0001 | 2.0172 |
| Gamma | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.19 | 2.4 | 2.41 | 22.6809 | 2.4513 | 0.4247 | 2.1594 | 0.3643 |
| Gamma | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 38.9891 | 0.1273 | 2.3446 | 1.5391 | 6.0474 |
| Gamma | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.57 | 1.68 | 0.75 | 28.3603 | 0.0 | 0.0 | 2.7208 | 2.0275 |
| Gamma | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.1 | 2.34 | 2.56 | 28.3503 | 0.2788 | 0.3405 | 2.6992 | 0.3959 |
| Gamma | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 39.0203 | 0.126 | 2.3347 | 1.5403 | 4.9825 |
| Gamma | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.58 | 0.34 | 0.08 | 33.3214 | 0.0 | 0.0 | 3.1967 | 2.0378 |
| Normal | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.88 | 2.67 | 2.45 | 3.9051 | 0.058 | 0.0558 | 1.0 | 0.3599 |
| Normal | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 7.2398 | 0.2672 | 0.2367 | 1.0 | 7.7327 |
| Normal | Scenario A (Clean) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.3 | 0.49 | 3.21 | 4.472 | 0.0 | 0.0 | 1.0 | 0.9746 |
| Normal | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.88 | 2.67 | 2.45 | 4.2969 | 0.0638 | 0.0614 | 1.1003 | 0.3885 |
| Normal | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 7.9664 | 0.294 | 0.2597 | 1.1004 | 9.6636 |
| Normal | Scenario B1 (GAWN Noise 0.1sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.32 | 0.49 | 3.19 | 4.925 | 0.0 | 0.0 | 1.1013 | 1.3446 |
| Normal | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.88 | 2.67 | 2.45 | 4.6888 | 0.0696 | 0.0671 | 1.2007 | 0.3959 |
| Normal | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 8.6928 | 0.3208 | 0.2834 | 1.2007 | 8.8713 |
| Normal | Scenario B2 (GAWN Noise 0.2sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.32 | 0.49 | 3.19 | 5.3741 | 0.0 | 0.0 | 1.2017 | 1.2353 |
| Normal | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.88 | 2.67 | 2.45 | 5.0806 | 0.0754 | 0.0727 | 1.301 | 0.3993 |
| Normal | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 9.4193 | 0.3476 | 0.3071 | 1.301 | 9.1928 |
| Normal | Scenario B3 (GAWN Noise 0.3sigma) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.32 | 0.49 | 3.19 | 5.8233 | 0.0 | 0.0 | 1.3022 | 1.2157 |
| Normal | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.02 | 2.65 | 2.33 | 4.0934 | 0.0727 | 0.0477 | 1.0482 | 0.4161 |
| Normal | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 11.6765 | 0.4972 | 0.165 | 1.6128 | 15.3847 |
| Normal | Scenario C1 (Impulse Spikes 1%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 95.32 | 0.98 | 3.7 | 4.472 | 0.0 | 0.0 | 1.0 | 1.1985 |
| Normal | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.23 | 2.4 | 2.37 | 7.3121 | 0.3712 | 0.6284 | 1.8724 | 0.3886 |
| Normal | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 12.0007 | 0.0291 | 0.0078 | 1.6576 | 5.4993 |
| Normal | Scenario C2 (Impulse Spikes 5%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.61 | 1.68 | 0.71 | 10.1046 | 0.0 | 0.0 | 2.2595 | 2.222 |
| Normal | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.08 | 2.34 | 2.58 | 10.0656 | 0.0993 | 0.1301 | 2.5776 | 0.3875 |
| Normal | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 100.0 | 0.0 | 0.0 | 12.0088 | 0.0435 | 0.0106 | 1.6587 | 7.619 |
| Normal | Scenario C3 (Impulse Spikes 10%) | In-Sample Adaptation | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.6 | 0.34 | 0.06 | 11.87 | 0.0 | 0.0 | 2.6543 | 2.259 |

---

## Protocol: One-Step-Ahead Pre-Sequential

| Distribution | Noise_Scenario | Eval_Mode | Chunk_Size | Target_Alpha | Target_Coverage_Pct | Target_Tail_FAR_Pct | Method | Empirical_Coverage_Pct | Left_FAR_Pct | Right_FAR_Pct | Mean_Interval_Width | Sigma_L_Stability | Sigma_R_Stability | Noise_Sensitivity_Ratio_NSR | Latency_per_Chunk_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-Distribution | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.88 | 2.62 | 2.5 | 4.046 | 0.0067 | 0.1187 | 1.0 | 0.3664 |
| F-Distribution | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.88 | 0.05 | 0.07 | 11.0432 | 0.0185 | 1.6472 | 1.0 | 8.035 |
| F-Distribution | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 98.08 | 0.07 | 1.85 | 4.5662 | 0.0163 | 0.0088 | 1.0002 | 2.0551 |
| F-Distribution | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.78 | 2.68 | 2.54 | 4.1039 | 0.0069 | 0.1107 | 1.0143 | 0.358 |
| F-Distribution | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.89 | 0.04 | 0.07 | 11.2917 | 0.0287 | 1.606 | 1.0225 | 6.8802 |
| F-Distribution | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 98.12 | 0.04 | 1.84 | 4.8257 | 0.0278 | 0.0155 | 1.0571 | 2.0332 |
| F-Distribution | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.76 | 2.68 | 2.56 | 4.2139 | 0.0165 | 0.116 | 1.0415 | 0.3717 |
| F-Distribution | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.89 | 0.05 | 0.06 | 11.6143 | 0.0728 | 1.5616 | 1.0517 | 7.1815 |
| F-Distribution | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 98.49 | 0.05 | 1.46 | 5.3888 | 0.0335 | 0.053 | 1.1804 | 1.9795 |
| F-Distribution | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.77 | 2.66 | 2.57 | 4.3282 | 0.0221 | 0.1302 | 1.0698 | 0.3733 |
| F-Distribution | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.87 | 0.07 | 0.06 | 11.9509 | 0.1178 | 1.514 | 1.0822 | 9.0925 |
| F-Distribution | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 98.35 | 0.11 | 1.54 | 5.5811 | 0.0398 | 0.0527 | 1.2226 | 2.0504 |
| F-Distribution | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.85 | 2.67 | 2.48 | 4.3566 | 0.0101 | 0.2112 | 1.0768 | 0.3608 |
| F-Distribution | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.84 | 0.09 | 0.07 | 16.0231 | 0.9147 | 1.6014 | 1.4509 | 11.6598 |
| F-Distribution | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.58 | 0.56 | 1.86 | 5.0184 | 0.0163 | 0.0718 | 1.0993 | 2.0147 |
| F-Distribution | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.18 | 2.39 | 2.43 | 9.1191 | 1.0545 | 0.1555 | 2.2538 | 0.3895 |
| F-Distribution | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.88 | 0.05 | 0.07 | 16.353 | 0.1548 | 1.569 | 1.4808 | 11.1428 |
| F-Distribution | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.51 | 0.22 | 0.27 | 12.9202 | 0.1404 | 0.0916 | 2.8303 | 2.0321 |
| F-Distribution | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.02 | 2.38 | 2.6 | 11.1079 | 0.1078 | 0.1776 | 2.7454 | 0.3662 |
| F-Distribution | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.86 | 0.07 | 0.07 | 16.3869 | 0.1087 | 1.5286 | 1.4839 | 11.1908 |
| F-Distribution | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.5 | 0.37 | 0.13 | 14.2752 | 0.0844 | 0.1472 | 3.1271 | 2.1916 |
| Uniform | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.93 | 2.58 | 2.49 | 94.7214 | 0.0968 | 0.1889 | 1.0 | 0.3476 |
| Uniform | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.83 | 0.09 | 0.08 | 99.7811 | 0.3411 | 0.3889 | 0.9997 | 14.9968 |
| Uniform | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 98.41 | 0.47 | 1.12 | 98.1123 | 0.2406 | 0.1926 | 0.9998 | 2.0486 |
| Uniform | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.02 | 2.5 | 2.48 | 95.9242 | 0.1893 | 0.4394 | 1.0127 | 0.3463 |
| Uniform | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.81 | 0.12 | 0.07 | 112.6433 | 3.2174 | 1.6776 | 1.1286 | 12.0694 |
| Uniform | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.9 | 1.39 | 1.71 | 98.4939 | 0.2396 | 0.2669 | 1.0037 | 2.0382 |
| Uniform | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.84 | 2.5 | 2.66 | 99.2614 | 0.0806 | 0.7891 | 1.0479 | 0.3603 |
| Uniform | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.85 | 0.1 | 0.05 | 127.5108 | 5.0789 | 2.7924 | 1.2776 | 10.8041 |
| Uniform | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.67 | 1.6 | 1.73 | 102.8031 | 0.1758 | 0.4999 | 1.0476 | 2.0771 |
| Uniform | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.72 | 2.64 | 2.64 | 103.5379 | 0.2888 | 0.9318 | 1.0931 | 0.4195 |
| Uniform | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.85 | 0.09 | 0.06 | 144.1883 | 6.2902 | 4.0652 | 1.4447 | 33.8514 |
| Uniform | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.95 | 0.94 | 2.11 | 110.7377 | 0.6235 | 0.6144 | 1.1284 | 2.3528 |
| Uniform | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.06 | 2.65 | 2.29 | 95.7271 | 0.2194 | 0.2674 | 1.0106 | 0.4916 |
| Uniform | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.82 | 0.1 | 0.08 | 331.6796 | 21.1901 | 12.9211 | 3.3232 | 19.5448 |
| Uniform | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.51 | 0.93 | 1.56 | 98.19 | 0.2433 | 0.0859 | 1.0006 | 3.5595 |
| Uniform | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.13 | 2.46 | 2.41 | 180.3659 | 22.4699 | 31.6649 | 1.9042 | 0.3683 |
| Uniform | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.9 | 0.05 | 0.05 | 343.1715 | 4.0564 | 3.3634 | 3.4383 | 6.2687 |
| Uniform | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.06 | 0.22 | 0.72 | 324.8339 | 3.6909 | 1.7722 | 3.3101 | 2.0166 |
| Uniform | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.05 | 2.38 | 2.57 | 288.7222 | 2.8405 | 4.2163 | 3.0481 | 0.3648 |
| Uniform | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.9 | 0.05 | 0.05 | 344.0051 | 2.9202 | 1.0856 | 3.4467 | 6.3801 |
| Uniform | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.52 | 0.39 | 0.09 | 339.802 | 2.2066 | 1.0131 | 3.4626 | 2.2669 |
| Wald | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.75 | 2.58 | 2.67 | 2.6863 | 0.0057 | 0.1597 | 1.0 | 0.679 |
| Wald | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.83 | 0.07 | 0.1 | 6.314 | 0.0221 | 0.949 | 0.9999 | 16.4458 |
| Wald | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 91.26 | 1.08 | 7.66 | 1.8733 | 0.008 | 0.0078 | 1.0016 | 2.6965 |
| Wald | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.69 | 2.67 | 2.64 | 2.7279 | 0.0118 | 0.1669 | 1.0155 | 0.4098 |
| Wald | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.79 | 0.1 | 0.11 | 6.4482 | 0.0573 | 0.9038 | 1.0211 | 16.3908 |
| Wald | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 91.43 | 1.0 | 7.57 | 1.9433 | 0.0124 | 0.0103 | 1.039 | 2.4396 |
| Wald | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.7 | 2.7 | 2.6 | 2.8024 | 0.0184 | 0.1632 | 1.0432 | 0.3549 |
| Wald | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.82 | 0.07 | 0.11 | 6.6828 | 0.0736 | 0.895 | 1.0583 | 15.3748 |
| Wald | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 92.39 | 0.42 | 7.19 | 2.1545 | 0.0251 | 0.0139 | 1.1519 | 2.4494 |
| Wald | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.6 | 2.76 | 2.64 | 2.9134 | 0.0267 | 0.1524 | 1.0846 | 0.3555 |
| Wald | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.84 | 0.05 | 0.11 | 6.9186 | 0.0942 | 0.8859 | 1.0956 | 15.4322 |
| Wald | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 95.01 | 0.32 | 4.67 | 2.6761 | 0.0407 | 0.0387 | 1.4308 | 2.3901 |
| Wald | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.72 | 2.7 | 2.58 | 2.8888 | 0.0076 | 0.1495 | 1.0754 | 0.3561 |
| Wald | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.81 | 0.11 | 0.08 | 9.5763 | 0.6019 | 0.8114 | 1.5165 | 17.3799 |
| Wald | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 93.16 | 1.58 | 5.26 | 2.2687 | 0.0078 | 0.0189 | 1.213 | 2.2423 |
| Wald | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.17 | 2.39 | 2.44 | 6.0107 | 0.6753 | 0.1012 | 2.2376 | 0.3683 |
| Wald | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.88 | 0.05 | 0.07 | 9.8325 | 0.1022 | 0.6878 | 1.557 | 9.1615 |
| Wald | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.3 | 1.7 | 1.0 | 7.3143 | 0.0038 | 0.0443 | 3.9106 | 2.1421 |
| Wald | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.0 | 2.38 | 2.62 | 7.3838 | 0.0718 | 0.0882 | 2.7487 | 0.443 |
| Wald | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.89 | 0.05 | 0.06 | 9.852 | 0.0727 | 0.6789 | 1.5601 | 11.7286 |
| Wald | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.41 | 0.37 | 0.22 | 8.5933 | 0.0563 | 0.0257 | 4.5944 | 8.2658 |
| Gamma | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.77 | 2.55 | 2.68 | 10.4827 | 0.0536 | 0.2983 | 1.0 | 0.3657 |
| Gamma | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.79 | 0.14 | 0.07 | 25.1461 | 0.0905 | 4.4975 | 1.0 | 17.8982 |
| Gamma | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 95.19 | 1.99 | 2.82 | 10.4009 | 0.0266 | 0.1572 | 0.9996 | 2.0534 |
| Gamma | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.65 | 2.69 | 2.66 | 10.5447 | 0.0548 | 0.3044 | 1.0059 | 0.3621 |
| Gamma | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.81 | 0.12 | 0.07 | 25.573 | 0.2719 | 4.3973 | 1.017 | 19.0276 |
| Gamma | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 95.1 | 2.14 | 2.76 | 10.5041 | 0.0325 | 0.1676 | 1.0095 | 2.1209 |
| Gamma | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.54 | 2.79 | 2.67 | 10.7486 | 0.0677 | 0.2941 | 1.0254 | 0.3711 |
| Gamma | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.79 | 0.14 | 0.07 | 26.1262 | 0.5349 | 4.2977 | 1.039 | 17.0086 |
| Gamma | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 93.85 | 3.42 | 2.73 | 10.6031 | 0.0267 | 0.1738 | 1.019 | 1.9952 |
| Gamma | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.6 | 2.72 | 2.68 | 11.1904 | 0.115 | 0.3082 | 1.0675 | 0.3671 |
| Gamma | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.76 | 0.16 | 0.08 | 26.7461 | 0.755 | 4.1831 | 1.0636 | 18.2921 |
| Gamma | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 94.21 | 3.05 | 2.74 | 10.9853 | 0.0417 | 0.18 | 1.0558 | 2.0067 |
| Gamma | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.86 | 2.64 | 2.5 | 11.1425 | 0.056 | 0.2641 | 1.0629 | 0.4011 |
| Gamma | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.81 | 0.12 | 0.07 | 37.8468 | 2.2665 | 3.0101 | 1.5051 | 13.7092 |
| Gamma | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 94.28 | 2.42 | 3.3 | 10.4419 | 0.0271 | 0.196 | 1.0035 | 2.0172 |
| Gamma | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.18 | 2.4 | 2.42 | 22.7138 | 2.4584 | 0.4612 | 2.1668 | 0.3643 |
| Gamma | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.91 | 0.05 | 0.04 | 38.8524 | 0.4075 | 2.4938 | 1.5451 | 6.0474 |
| Gamma | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.53 | 1.7 | 0.77 | 28.3416 | 0.0151 | 0.1716 | 2.7238 | 2.0275 |
| Gamma | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.06 | 2.38 | 2.56 | 28.3695 | 0.2781 | 0.3863 | 2.7063 | 0.3959 |
| Gamma | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.91 | 0.05 | 0.04 | 38.9208 | 0.2821 | 2.4161 | 1.5478 | 4.9825 |
| Gamma | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.52 | 0.37 | 0.11 | 33.2895 | 0.2179 | 0.0992 | 3.1994 | 2.0378 |
| Normal | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.82 | 2.7 | 2.48 | 3.9006 | 0.0579 | 0.07 | 1.0 | 0.3599 |
| Normal | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.81 | 0.08 | 0.11 | 7.1962 | 0.2971 | 0.3307 | 1.0 | 7.7327 |
| Normal | Scenario A (Clean) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.24 | 0.52 | 3.24 | 4.4622 | 0.0679 | 0.0299 | 1.0 | 0.9746 |
| Normal | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.82 | 2.7 | 2.48 | 4.2919 | 0.0637 | 0.077 | 1.1003 | 0.3885 |
| Normal | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.81 | 0.08 | 0.11 | 7.9184 | 0.327 | 0.3634 | 1.1004 | 9.6636 |
| Normal | Scenario B1 (GAWN Noise 0.1sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.26 | 0.52 | 3.22 | 4.9142 | 0.0747 | 0.0333 | 1.1013 | 1.3446 |
| Normal | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.82 | 2.7 | 2.48 | 4.6833 | 0.0695 | 0.0841 | 1.2007 | 0.3959 |
| Normal | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.81 | 0.08 | 0.11 | 8.6405 | 0.3568 | 0.3965 | 1.2007 | 8.8713 |
| Normal | Scenario B2 (GAWN Noise 0.2sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.26 | 0.52 | 3.22 | 5.3623 | 0.0815 | 0.0363 | 1.2017 | 1.2353 |
| Normal | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.82 | 2.7 | 2.48 | 5.0747 | 0.0753 | 0.0911 | 1.301 | 0.3993 |
| Normal | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.81 | 0.08 | 0.11 | 9.3626 | 0.3866 | 0.4297 | 1.301 | 9.1928 |
| Normal | Scenario B3 (GAWN Noise 0.3sigma) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 96.26 | 0.52 | 3.22 | 5.8104 | 0.0883 | 0.0394 | 1.3021 | 1.2157 |
| Normal | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 94.89 | 2.71 | 2.4 | 4.0888 | 0.0729 | 0.0593 | 1.0483 | 0.4161 |
| Normal | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.82 | 0.1 | 0.08 | 11.5931 | 0.626 | 0.4476 | 1.611 | 15.3847 |
| Normal | Scenario C1 (Impulse Spikes 1%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 95.28 | 1.01 | 3.71 | 4.4639 | 0.0679 | 0.0129 | 1.0004 | 1.1985 |
| Normal | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.2 | 2.4 | 2.4 | 7.3357 | 0.3749 | 0.6474 | 1.8807 | 0.3886 |
| Normal | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.91 | 0.05 | 0.04 | 11.9748 | 0.141 | 0.1174 | 1.664 | 5.4993 |
| Normal | Scenario C2 (Impulse Spikes 5%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 97.57 | 1.7 | 0.73 | 10.0979 | 0.0054 | 0.0611 | 2.263 | 2.222 |
| Normal | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Traditional Offline | 95.05 | 2.38 | 2.57 | 10.0727 | 0.0991 | 0.1471 | 2.5824 | 0.3875 |
| Normal | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Cumulative Online | 99.9 | 0.04 | 0.06 | 11.9958 | 0.1 | 0.038 | 1.667 | 7.619 |
| Normal | Scenario C3 (Impulse Spikes 10%) | One-Step-Ahead Pre-Sequential | 100 | 0.05 | 95.0 | 2.5 | Proposed RBULT | 99.54 | 0.37 | 0.09 | 11.8587 | 0.0775 | 0.0352 | 2.6576 | 2.259 |

---

## Overall Aggregated Comparative Summary Across All Distributions & Protocols

| Eval_Mode | Noise_Scenario | Method | Empirical_Coverage_Pct | Left_FAR_Pct | Right_FAR_Pct | Mean_Interval_Width | Sigma_L_Stability | Noise_Sensitivity_Ratio_NSR | Latency_per_Chunk_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| In-Sample Adaptation | Scenario A (Clean) | Cumulative Online | 100.0 | 0.0 | 0.0 | 30.01 | 0.11 | 1.0 | 19.49 |
| In-Sample Adaptation | Scenario A (Clean) | Proposed RBULT | 95.46 | 1.29 | 3.25 | 23.66 | 0.0 | 1.0 | 4.82 |
| In-Sample Adaptation | Scenario A (Clean) | Traditional Offline | 96.92 | 1.55 | 1.53 | 24.42 | 0.05 | 1.0 | 0.4 |
| In-Sample Adaptation | Scenario B1 (GAWN Noise 0.1sigma) | Cumulative Online | 100.0 | 0.0 | 0.0 | 32.95 | 0.74 | 1.06 | 20.79 |
| In-Sample Adaptation | Scenario B1 (GAWN Noise 0.1sigma) | Proposed RBULT | 95.39 | 1.43 | 3.18 | 24.66 | 0.0 | 1.05 | 4.78 |
| In-Sample Adaptation | Scenario B1 (GAWN Noise 0.1sigma) | Traditional Offline | 96.89 | 1.59 | 1.52 | 25.13 | 0.08 | 1.03 | 0.4 |
| In-Sample Adaptation | Scenario B2 (GAWN Noise 0.2sigma) | Cumulative Online | 100.0 | 0.0 | 0.0 | 36.36 | 1.16 | 1.13 | 20.85 |
| In-Sample Adaptation | Scenario B2 (GAWN Noise 0.2sigma) | Proposed RBULT | 95.38 | 1.48 | 3.14 | 26.45 | 0.0 | 1.11 | 5.17 |
| In-Sample Adaptation | Scenario B2 (GAWN Noise 0.2sigma) | Traditional Offline | 96.85 | 1.61 | 1.54 | 26.43 | 0.09 | 1.08 | 0.39 |
| In-Sample Adaptation | Scenario B3 (GAWN Noise 0.3sigma) | Cumulative Online | 100.0 | 0.0 | 0.0 | 40.11 | 1.42 | 1.2 | 24.24 |
| In-Sample Adaptation | Scenario B3 (GAWN Noise 0.3sigma) | Proposed RBULT | 95.8 | 1.3 | 2.9 | 28.81 | 0.0 | 1.21 | 5.66 |
| In-Sample Adaptation | Scenario B3 (GAWN Noise 0.3sigma) | Traditional Offline | 96.82 | 1.63 | 1.55 | 28.05 | 0.15 | 1.14 | 0.45 |
| In-Sample Adaptation | Scenario C1 (Impulse Spikes 1%) | Cumulative Online | 100.0 | 0.0 | 0.0 | 82.16 | 3.45 | 1.89 | 26.45 |
| In-Sample Adaptation | Scenario C1 (Impulse Spikes 1%) | Proposed RBULT | 95.32 | 1.74 | 2.94 | 24.01 | 0.0 | 1.1 | 5.23 |
| In-Sample Adaptation | Scenario C1 (Impulse Spikes 1%) | Traditional Offline | 96.95 | 1.59 | 1.46 | 32.21 | 3.72 | 1.24 | 0.4 |
| In-Sample Adaptation | Scenario C2 (Impulse Spikes 5%) | Cumulative Online | 100.0 | 0.0 | 0.0 | 84.25 | 0.41 | 1.93 | 15.83 |
| In-Sample Adaptation | Scenario C2 (Impulse Spikes 5%) | Proposed RBULT | 97.7 | 1.47 | 0.82 | 71.97 | 0.0 | 2.96 | 4.44 |
| In-Sample Adaptation | Scenario C2 (Impulse Spikes 5%) | Traditional Offline | 97.08 | 1.43 | 1.49 | 61.05 | 3.06 | 2.21 | 0.39 |
| In-Sample Adaptation | Scenario C3 (Impulse Spikes 10%) | Cumulative Online | 100.0 | 0.0 | 0.0 | 84.35 | 0.3 | 1.93 | 18.11 |
| In-Sample Adaptation | Scenario C3 (Impulse Spikes 10%) | Proposed RBULT | 99.52 | 0.36 | 0.12 | 81.44 | 0.0 | 3.41 | 5.05 |
| In-Sample Adaptation | Scenario C3 (Impulse Spikes 10%) | Traditional Offline | 97.02 | 1.4 | 1.58 | 74.45 | 0.41 | 2.59 | 0.4 |
| One-Step-Ahead Pre-Sequential | Scenario A (Clean) | Cumulative Online | 99.82 | 0.09 | 0.09 | 29.85 | 0.15 | 1.0 | 19.49 |
| One-Step-Ahead Pre-Sequential | Scenario A (Clean) | Proposed RBULT | 95.41 | 1.33 | 3.26 | 23.65 | 0.05 | 1.0 | 4.82 |
| One-Step-Ahead Pre-Sequential | Scenario A (Clean) | Traditional Offline | 96.84 | 1.59 | 1.58 | 24.4 | 0.06 | 1.0 | 0.4 |
| One-Step-Ahead Pre-Sequential | Scenario B1 (GAWN Noise 0.1sigma) | Cumulative Online | 99.82 | 0.09 | 0.09 | 32.73 | 0.79 | 1.06 | 20.79 |
| One-Step-Ahead Pre-Sequential | Scenario B1 (GAWN Noise 0.1sigma) | Proposed RBULT | 95.34 | 1.47 | 3.19 | 24.62 | 0.07 | 1.05 | 4.78 |
| One-Step-Ahead Pre-Sequential | Scenario B1 (GAWN Noise 0.1sigma) | Traditional Offline | 96.8 | 1.63 | 1.57 | 25.11 | 0.09 | 1.03 | 0.4 |
| One-Step-Ahead Pre-Sequential | Scenario B2 (GAWN Noise 0.2sigma) | Cumulative Online | 99.83 | 0.09 | 0.08 | 36.08 | 1.24 | 1.13 | 20.85 |
| One-Step-Ahead Pre-Sequential | Scenario B2 (GAWN Noise 0.2sigma) | Proposed RBULT | 95.33 | 1.52 | 3.15 | 26.37 | 0.12 | 1.11 | 5.17 |
| One-Step-Ahead Pre-Sequential | Scenario B2 (GAWN Noise 0.2sigma) | Traditional Offline | 96.76 | 1.65 | 1.59 | 26.4 | 0.1 | 1.08 | 0.39 |
| One-Step-Ahead Pre-Sequential | Scenario B3 (GAWN Noise 0.3sigma) | Cumulative Online | 99.82 | 0.1 | 0.08 | 39.77 | 1.55 | 1.2 | 24.24 |
| One-Step-Ahead Pre-Sequential | Scenario B3 (GAWN Noise 0.3sigma) | Proposed RBULT | 95.74 | 1.34 | 2.92 | 28.7 | 0.21 | 1.21 | 5.66 |
| One-Step-Ahead Pre-Sequential | Scenario B3 (GAWN Noise 0.3sigma) | Traditional Offline | 96.74 | 1.67 | 1.59 | 28.01 | 0.15 | 1.14 | 0.45 |
| One-Step-Ahead Pre-Sequential | Scenario C1 (Impulse Spikes 1%) | Cumulative Online | 99.81 | 0.11 | 0.08 | 81.07 | 5.55 | 1.88 | 26.45 |
| One-Step-Ahead Pre-Sequential | Scenario C1 (Impulse Spikes 1%) | Proposed RBULT | 95.29 | 1.77 | 2.94 | 24.22 | 0.05 | 1.1 | 5.23 |
| One-Step-Ahead Pre-Sequential | Scenario C1 (Impulse Spikes 1%) | Traditional Offline | 96.89 | 1.63 | 1.48 | 32.26 | 3.73 | 1.24 | 0.4 |
| One-Step-Ahead Pre-Sequential | Scenario C2 (Impulse Spikes 5%) | Cumulative Online | 99.86 | 0.07 | 0.07 | 83.91 | 1.25 | 1.94 | 15.83 |
| One-Step-Ahead Pre-Sequential | Scenario C2 (Impulse Spikes 5%) | Proposed RBULT | 97.7 | 1.47 | 0.83 | 71.98 | 0.8 | 2.96 | 4.44 |
| One-Step-Ahead Pre-Sequential | Scenario C2 (Impulse Spikes 5%) | Traditional Offline | 97.06 | 1.45 | 1.49 | 61.24 | 3.18 | 2.22 | 0.39 |
| One-Step-Ahead Pre-Sequential | Scenario C3 (Impulse Spikes 10%) | Cumulative Online | 99.87 | 0.07 | 0.07 | 84.17 | 0.57 | 1.94 | 18.11 |
| One-Step-Ahead Pre-Sequential | Scenario C3 (Impulse Spikes 10%) | Proposed RBULT | 99.44 | 0.4 | 0.16 | 81.34 | 0.34 | 3.41 | 5.05 |
| One-Step-Ahead Pre-Sequential | Scenario C3 (Impulse Spikes 10%) | Traditional Offline | 96.99 | 1.42 | 1.59 | 74.5 | 0.45 | 2.6 | 0.4 |