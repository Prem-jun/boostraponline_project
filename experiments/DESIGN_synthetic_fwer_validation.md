# Design: Synthetic Multivariate FWER Validation

**Status:** design, not yet implemented.
**Purpose:** test the paper's central theoretical claim — that Bonferroni correction
$\alpha_{\text{dim}} = \alpha_{\text{sys}}/D$ delivers joint false alarm rate
$\le \alpha_{\text{sys}}$ — under conditions where the ground truth is known exactly.

---

## 1. The gap this fills

The claim is currently tested **only on data where it fails**. Joint coverage falls below the
95% Bonferroni target on 6 of 8 Tier 2 datasets, reaching 25% on TEP Mode 3. On real data
three causes are confounded and cannot be separated:

| Candidate cause | Separable on real data? |
|---|---|
| The Bonferroni correction is itself insufficient | No |
| RBULT's estimator does not attain its nominal per-dimension level | No |
| Dependence between dimensions | No — the true covariance is unknown |

What is known is that per-dimension FAR on TEP runs **15–43× the nominal $\alpha_{\text{dim}}$**.
Whether that is intrinsic to the estimator or specific to that data is unanswerable without a
controlled experiment.

There is also a structural gap in the evaluation:

| Tier | Dimensionality | Ground truth | Exists |
|---|---|---|---|
| Tier 1 | 1D | Known distribution | Yes |
| **This study** | **2–34D** | **Known covariance** | **No** |
| Tier 2 | 5–34D | None | Yes |

## 2. Research questions

- **RQ1** Does Bonferroni deliver joint FAR $\le \alpha_{\text{sys}}$ when the per-dimension
  intervals attain their nominal level exactly?
- **RQ2** Does RBULT attain its nominal per-dimension level $1-\alpha_{\text{dim}}$, and how
  does that degrade with $D$?
- **RQ3** When joint FAR exceeds $\alpha_{\text{sys}}$, how much is attributable to the
  correction versus to the estimator?
- **RQ4** How does dependence $\rho$ change Bonferroni's conservativeness?
- **RQ5** Does the non-monotonic chunk-size dependence found on TEP reproduce under control?

## 3. Design

### 3.1 Three arms — the oracle is the linchpin

RQ3 is answerable only by comparing against an estimator that attains nominal **by
construction**. Without this arm the study cannot separate the correction from the estimator,
and reduces to a restatement of the Tier 2 finding.

| Arm | Limits per dimension $d$ | Isolates |
|---|---|---|
| **Oracle** | true quantiles $[F_d^{-1}(\alpha_{\text{dim}}/2),\; F_d^{-1}(1-\alpha_{\text{dim}}/2)]$ | Bonferroni alone (RQ1, RQ4) |
| **Empirical** | empirical quantiles at the same levels, from the training sample | the finite-sample penalty |
| **RBULT** | `RBULTControlChart` | the method under test |

**Oracle vs $\alpha_{\text{sys}}$** answers RQ1.
**RBULT vs Oracle** answers RQ2 and RQ3.
**Empirical vs Oracle** separates "finite sample" from "this particular estimator".

### 3.2 Factors

| Factor | Levels | Rationale |
|---|---|---|
| $D$ | 2, 5, 10, 34 | matches Tier 2 (Pump 5, MetroPT-3 7, Water Pump 10, TEP 34) |
| $\rho$ | 0, 0.3, 0.6, 0.9 | equicorrelation; Bonferroni's conservativeness depends on dependence, and TEP's 34 channels are strongly correlated |
| Marginal | Gaussian, Lognormal | Gaussian matches the theory exactly; lognormal is skewed and heavy-tailed, RBULT's actual target domain |
| $\alpha_{\text{sys}}$ | 0.05 | 0.01 as a secondary sweep on a reduced grid |

$4 \times 4 \times 2 = 32$ cells $\times$ 3 arms $\times$ 1,000 replications.

Data generation: draw $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \Sigma)$ with
$\Sigma_{ij} = \rho$ for $i \ne j$, $1$ on the diagonal (positive semi-definite for all
$\rho \ge 0$). For non-Gaussian marginals apply a Gaussian copula: transform
$u_d = \Phi(z_d)$ then $x_d = F_d^{-1}(u_d)$. **The dependence structure is preserved and the
marginal quantiles remain known in closed form**, which is what keeps the oracle exact.

### 3.3 Protocol — held-out evaluation

Each replication:

1. **Train.** Generate $N_{\text{train}}$ in-control samples, stream them to the chart in
   chunks of $k$, establishing $[L_d, R_d]$.
2. **Evaluate.** Generate $N_{\text{eval}}$ **fresh** in-control samples from the same
   distribution and score them against the frozen limits.

Evaluating on held-out data is deliberate. Tier 2's coverage is in-sample — final limits
applied retrospectively to the data that produced them — which we measured as optimistic by a
factor of 1.62 on MetroPT-3. Here the separation is structural, so the FAR estimate is
unbiased by construction and the study does not inherit the flaw it is meant to examine.

**No faults are generated.** The claim under test concerns false alarms under in-control
operation. Detection power is a separate question and deliberately out of scope.

Base configuration: $N_{\text{train}} = 10{,}000$, $N_{\text{eval}} = 10{,}000$, $k = 500$.

### 3.4 Chunk-size sub-study (RQ5)

On a reduced grid ($D \in \{5, 34\}$, $\rho \in \{0, 0.6\}$, Gaussian, 200 reps), sweep
$k \in \{100, 250, 500, 1000, 2000\}$ at fixed $N_{\text{train}}$. On TEP the final interval
width varied non-monotonically with $k$ (widths 62.98 / 31.22 / 34.32 / 36.10 / 45.37 at
$k = 100/150/300/600/1200$) with negligible RNG spread. This tests whether that is a property
of the estimator or of that dataset.

## 4. Metrics

Per replication, per arm:

| Metric | Definition | Target |
|---|---|---|
| `far_dim_d` | fraction of eval samples outside $[L_d, R_d]$, per $d$ | $\alpha_{\text{dim}}$ |
| `far_dim_mean` | mean over $d$ | $\alpha_{\text{dim}}$ |
| `far_joint` | fraction outside the hyper-rectangle $\prod_d [L_d, R_d]$ | $\le \alpha_{\text{sys}}$ |
| `width_d` | $R_d - L_d$ | — |
| `efficiency` | `width_d` / oracle `width_d`, averaged over $d$ | $\approx 1$ |

`efficiency` matters because a FAR below target is trivially achievable by widening. It is
the synthetic counterpart of `width_ratio_local` and must be reported alongside the FAR, for
the same reason.

## 5. Analysis plan

For each cell, across the 1,000 replications:

- mean `far_joint` with a Monte Carlo 95% CI
- one-sided test of $H_0:\ \mathbb{E}[\texttt{far\_joint}] \le \alpha_{\text{sys}}$
- the ratio $\mathbb{E}[\texttt{far\_dim\_mean}] / \alpha_{\text{dim}}$ — the direct synthetic
  analogue of the 15–43× overshoot measured on TEP
- `efficiency`, so a low FAR bought by a wide interval is visible

### Interpretation, fixed in advance

Committing to these readings now prevents rationalising whatever comes out.

| Oracle | RBULT | Reading |
|---|---|---|
| $\le \alpha_{\text{sys}}$ | $\le \alpha_{\text{sys}}$ | Both correction and estimator sound; the Tier 2 failures are data-driven (non-stationarity, dependence beyond this model) |
| $\le \alpha_{\text{sys}}$ | $> \alpha_{\text{sys}}$ | **Bonferroni is fine; the estimator does not attain nominal.** The Tier 2 joint-coverage collapse is a property of RBULT, and must be reported as a limitation |
| $> \alpha_{\text{sys}}$ | any | The correction itself is insufficient as applied — would require revisiting the FWER argument |
| either | $\le \alpha_{\text{sys}}$ with efficiency $\gg 1$ | FAR met only by inflating the interval — not a genuine pass |

Given that RBULT's per-dimension FAR runs 15–43× nominal on TEP, **row 2 is the outcome I
expect**. If it holds, the paper gains a controlled, quantified statement of a limitation it
currently states only vaguely — which is more defensible than leaving a reviewer to find it.

## 6. Cost

Dominated by RBULT at $D = 34$ (~10 ms per chunk measured). Base study: 32 cells × 1,000 reps
× 20 train chunks ≈ 40–60 min single-threaded, less with the 2-at-a-time parallelism used
elsewhere in this repo. The oracle and empirical arms are negligible by comparison.

A `--quick` mode with 100 reps gives ~3× wider CIs and is enough to see the direction; the
full 1,000 is for the reported numbers.

## 7. Threats to validity

- **Equicorrelation is a simplification.** Real covariance is not exchangeable. It isolates
  the effect of dependence strength cleanly, which is the point, but conclusions about
  *specific* datasets do not follow. A factor-model covariance would be the natural extension.
- **The Gaussian copula fixes the dependence family.** Tail dependence differs under other
  copulas, and tail behaviour is exactly what RBULT estimates. A t-copula arm would test this
  and is the first extension I would add.
- **Stationary and i.i.d. by construction.** Real streams are neither, and we have shown
  non-stationarity inflates RBULT's intervals substantially. This study therefore gives an
  *optimistic* bound on real-world behaviour — it cannot explain the whole Tier 2 gap, and
  should not be presented as if it does.
- **RNG seeding is required** for this study to be reproducible; Tier 2 currently has none.
  Seed per replication as `seed = base + rep_index` and record it.

## 8. Deliverables

- `experiments/exp_synthetic_fwer.py`
- `results/synthetic_fwer_results.csv` — one row per (cell, arm, replication)
- `results/synthetic_fwer_report.md` — cell summaries, CIs, the decomposition table
- Two figures: joint FAR vs $D$ by arm (one panel per $\rho$); efficiency vs $D$
