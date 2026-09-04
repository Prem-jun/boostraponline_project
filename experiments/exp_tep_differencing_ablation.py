"""
Does TEP need stationary differencing?

Differencing is applied WITHIN each simulation run only, never across run boundaries —
the mistake that made AI4I's 'Tool wear Rate' meaningless. With k = 600 = one run, that
means dropping each run's first sample, giving 599 samples per chunk.

Compares, per mode:
  RAW   : x[t]              (current pipeline)
  DIFF  : x[t] - x[t-1]     within-run first-order difference

on coverage, joint coverage, interval width relative to local variation, and — decisively
— AUC of the violation statistic against the true fault label.
"""
import os
import sys
import io
import pickle
import contextlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = '/Users/premjunsawang/Documents/GitHub/boostraponline_project'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from online_bootstrap.spc_rbult import RBULTControlChart

_buf = io.StringIO()


def auc(y, s):
    y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def stationarity_probe(sig, cols):
    """Within a run, does the mean drift? Compare first vs last third of each run."""
    a = sig[:, :200, :][:, :, cols].mean(axis=1)
    b = sig[:, 400:, :][:, :, cols].mean(axis=1)
    sd = sig[:, :, cols].reshape(-1, len(cols)).std(axis=0)
    shift = np.abs(b - a).mean(axis=0) / np.maximum(sd, 1e-12)
    return float(np.median(shift)), float(np.max(shift))


def run_mode(mode):
    with open(f'TEPDataset_M1_M5/TEPDataset_Mode{mode}.pickle', 'rb') as f:
        d = pickle.load(f)
    sig, lb = d['Signals'], np.asarray(d['Labels'])
    R, T, D = sig.shape
    cols = [i for i in range(D) if sig.reshape(-1, D)[:, i].std() > 1e-9]
    feats = [f'sensor_{i:02d}' for i in cols]
    labels = (lb > 0).astype(int)

    med, mx = stationarity_probe(sig, cols)
    print(f"\n{'='*92}")
    print(f"TEP Mode {mode}  ({R} runs x {T} steps, {len(cols)} non-constant channels)")
    print(f"  within-run mean shift (last third vs first third), in units of channel sd:"
          f"  median {med:.3f}   max {mx:.3f}")

    out = {}
    for label, arr in (('RAW ', sig[:, :, cols]),
                       ('DIFF', np.diff(sig[:, :, cols], axis=1))):
        k = arr.shape[1]
        flat = arr.reshape(-1, len(cols))
        df = pd.DataFrame(flat, columns=feats)
        with contextlib.redirect_stdout(_buf):
            chart = RBULTControlChart(features=feats)
            counts = []
            for i in range(R):
                s = chart.update_chunk(df.iloc[i * k:(i + 1) * k])
                counts.append([s['ooc_flags'].get(f, {}).get('ooc_count', 0) for f in feats])
            m = chart.compute_spc_metrics(true_labels=list(labels), sample_df=df)
        counts = np.array(counts)
        a_max = auc(labels, counts.max(axis=1))
        a_sum = auc(labels, counts.sum(axis=1))
        out[label] = dict(k=k, coverage=m['overall_coverage_pct'], joint=m['joint_coverage_pct'],
                          preq=m.get('prequential_coverage_pct', float('nan')),
                          ratio_local=m.get('width_ratio_local', float('nan')),
                          ratio_global=m.get('width_ratio_global', float('nan')),
                          chunk_far=m.get('false_alarm_rate', float('nan')) * 100,
                          auc_max=a_max, auc_sum=a_sum)

    print(f"  {'':5} {'k':>4} {'cov%':>7} {'preq%':>7} {'joint%':>7} {'r_local':>8} "
          f"{'r_global':>9} {'chunkFAR%':>10} {'AUC max':>8} {'AUC sum':>8}")
    for label, r in out.items():
        print(f"  {label:5} {r['k']:>4} {r['coverage']:>7.2f} {r['preq']:>7.2f} {r['joint']:>7.2f} "
              f"{r['ratio_local']:>8.2f} {r['ratio_global']:>9.2f} {r['chunk_far']:>10.2f} "
              f"{r['auc_max']:>8.3f} {r['auc_sum']:>8.3f}")
    return {'mode': mode, **{f'{k}_{kk}': vv for k, r in out.items() for kk, vv in r.items()}}


if __name__ == '__main__':
    rows = [run_mode(m) for m in ['1', '3', '4', '5']]
    pd.DataFrame(rows).to_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tep_diff.csv'), index=False)
    print('\nsaved tep_diff.csv')
