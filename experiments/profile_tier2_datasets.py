"""Characterise the Tier-2 datasets: are they genuinely multivariate time series?"""
import os
import sys
import io
import contextlib

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, shapiro

ROOT = '/Users/premjunsawang/Documents/GitHub/boostraponline_project'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
_buf = io.StringIO()


def profile(name, df, feats, label_col, note=''):
    X = df[feats].astype(float)
    n = len(df)
    # lag-1 autocorrelation per feature (time-series signature)
    ac = []
    for f in feats:
        v = X[f].values
        v = v[~np.isnan(v)]
        if len(v) > 2 and v.std() > 0:
            ac.append(np.corrcoef(v[:-1], v[1:])[0, 1])
    ac = np.array(ac)

    sk = np.array([abs(skew(X[f].dropna())) for f in feats])
    ku = np.array([kurtosis(X[f].dropna()) for f in feats])

    # normality: Shapiro on a 5000-sample subsample per feature
    rng = np.random.default_rng(0)
    n_norm = 0
    for f in feats:
        v = X[f].dropna().values
        if len(v) > 20 and v.std() > 0:
            s = rng.choice(v, size=min(5000, len(v)), replace=False)
            if shapiro(s).pvalue > 0.05:
                n_norm += 1

    lab = df[label_col].values
    # how many contiguous fault episodes
    d = np.diff(np.concatenate([[0], (lab > 0).astype(int), [0]]))
    n_episodes = int((d == 1).sum())

    print(f"\n{'='*88}")
    print(f"{name}   N={n:,}  D={len(feats)}")
    if note:
        print(f"  {note}")
    print(f"  lag-1 autocorrelation : median {np.median(ac):.3f}   min {ac.min():.3f}  max {ac.max():.3f}")
    print(f"  |skewness|            : median {np.median(sk):.2f}    max {sk.max():.2f}")
    print(f"  excess kurtosis       : median {np.median(ku):.2f}    max {ku.max():.2f}")
    print(f"  features passing Shapiro normality (p>0.05): {n_norm}/{len(feats)}")
    print(f"  anomaly samples       : {int((lab>0).sum()):,} ({(lab>0).mean()*100:.2f}%)  in {n_episodes} contiguous episode(s)")
    return dict(name=name, n=n, D=len(feats), ac_med=np.median(ac),
                skew_med=np.median(sk), kurt_med=np.median(ku),
                n_normal=n_norm, pct_anom=(lab > 0).mean()*100, episodes=n_episodes)


rows = []

# ---- AI4I 2020 ----
raw = pd.read_csv('ai4i2020_Predictive Maintenance Dataset.csv')
df = raw.copy()
df['Tool wear Rate [min diff]'] = df['Tool wear [min]'].diff().fillna(0)
f_ai4i = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
          'Torque [Nm]', 'Tool wear Rate [min diff]']
rows.append(profile('AI4I 2020', df, f_ai4i, 'Machine failure',
                    note='columns: ' + ', '.join(raw.columns[:5])))
print(f"  UDI monotonic? {bool((raw['UDI'].diff().dropna() == 1).all())}   "
      f"Tool wear resets: {int((raw['Tool wear [min]'].diff() < 0).sum())} times "
      f"-> each row is an independent product, not a time step")

# ---- Industrial Pump ----
with contextlib.redirect_stdout(_buf):
    from experiments.exp_pump_benchmark import load_and_preprocess_pump_data
    dfp = load_and_preprocess_pump_data('Large_Industrial_Pump_Maintenance_Dataset.csv')
f_pump = ['Temperature', 'Vibration', 'Pressure', 'Flow_Rate', 'RPM']
rows.append(profile('Industrial Pump', dfp, f_pump, 'Maintenance_Flag'))
if 'Pump_ID' in dfp.columns:
    print(f"  distinct Pump_ID: {dfp['Pump_ID'].nunique()}  "
          f"-> stream is {dfp['Pump_ID'].nunique()} separate machines concatenated")
del dfp

# ---- Water Pump ----
with contextlib.redirect_stdout(_buf):
    from experiments.exp_waterpump_benchmark import load_and_preprocess_waterpump_data
    dfw, f_wp = load_and_preprocess_waterpump_data('sensor.csv')
rows.append(profile('Water Pump (SCADA)', dfw, list(f_wp), 'failure_label'))
del dfw

# ---- MetroPT-3 ----
with contextlib.redirect_stdout(_buf):
    from experiments.exp_metropt3_benchmark import load_and_label_metropt3
    dfm = load_and_label_metropt3('MetroPT3/MetroPT3_AirCompressor.csv')
f_m = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
rows.append(profile('MetroPT-3', dfm, f_m, 'failure_label'))
del dfm

# ---- TEP Mode 1 ----
import pickle
with open('TEPDataset_M1_M5/TEPDataset_Mode1.pickle', 'rb') as fh:
    tep = pickle.load(fh)
sig, lb = tep['Signals'], tep['Labels']
print(f"\n  [TEP raw shape] {sig.shape} = {sig.shape[0]} independent simulation runs "
      f"x {sig.shape[1]} time steps x {sig.shape[2]} variables")
print(f"  fault classes present: {sorted(set(np.asarray(lb).tolist()))[:12]} ...")
with contextlib.redirect_stdout(_buf):
    from experiments.exp_tep_benchmark import load_and_preprocess_tep_data
    dft, f_t = load_and_preprocess_tep_data('TEPDataset_M1_M5/TEPDataset_Mode1.pickle')
rows.append(profile('TEP Mode 1', dft, list(f_t), 'failure_label',
                    note=f'{sig.shape[0]} runs of {sig.shape[1]} steps flattened into one stream'))

# within-run vs across-run autocorrelation
v = sig[:, :, 0]
within = np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in v[:200] if r.std() > 0])
flat = v.reshape(-1)
across = np.corrcoef(flat[:-1], flat[1:])[0, 1]
print(f"  sensor_00 lag-1 autocorr WITHIN a run: {within:.3f}   after flattening: {across:.3f}")
print(f"  -> {sig.shape[0]-1} artificial discontinuities are introduced at run boundaries")

print('\n\n' + '=' * 88)
print('SUMMARY')
print('=' * 88)
print(pd.DataFrame(rows).round(3).to_string(index=False))
