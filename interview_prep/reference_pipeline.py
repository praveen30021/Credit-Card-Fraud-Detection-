"""
Reference pipeline that rebuilds the CV version of the project, so you can get REAL numbers.

What it does (same order as the architecture image):
  load -> features -> split FIRST -> SMOTENC on train only -> XGBoost (scale_pos_weight)
  -> pick threshold with F2 on a validation part -> evaluate ONCE on the untouched test set.
It also trains a plain baseline XGBoost (no imbalance handling, threshold 0.5) and prints
how many frauds each model missed, so you can see where a "missed frauds down X%" number comes from.

Your numbers may NOT equal the CV numbers (F2 0.88, AUC 0.92, 16.67%). Whatever this prints is
what this code really gives. Only quote numbers you have actually seen from your own run.

Usage:
  python reference_pipeline.py --csv fraudTest.csv            # full file
  python reference_pipeline.py --csv fraudTest.csv fraudTrain.csv
  python reference_pipeline.py --synthetic                    # quick smoke test, fake data
Needs: pandas numpy scikit-learn xgboost imbalanced-learn
"""
import argparse

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             fbeta_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42
NUMERIC = ["amt", "distance", "age", "time_diff", "hour", "month_number", "week_number",
           "count_1_day", "count_7_days", "count_30_days",
           "count_1_merch_day", "count_7_merch_days", "count_30_merch_days"]
BINARY = ["hourEnc", "gender", "first_txn"]


def haversine_miles(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def rolling_counts(df, key, prefix):
    """Past-only counts in the last 1/7/30 days per `key` (current row is not counted)."""
    df = df.sort_values([key, "trans_date_trans_time"])
    for days in (1, 7, 30):
        r = (df.groupby(key)
               .rolling(f"{days}D", on="trans_date_trans_time")["amt"].count() - 1)
        df[f"{prefix}{days}"] = r.to_numpy()
    return df


def build_features(df):
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["month_number"] = df["trans_date_trans_time"].dt.month
    df["week_number"] = df["trans_date_trans_time"].dt.dayofweek
    df["hourEnc"] = ((df["hour"] < 5) | (df["hour"] > 21)).astype(int)   # night flag (0/1)
    df["gender"] = (df["gender"] == "M").astype(int)
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.total_seconds() / (365.25 * 86400)
    df["distance"] = haversine_miles(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    # time since previous transaction of the same card (total_seconds, not .seconds)
    gap = df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
    df["first_txn"] = gap.isna().astype(int)
    df["time_diff"] = gap.fillna(0)

    df = rolling_counts(df, "cc_num", "count_")          # count_1, count_7, count_30
    df = df.rename(columns={"count_1": "count_1_day", "count_7": "count_7_days", "count_30": "count_30_days"})
    df = rolling_counts(df, "merchant", "count_m")
    df = df.rename(columns={"count_m1": "count_1_merch_day", "count_m7": "count_7_merch_days",
                            "count_m30": "count_30_merch_days"})
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    cats = pd.get_dummies(df["category"], prefix="category", dtype=int)
    X = pd.concat([df[NUMERIC + BINARY], cats], axis=1)
    return X, df["is_fraud"].astype(int), df["trans_date_trans_time"]


def f2_threshold(y_val, prob_val):
    """Scan thresholds on VALIDATION data and return the one with best F2."""
    best_t, best = 0.5, -1.0
    for t in np.arange(0.01, 1.0, 0.01):
        score = fbeta_score(y_val, prob_val >= t, beta=2, zero_division=0)
        if score > best:
            best_t, best = t, score
    return best_t, best


def report(name, y_true, prob, t):
    pred = prob >= t
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f"\n[{name}] threshold={t:.2f}")
    print(f"  TN={tn} FP={fp} FN(missed frauds)={fn} TP={tp}")
    print(f"  precision={precision_score(y_true, pred, zero_division=0):.3f}  "
          f"recall={recall_score(y_true, pred):.3f}  "
          f"F2={fbeta_score(y_true, pred, beta=2):.3f}")
    print(f"  ROC-AUC={roc_auc_score(y_true, prob):.3f}  PR-AUC={average_precision_score(y_true, prob):.3f}")
    return fn


def synthetic(n=30000, fraud_rate=0.01):
    """Fake data with the same columns, ONLY to test that the code runs."""
    rng = np.random.default_rng(SEED)
    t0 = pd.Timestamp("2020-06-21")
    df = pd.DataFrame({
        "trans_date_trans_time": t0 + pd.to_timedelta(rng.integers(0, 190 * 86400, n), unit="s"),
        "cc_num": rng.integers(1, 300, n), "merchant": rng.integers(1, 100, n).astype(str),
        "category": rng.choice(["food", "travel", "shopping_net", "home"], n),
        "amt": rng.gamma(2, 30, n), "gender": rng.choice(["M", "F"], n),
        "lat": rng.uniform(30, 45, n), "long": rng.uniform(-120, -75, n),
        "merch_lat": rng.uniform(30, 45, n), "merch_long": rng.uniform(-120, -75, n),
        "dob": pd.Timestamp("1980-01-01"), "is_fraud": (rng.random(n) < fraud_rate).astype(int)})
    df.loc[df.is_fraud == 1, "amt"] *= 8
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*", help="one or more Kaggle csv files")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--split", choices=["stratified", "time"], default="stratified")
    ap.add_argument("--smote-ratio", type=float, default=0.1,
                    help="fraud/normal ratio after SMOTENC (0.1 = fraud becomes ~9%% of train)")
    args = ap.parse_args()

    df = synthetic() if args.synthetic else pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    print(f"rows={len(df):,}  fraud rate={df['is_fraud'].mean():.4%}")
    X, y, ts = build_features(df)

    # 1) SPLIT FIRST. test = untouched, original class ratio.
    if args.split == "time":
        cut = ts.quantile(0.8)
        tr, te = (ts <= cut).to_numpy(), (ts > cut).to_numpy()
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    # validation part (for threshold) is carved out of TRAIN
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=SEED)
    print(f"train={len(X_fit):,} val={len(X_val):,} test={len(X_te):,}  frauds(test)={int(y_te.sum())}")

    # 2) BASELINE: plain XGBoost, no imbalance handling, threshold 0.5
    base = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric="aucpr",
                         tree_method="hist", random_state=SEED, n_jobs=-1)
    base.fit(X_fit, y_fit)
    fn_base = report("BASELINE xgb, no imbalance handling", y_te, base.predict_proba(X_te)[:, 1], 0.5)

    # 3) SMOTENC on TRAIN ONLY. Binary / one-hot columns are treated as categorical.
    cat_idx = [i for i, c in enumerate(X_fit.columns) if c in BINARY or c.startswith("category_")]
    sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=args.smote_ratio,
                 k_neighbors=min(5, int(y_fit.sum()) - 1), random_state=SEED)
    X_res, y_res = sm.fit_resample(X_fit, y_fit)
    print(f"\nafter SMOTENC (train only): normal={(y_res == 0).sum():,} fraud={(y_res == 1).sum():,}")

    # 4) scale_pos_weight on what is STILL imbalanced after SMOTENC (computed from train)
    spw = (y_res == 0).sum() / (y_res == 1).sum()
    print(f"scale_pos_weight = normal/fraud on resampled train = {spw:.2f}")
    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, scale_pos_weight=spw,
                          eval_metric="aucpr", tree_method="hist", random_state=SEED, n_jobs=-1)
    model.fit(X_res, y_res)

    # 5) threshold by F2 on VALIDATION (never on test)
    t, f2_val = f2_threshold(y_val, model.predict_proba(X_val)[:, 1])
    print(f"best threshold on validation = {t:.2f}  (F2 on validation = {f2_val:.3f})")

    # 6) evaluate ONCE on the untouched test set
    fn_final = report("FINAL xgb + SMOTENC + scale_pos_weight + F2 threshold", y_te, model.predict_proba(X_te)[:, 1], t)

    if fn_base:
        print(f"\nMissed frauds: baseline={fn_base} -> final={fn_final}  "
              f"=> change = {(fn_base - fn_final) / fn_base:.2%}   (this is how a 'missed frauds down X%' number is computed)")
    print("\nRemember: run it with different seeds / --split time and report the range, not one lucky number.")


if __name__ == "__main__":
    main()
