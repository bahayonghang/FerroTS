"""Profile the Yangquan rotary kiln twin sample and the cement-twin-v1 protocol.

Usage: PYTHONIOENCODING=utf-8 uv run --with polars python tools/datasets/profile_cement.py

Prints statistics only. It does not print raw data rows.
Parameter definitions: docs/datasets/cement-yangquan.md.
"""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "阳泉回转窑联合训练孪生样本"

# cement-twin-v1 parameters.
MIN_SAMPLES = 3  # valid raw samples in one minute for a valid 1 min mean
RUN_COL = "窑头煤反馈"
RUN_THRESHOLD = 5.0  # running minute: 1 min mean of RUN_COL > RUN_THRESHOLD
MAX_GAP_MIN = 5  # a missing run longer than this ends a running segment
MIN_SEGMENT_MIN = 30
MAX_STALE_MIN = 10  # causal forward fill limit inside a running segment
SPLIT = (0.7, 0.1, 0.2)
L = 96
HORIZONS = (15, 30, 60)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def table(rows: list[tuple], header: list[str]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs where mask is True."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def running_segments(state: np.ndarray) -> list[tuple[int, int]]:
    """state: 1 running, 0 stopped, -1 missing. Returns [start, end) minute indices."""
    segs = []
    for a, b in runs(state != 0):  # stopped minutes always end a segment
        start = None
        miss = 0
        for i in range(a, b):
            if state[i] == 1:
                if start is None:
                    start = i
                last = i
                miss = 0
            else:
                miss += 1
                if miss > MAX_GAP_MIN and start is not None:
                    segs.append((start, last + 1))
                    start = None
        if start is not None:
            segs.append((start, last + 1))
    return [(a, b) for a, b in segs if b - a >= MIN_SEGMENT_MIN]


def main() -> None:
    files = [DATA_DIR / "data_in.csv", DATA_DIR / "data_out.csv"]
    print("## files\n")
    table([(f.name, f.stat().st_size, sha256(f)) for f in files], ["file", "bytes", "sha256"])

    # Raw 10 s data, in file order.
    df = pl.read_csv(files[0], infer_schema_length=0)
    cols = df.columns[1:]
    t = df["DateTime"].str.to_datetime()
    d = t.diff().dt.total_seconds().drop_nulls()
    print("## raw\n")
    table(
        [
            ("rows", df.height),
            ("variables", len(cols)),
            ("first time", t.min()),
            ("last time", t.max()),
            ("interval == 10 s", int((d == 10).sum())),
            ("interval 9-13 s, not 10 s", int(((d >= 9) & (d <= 13) & (d != 10)).sum())),
            ("interval <= 0 s", int((d <= 0).sum())),
            ("rows with a duplicated timestamp", int(t.is_duplicated().sum())),
            ("gaps > 60 s", int((d > 60).sum())),
            ("gaps > 1 h", int((d > 3600).sum())),
            ("max gap h", round(d.max() / 3600, 1)),
            (f"rows with {RUN_COL} > {RUN_THRESHOLD}", round(float((df[RUN_COL].cast(pl.Float64) > RUN_THRESHOLD).mean()), 4)),
        ],
        ["item", "value"],
    )
    rows = []
    for c in cols:
        s = df[c]
        num = s.cast(pl.Float64, strict=False)
        bad = int((num.is_null() & s.is_not_null()).sum())
        q = [round(num.quantile(p), 3) for p in (0.05, 0.5, 0.95)]
        rows.append((c, s.null_count(), bad, round(num.min(), 3), *q, round(num.max(), 3)))
    table(rows, ["variable", "null", "non-numeric", "min", "p05", "p50", "p95", "max"])

    # 1 min grid. Sort by time; keep the first row of a duplicated timestamp.
    df = (
        df.with_columns(t.alias("t"), *[pl.col(c).cast(pl.Float64, strict=False) for c in cols])
        .drop("DateTime")
        .sort("t", maintain_order=True)
        .unique("t", keep="first", maintain_order=True)
    )
    agg = df.group_by(pl.col("t").dt.truncate("1m").alias("minute")).agg(
        [pl.col(c).mean().alias(c) for c in cols] + [pl.col(c).count().alias(c + "#n") for c in cols]
    )
    grid = pl.datetime_range(agg["minute"].min(), agg["minute"].max(), "1m", eager=True, time_unit=agg.schema["minute"].time_unit)
    m = pl.DataFrame({"minute": grid}).join(agg, on="minute", how="left").sort("minute")
    minutes = m["minute"]
    n = m.height
    x = np.column_stack(
        [np.where(m[c + "#n"].fill_null(0).to_numpy() >= MIN_SAMPLES, m[c].fill_null(np.nan).to_numpy(), np.nan) for c in cols]
    )
    run_x = x[:, cols.index(RUN_COL)]
    state = np.where(np.isnan(run_x), -1, np.where(run_x > RUN_THRESHOLD, 1, 0))
    print("## 1 min grid\n")
    table(
        [
            ("rows removed by timestamp de-duplication", len(t) - df.height),
            ("grid minutes", n),
            ("minutes without raw rows", n - agg.height),
            ("running minutes", int((state == 1).sum())),
            ("stopped minutes", int((state == 0).sum())),
            (f"minutes with {RUN_COL} missing", int((state == -1).sum())),
        ],
        ["item", "value"],
    )
    table([(c, int(np.isnan(x[:, j]).sum())) for j, c in enumerate(cols)], ["variable", "missing minutes"])

    # Running segments.
    segs = running_segments(state)
    lens = np.array([b - a for a, b in segs])
    print("## running segments\n")
    pct = np.percentile(lens, [0, 10, 50, 90, 100]).round().astype(int)
    table(
        [("segments", len(segs)), ("minutes", int(lens.sum()))] + [(f"length p{p} min", v) for p, v in zip((0, 10, 50, 90, 100), pct)],
        ["item", "value"],
    )

    # Causal forward fill inside each segment, at most MAX_STALE_MIN minutes.
    filled = np.full_like(x, np.nan)
    fill_count = np.zeros(len(cols), dtype=int)
    for a, b in segs:
        seg = x[a:b].copy()
        for j in range(len(cols)):
            last, age = np.nan, 0
            for i in range(b - a):
                if np.isnan(seg[i, j]):
                    age += 1
                    if not np.isnan(last) and age <= MAX_STALE_MIN:
                        seg[i, j] = last
                        fill_count[j] += 1
                else:
                    last, age = seg[i, j], 0
        filled[a:b] = seg
    inside = np.zeros(n, dtype=bool)
    for a, b in segs:
        inside[a:b] = True
    table(
        [
            (c, int((np.isnan(x[:, j]) & inside).sum()), int(fill_count[j]), int((np.isnan(filled[:, j]) & inside).sum()))
            for j, c in enumerate(cols)
        ],
        ["variable", "missing in segments", "filled", "missing after fill"],
    )

    # Usable runs: minutes inside a segment with all variables present after fill.
    usable = inside & ~np.isnan(filled).any(axis=1)
    usable_runs = runs(usable)

    # Split whole segments by cumulative segment minutes; boundaries fall between segments.
    cum = np.cumsum(lens)
    total = cum[-1]
    split_of = []
    for k, (a, b) in enumerate(segs):
        mid = cum[k] - lens[k] / 2  # a segment goes to the split that holds its midpoint
        split_of.append(0 if mid < SPLIT[0] * total else 1 if mid < (SPLIT[0] + SPLIT[1]) * total else 2)
    print("## splits\n")
    rows = []
    for s, name in enumerate(("train", "val", "test")):
        ks = [k for k in range(len(segs)) if split_of[k] == s]
        lo, hi = segs[ks[0]][0], segs[ks[-1]][1]
        ur = [(max(a, lo), min(b, hi)) for a, b in usable_runs if a < hi and b > lo]
        win = [sum(max(0, b - a - L - h + 1) for a, b in ur) for h in HORIZONS]
        seg_min = int(sum(lens[k] for k in ks))
        rows.append(
            (name, len(ks), seg_min, round(seg_min / total, 3), minutes[lo], minutes[hi - 1], int(sum(b - a for a, b in ur)), len(ur), *win)
        )
    table(rows, ["split", "segments", "segment min", "fraction", "first minute", "last minute", "usable min", "usable runs"] + [f"windows H={h}" for h in HORIZONS])

    # Scaler statistics: train split usable minutes, population std.
    ks = [k for k in range(len(segs)) if split_of[k] == 0]
    lo, hi = segs[ks[0]][0], segs[ks[-1]][1]
    tr = filled[lo:hi][usable[lo:hi]]
    print("## train scaler (population std)\n")
    table([(c, round(float(tr[:, j].mean()), 4), round(float(tr[:, j].std()), 4)) for j, c in enumerate(cols)], ["variable", "mean", "std"])


if __name__ == "__main__":
    main()
