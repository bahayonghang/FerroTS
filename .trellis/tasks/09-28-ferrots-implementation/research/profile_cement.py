# 阳泉回转窑数据盘点脚本。用法：PYTHONIOENCODING=utf-8 uv run --with polars python profile_cement.py
# 只输出统计量，不输出原始行（重复时间戳示例除外）。
import polars as pl
pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(40)
base = r"../../../../data/阳泉回转窑联合训练孪生样本/"
df = pl.read_csv(base + "data_in.csv", try_parse_dates=True, infer_schema_length=0)
cols = df.columns[1:]
t = df["DateTime"].str.to_datetime()
d = t.diff().dt.total_seconds().drop_nulls()
print("IN gaps>60s:", (d>60).sum(), " >3600s:", (d>3600).sum(), " max gap h:", d.max()/3600, " neg:", (d<=0).sum())
big = pl.DataFrame({"t": t[1:], "gap_h": d/3600}).filter(pl.col("gap_h")>1).sort("gap_h", descending=True).head(10)
print(big)
rows=[]
for c in cols:
    s = df[c]
    num = s.cast(pl.Float64, strict=False)
    bad = (num.is_null() & s.is_not_null()).sum()
    rows.append((c, s.null_count(), bad, num.min(), num.quantile(0.05), num.quantile(0.5), num.quantile(0.95), num.max()))
    if bad: print("non-numeric in", c, s.filter(num.is_null() & s.is_not_null()).unique().head(5).to_list())
print(pl.DataFrame(rows, schema=["col","null","nonnum","min","p05","p50","p95","max"], orient="row"))
# kiln running proxy: coal feed > 5
coal = df["窑头煤反馈"].cast(pl.Float64)
print("coal>5 fraction:", (coal>5).mean())
# f_cao 部分已移除：扩充样本目录于 2026-09-28 删除。
