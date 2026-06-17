import pathlib
import pandas as pd

def test_silver_file_and_basic_quality():
    p = pathlib.Path("data/silver/2026-06-06.parquet")
    assert p.exists(), f"{p} missing"

    df = pd.read_parquet(p)
    assert len(df) == 244
    expected_cols = {"id", "title", "summary", "content", "author", "source", "url", "published", "retrieved_at"}
    assert expected_cols.issubset(set(df.columns))

    # summary should be mostly filled
    pct_missing_summary = df["summary"].isna().mean()
    assert pct_missing_summary == 0

    # timestamps parseable
    assert pd.api.types.is_datetime64_any_dtype(df["published"])
    assert pd.api.types.is_datetime64_any_dtype(df["retrieved_at"])