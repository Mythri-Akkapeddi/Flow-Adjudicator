"""Throwaway: estimate the clock offset between corrected and original timestamps."""
import duckdb

from flowadj.config import PROCESSED_DIR

o = str(PROCESSED_DIR / "flows_original.parquet")
c = str(PROCESSED_DIR / "flows_corrected_raw.parquet")

sql = f"""
WITH o AS (
    SELECT flow_id, MIN(timestamp) AS ts, MIN(capture_day) AS day
    FROM read_parquet('{o}') GROUP BY flow_id HAVING COUNT(*) = 1
),
c AS (
    SELECT flow_id, MIN(timestamp) AS ts
    FROM read_parquet('{c}') GROUP BY flow_id HAVING COUNT(*) = 1
),
j AS (
    SELECT o.day AS day, round(date_diff('minute', o.ts, c.ts) / 60.0) AS offset_hours
    FROM o JOIN c USING (flow_id)
),
g AS (
    SELECT day, offset_hours, COUNT(*) AS n FROM j GROUP BY 1, 2
)
SELECT * FROM g
QUALIFY row_number() OVER (PARTITION BY day ORDER BY n DESC) <= 3
ORDER BY day, n DESC
"""
print(duckdb.connect().execute(sql).df().to_string(index=False))