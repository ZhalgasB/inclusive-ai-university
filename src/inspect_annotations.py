import json
from pathlib import Path

import pandas as pd
from benchmark import validate_rows

# 1. Находим файл с данными.
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "sample" / "annotations.jsonl"

# 2. Читаем JSONL и проверяем исходные записи.
with input_file.open(encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]

validate_rows(rows)
print("Проверка структуры: OK")

# 3. Превращаем проверенные записи в таблицу Pandas.
df = pd.DataFrame(rows)
print(f"Всего записей: {len(df)}")

columns = [
    "item_id",
    "track",
    "condition",
    "status",
    "preparation_time_s",
    "critical_error",
]
print(df[columns].to_string(index=False))

# 4. Помечаем сбои, сохраняя все строки.
df["is_failed"] = df["status"].eq("failed")

# 5. Считаем доли ошибок по группам.
guardrails = df.groupby(["track", "condition"]).agg(
    attempts=("item_id", "size"),
    failure_rate_pct=("is_failed", "mean"),
    critical_error_rate_pct=("critical_error", "mean"),
)

rate_columns = ["failure_rate_pct", "critical_error_rate_pct"]
guardrails[rate_columns] = guardrails[rate_columns] * 100

print("\nЧисло попыток и доли ошибок (%):")
print(guardrails.to_string())