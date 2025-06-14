import csv
import os
from datetime import datetime
from typing import Optional

LOG_FILE = "logs/interaction_log.csv"

# Проверка, существует ли лог, иначе создаём с заголовком
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "user_input", "predicted_sql", "sql_valid", "source", "notes"])

def log_interaction(
    user_input: str,
    predicted_sql: str,
    sql_valid: Optional[bool] = None,
    source: str = "inference",
    notes: Optional[str] = ""
):
    """
    Логирует взаимодействие пользователя с моделью.

    :param user_input: Входной текст на естественном языке
    :param predicted_sql: Предсказанный SQL
    :param sql_valid: True/False, если выполнялась проверка синтаксиса SQL
    :param source: "inference", "api", "batch_eval", "interactive", и т.п.
    :param notes: Доп. комментарии (например, ошибка, feedback)
    """
    with open(LOG_FILE, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_input.strip(),
            predicted_sql.strip(),
            sql_valid if sql_valid is not None else "",
            source,
            notes
        ])
