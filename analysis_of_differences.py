import pandas as pd
from difflib import ndiff
from IPython.display import display, HTML

def highlight_differences(expected: str, predicted: str) -> str:
    """
    Возвращает HTML-разметку с подсветкой отличий между эталонным и предсказанным SQL.
    - Удалённые токены помечаются красным.
    - Добавленные — зелёным.
    - Совпадающие — без изменений.
    """
    diff = ndiff(expected.split(), predicted.split())
    result = []
    for token in diff:
        if token.startswith("- "):
            result.append(f'<span style="background-color:#fdd; color:#900">[- {token[2:]}]</span>')
        elif token.startswith("+ "):
            result.append(f'<span style="background-color:#dfd; color:#090">[+ {token[2:]}]</span>')
        elif token.startswith("? "):
            continue  # Подсказка символов (не используется)
        else:
            result.append(token[2:])
    return " ".join(result)

# === Пример анализа на датасете ===
def analyze_errors(df: pd.DataFrame, limit: int = 5):
    """
    Показывает расхождения для первых N примеров, где предсказание ≠ эталон.
    """
    errors = df[df['Predicted SQL'].str.strip().str.lower() != df['Expected SQL'].str.strip().str.lower()]
    display(HTML("<h3>Примеры различий (предсказание ≠ эталон)</h3>"))

    for i, row in errors.head(limit).iterrows():
        html = f"""
        <b>Входной текст:</b><br>{row['Input']}<br><br>
        <b>Различия:</b><br>{highlight_differences(row['Expected SQL'], row['Predicted SQL'])}<br><br>
        <hr>
        """
        display(HTML(html))

# === Загрузка результатов (если ещё не загружены) ===
df = pd.read_csv("data/evaluation_results.csv")  # Файл с колонками: Input, Expected SQL, Predicted SQL
analyze_errors(df, limit=5)
