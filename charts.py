import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Загружаем результаты
results = pd.read_csv('data/evaluation_results.csv')

# Вычисляем BLEU для каждой строки
def calc_bleu(row):
    ref = row['Expected SQL'].strip().lower().split()
    hyp = row['Predicted SQL'].strip().lower().split()
    return sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method1)

results['BLEU'] = results.apply(calc_bleu, axis=1)

# Добавляем флаг точного совпадения
results['Exact Match'] = results.apply(
    lambda row: int(row['Expected SQL'].strip().lower() == row['Predicted SQL'].strip().lower()), axis=1
)

# === График BLEU ===
plt.figure(figsize=(10, 6))
sns.histplot(results['BLEU'], bins=20, kde=True, color='skyblue')
plt.title('Распределение BLEU-оценок')
plt.xlabel('BLEU score')
plt.ylabel('Количество примеров')
plt.grid(True)
plt.tight_layout()
plt.savefig('data/bleu_distribution.png')
plt.show()

# === График точных совпадений ===
match_counts = results['Exact Match'].value_counts().sort_index()
labels = ['Не совпало', 'Совпало']
plt.figure(figsize=(6, 6))
plt.pie(match_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'lightgreen'])
plt.title('Доля точных совпадений SQL')
plt.tight_layout()
plt.savefig('data/exact_match_pie.png')
plt.show()

# === Визуализация первых N предсказаний ===
N = 10
subset = results.head(N)
plt.figure(figsize=(12, N * 0.7))
for i, row in subset.iterrows():
    plt.text(0.01, 1 - i * 0.1, f"Input: {row['Input']}", fontsize=9, color='black')
    plt.text(0.02, 0.97 - i * 0.1, f"Expected: {row['Expected SQL']}", fontsize=8, color='green')
    plt.text(0.02, 0.94 - i * 0.1, f"Predicted: {row['Predicted SQL']}", fontsize=8, color='blue')
plt.axis('off')
plt.title('Первые N предсказаний модели и эталонные запросы', loc='left')
plt.tight_layout()
plt.savefig('data/sql_examples.png')
plt.show()
