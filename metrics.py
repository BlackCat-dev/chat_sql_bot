import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Загружаем модель и токенизатор
model_path = 'data/model_t5_sql'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
model.eval()

# Загружаем и подготавливаем данные
data = pd.read_csv('data/training_data.csv')[['text', 'sql']].dropna()
train, test = train_test_split(data, test_size=0.1, random_state=42)

# Функция для генерации SQL по тексту
def generate_sql(input_text):
    input_ids = tokenizer.encode("translate Russian to SQL: " + input_text, return_tensors="pt", max_length=64, truncation=True)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Сравнение предсказаний с эталоном
y_true = test['sql'].tolist()
y_pred = [generate_sql(txt) for txt in test['text'].tolist()]

# Приведение строк к нижнему регистру и удаление лишних пробелов
y_true_clean = [s.strip().lower() for s in y_true]
y_pred_clean = [s.strip().lower() for s in y_pred]

# Accuracy — строгое совпадение
exact_match_accuracy = accuracy_score(y_true_clean, y_pred_clean)

# BLEU — оценка сходства текстов
bleu_scores = [sentence_bleu([ref.split()], hyp.split(), smoothing_function=SmoothingFunction().method1)
               for ref, hyp in zip(y_true_clean, y_pred_clean)]
avg_bleu = sum(bleu_scores) / len(bleu_scores)

print(f"Exact Match Accuracy: {exact_match_accuracy:.2%}")
print(f"Average BLEU Score: {avg_bleu:.2f}")

# Можно также сохранить результаты
results = pd.DataFrame({
    'Input': test['text'],
    'Expected SQL': y_true,
    'Predicted SQL': y_pred
})
results.to_csv('data/evaluation_results.csv', index=False)