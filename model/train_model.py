import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
import torch

# 1. Загружаем данные
data = pd.read_csv('data/training_data.csv')
data = data[['text', 'sql']].dropna()
assert len(data) == 11144, f"В датасете {len(data)} строк, ожидается 11144"

# 2. Делим на train/val (например, 90/10)
split_idx = int(len(data) * 0.9)
train_df = data.iloc[:split_idx]
val_df = data.iloc[split_idx:]

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df.rename(columns={'text': 'input_text', 'sql': 'target_text'})),
    'validation': Dataset.from_pandas(val_df.rename(columns={'text': 'input_text', 'sql': 'target_text'}))
})

# 3. Загружаем токенизатор и модель
model_name = 'cointegrated/rut5-small'  # Лучше для русского, чем t5-small
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 4. Токенизация
def preprocess(example):
    # example['input_text'] — это список строк
    inputs = ['translate Russian to SQL: ' + txt for txt in example['input_text']]
    targets = example['target_text']
    return tokenizer(
        inputs,
        text_target=targets,
        max_length=64,
        truncation=True
    )

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset['train'].column_names
)

# 5. Аргументы обучения (адаптированы для маленького датасета)
training_args = TrainingArguments(
    output_dir='data/results_t5_sql',
    per_device_train_batch_size=1,        # Можно уменьшить до 1-2 если не хватает памяти
    per_device_eval_batch_size=1,         # Аналогично
    num_train_epochs=2,                   # Сначала лучше попробовать 1-2 эпохи
    learning_rate=3e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    data_collator=data_collator,
)

# 7. Обучение
trainer.train()

# 8. Сохраняем модель и токенизатор
model.save_pretrained('data/model_t5_sql')
tokenizer.save_pretrained('data/model_t5_sql')
print("Обучение завершено, модель сохранена в data/model_t5_sql")