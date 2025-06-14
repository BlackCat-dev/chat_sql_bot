from flask import Flask, render_template, request
import sqlite3
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datetime import datetime
import random
from log_interaction import log_interaction

ENCOURAGEMENTS = [
    "Отличный запрос! Так держать!",
    "Вы молодец, продолжайте в том же духе!",
    "Ваши вопросы вдохновляют!",
    "Не бойтесь экспериментировать — всё получится!",
    "С каждым разом у вас получается всё лучше!",
    "Вы на верном пути к успеху!",
    "Супер! Ваш запрос обработан.",
    "Спасибо за интересный вопрос!",
    "Замечательно! Если нужно ещё что-то — пишите.",
    "У вас очень хорошо получается! Продолжайте!"
]

app = Flask(__name__)

# Загрузка модели и токенизатора
model_dir = "data/model_t5_sql"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
device = torch.device("cpu")  # CPU для совместимости на Mac/Windows
model = model.to(device)
model.eval()

def get_sql_query(user_input):
    prompt = f"translate Russian to SQL: {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

def execute_query(query):
    conn = sqlite3.connect("data/database.db")
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
    except Exception as e:
        results = [[f"Ошибка: {e} (SQL: {query})"]]
        columns = ["Ошибка"]
    conn.close()
    return columns, results

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    columns = []
    encouragement = None
    if request.method == "POST":
        user_input = request.form["query"]
        sql_query = get_sql_query(user_input)
        print(sql_query)
        columns, response = execute_query(sql_query)
        encouragement = random.choice(ENCOURAGEMENTS)
        log_interaction(user_input, sql_query)
    return render_template("index.html", response=response, columns=columns, year=datetime.now().year, encouragement=encouragement)

if __name__ == "__main__":
    app.run(debug=True)