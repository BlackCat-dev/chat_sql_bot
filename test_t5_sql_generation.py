from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

# Загрузка модели и токенизатора
model = T5ForConditionalGeneration.from_pretrained('data/model_t5_sql')
tokenizer = T5Tokenizer.from_pretrained('data/model_t5_sql')

def generate_sql(text, max_length=128):
    input_text = 'translate Russian to SQL: ' + text
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=4)
    sql = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return sql

def normalize_sql(sql):
    # Приводим к нижнему регистру, убираем лишние пробелы, точку с запятой, табы, переносы строк
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.replace(';', '').strip()
    return sql

def soft_compare_sql(sql1, sql2):
    # Мягкое сравнение: убираем порядок условий WHERE, столбцов SELECT (грубое сравнение)
    sql1_norm = normalize_sql(sql1)
    sql2_norm = normalize_sql(sql2)
    if sql1_norm == sql2_norm:
        return True

    # Попробуем сортировать поля SELECT
    def sort_select_fields(sql):
        m = re.match(r"(select )(.+?)( from .+)", sql)
        if not m:
            return sql
        select, fields, rest = m.groups()
        fields_list = [f.strip() for f in fields.split(',')]
        fields_list.sort()
        return select + ', '.join(fields_list) + rest

    s1 = sort_select_fields(sql1_norm)
    s2 = sort_select_fields(sql2_norm)
    if s1 == s2:
        return True

    # Для WHERE: сортируем условия, если есть
    def sort_where_conditions(sql):
        m = re.match(r"(.+? where )(.+)", sql)
        if not m:
            return sql
        start, rest = m.groups()
        if ' order by ' in rest:
            where, order = rest.split(' order by ', 1)
            order = ' order by ' + order
        else:
            where = rest
            order = ''
        conds = [w.strip() for w in where.split(' and ')]
        conds.sort()
        return start + ' and '.join(conds) + order

    s1 = sort_where_conditions(s1)
    s2 = sort_where_conditions(s2)
    return s1 == s2

# 35 эталонных SQL запросов для таблиц
test_cases = [
    {
        "input": "Показать всех сотрудников",
        "target_sql": "SELECT * FROM Сотрудники;"
    },
    {
        "input": "Вывести ФИО и должность сотрудников из института химии",
        "target_sql": "SELECT ФИО, должность FROM Сотрудники WHERE подразделение = 'Институт химии';"
    },
    {
        "input": "Найти сотрудников, у которых ученое звание Профессор",
        "target_sql": "SELECT Сотрудники.ФИО FROM Сотрудники WHERE ученое_звание = 'Профессор';"
    },
    {
        "input": "Посчитать количество сотрудников в каждом подразделении",
        "target_sql": "SELECT подразделение, COUNT(*) FROM Сотрудники GROUP BY подразделение;"
    },
    {
        "input": "Показать сотрудников старше 50 лет",
        "target_sql": "SELECT * FROM Сотрудники WHERE возраст > 50;"
    },
    {
        "input": "Вывести ФИО сотрудников из кафедры нанофизики",
        "target_sql": "SELECT ФИО FROM Сотрудники WHERE оргструктура = 'Кафедра нанофизики';"
    },
    {
        "input": "Показать всех доцентов",
        "target_sql": "SELECT * FROM Сотрудники WHERE должность = 'Доцент';"
    },
    {
        "input": "Найти сотрудников с совместительством",
        "target_sql": "SELECT * FROM Сотрудники WHERE вид_занятости = 'Совместительство';"
    },
    {
        "input": "Вывести средний возраст сотрудников по подразделениям",
        "target_sql": "SELECT подразделение, AVG(возраст) FROM Сотрудники GROUP BY подразделение;"
    },
    {
        "input": "Показать сотрудников без ученой степени",
        "target_sql": "SELECT * FROM Сотрудники WHERE ученая_степень = '';"
    },
    {
        "input": "Найти сотрудников на полной ставке",
        "target_sql": "SELECT * FROM Сотрудники WHERE ставка = 1;"
    },
    {
        "input": "Вывести всех сотрудников с категорией персонала Преподавательский состав",
        "target_sql": "SELECT * FROM Сотрудники WHERE категория_персонала = 'Преподавательский состав';"
    },
    {
        "input": "Показать всех сотрудников кафедры архитектурного проектирования",
        "target_sql": "SELECT * FROM Сотрудники WHERE оргструктура IN ('Кафедра архитектурного проектирования');"
    },
    {
        "input": "Посчитать количество профессоров в каждом подразделении",
        "target_sql": "SELECT подразделение, COUNT(*) FROM Сотрудники WHERE должность = 'Профессор' GROUP BY подразделение;"
    },
    {
        "input": "Показать научные работы за 2022 год",
        "target_sql": "SELECT * FROM Научные_работы WHERE год = 2022;"
    },
    {
        "input": "Вывести названия научных работ и их авторов",
        "target_sql": "SELECT название, авторы FROM Научные_работы;"
    },
    {
        "input": "Найти научные работы, опубликованные в журнале нанотехнологий",
        "target_sql": "SELECT * FROM Научные_работы WHERE название_журнала = 'Журнал нанотехнологий';"
    },
    {
        "input": "Показать количество научных работ по каждому автору",
        "target_sql": "SELECT автор, COUNT(*) FROM Научные_работы GROUP BY автор;"
    },
    {
        "input": "Показать научные работы, где база цитирования Scopus",
        "target_sql": "SELECT * FROM Научные_работы WHERE база_цитирования = 'Scopus';"
    },
    {
        "input": "Найти научные работы без вторых названий",
        "target_sql": "SELECT * FROM Научные_работы WHERE второе_название = '';"
    },
    {
        "input": "Посчитать количество публикаций в каждом журнале",
        "target_sql": "SELECT название_журнала, COUNT(*) FROM Научные_работы GROUP BY название_журнала;"
    },
    {
        "input": "Показать публикации, где год после 2020",
        "target_sql": "SELECT * FROM Научные_работы WHERE год > 2020;"
    },
    {
        "input": "Вывести ссылки на публикации автора Иванов И.И.",
        "target_sql": "SELECT ссылка FROM Научные_работы WHERE автор = 'Иванов И.И.';"
    },
    {
        "input": "Показать научные работы, где указано несколько авторов",
        "target_sql": "SELECT * FROM Научные_работы WHERE авторы LIKE '%,%';"
    },
    {
        "input": "Вывести все научные работы, опубликованные в РИНЦ",
        "target_sql": "SELECT * FROM Научные_работы WHERE база_цитирования = 'РИНЦ';"
    },
    {
        "input": "Показать всех профессоров",
        "target_sql": "SELECT * FROM Сотрудники WHERE должность LIKE '%Профессор%';"
    },
    {
        "input": "Какие научные работы опубликовал Иванов",
        "target_sql": "SELECT * FROM Научные_работы WHERE автор LIKE '%Иванов%';"
    },
    {
        "input": "Научные статьи за 2022 год",
        "target_sql": "SELECT * FROM Научные_работы WHERE год = 2022;"
    },
    {
        "input": "Показать все научные работы",
        "target_sql": "SELECT * FROM Научные_работы;"
    },
    {
        "input": "Сотрудники из института физики",
        "target_sql": "SELECT * FROM Сотрудники WHERE подразделение LIKE '%физики%';"
    },
    {
        "input": "Все публикации за 2022 год",
        "target_sql": "SELECT * FROM Научные_работы WHERE год = 2022;"
    },
    {
        "input": "Сотрудники института химии",
        "target_sql": "SELECT * FROM Сотрудники WHERE подразделение LIKE '%химии%';"
    },
    {
        "input": "Профессора института строительства",
        "target_sql": "SELECT * FROM Профессора WHERE институт = 'строительство';"
    },
    {
        "input": "Кто работает в институте экономики",
        "target_sql": "SELECT * FROM Сотрудники WHERE подразделение = 'Экономика';"
    },
    {
        "input": "Сотрудники с ученой степенью кандидата наук",
        "target_sql": "SELECT * FROM Сотрудники WHERE ученая_степень = 'Кандидат наук';"
    }
]

print("=== Тестирование генерации SQL (мягкое сравнение) ===\n")

correct = 0
total = len(test_cases)

for idx, case in enumerate(test_cases, 1):
    user_query = case["input"]
    reference_sql = case["target_sql"]
    generated_sql = generate_sql(user_query)

    match = soft_compare_sql(generated_sql, reference_sql)

    print(f"{idx}. Вопрос: {user_query}")
    print(f"   Эталонный SQL: {reference_sql}")
    print(f"   Сгенерированный SQL: {generated_sql}")
    print(f"   Совпадение (soft): {match}\n")

    if match:
        correct += 1

print(f"Всего совпадений (soft): {correct} из {total} ({correct/total*100:.1f}%)")