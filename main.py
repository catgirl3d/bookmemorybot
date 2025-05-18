# main.py

import chromadb
from utils import split_text, embed_texts, ask_gpt
from pathlib import Path
import os
from tqdm import tqdm
import pickle
import time
import sys

# Настройка клиента ChromaDB (новый API)
client = chromadb.PersistentClient(path="db/")

# Обработка всех .txt-файлов в папке data
data_path = Path("data/")
text_files = list(data_path.glob("*.txt"))
assert text_files, "Положи хотя бы один .txt-файл в папку data/"

for file_path in text_files:
    print(f"\nОбработка файла: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Разбиение на фрагменты
    chunks = split_text(text)

    # Кэширование эмбеддингов по батчам
    cache_path = file_path.with_suffix(".pkl")
    embeddings = []
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
        print("Загружены эмбеддинги из кэша.")
    else:
        retry_delay = 1
        for i in tqdm(range(0, len(chunks), 10), desc="Создание эмбеддингов"):
            batch = chunks[i:i+10]
            while True:
                try:
                    batch_embeddings = embed_texts(batch)
                    embeddings.extend(batch_embeddings)
                    with open(cache_path, "wb") as f:
                        pickle.dump(embeddings, f)
                    retry_delay = 1
                    break
                except Exception as e:
                    print(f"Ошибка при создании эмбеддингов: {e}. Повтор через {retry_delay} секунд...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)
        print("Эмбеддинги сохранены в кэш.")

    # Создание отдельной коллекции для файла
    collection_name = file_path.stem.replace(" ", "_").lower()
    collection = client.get_or_create_collection(name=collection_name)

    # Сохранение в коллекцию
existing_ids = set(collection.get(ids=None)["ids"])
for i, chunk in enumerate(tqdm(chunks, desc="Сохранение в базу")):
    doc_id = f"{file_path.stem}_chunk_{i}"
    if doc_id in existing_ids:
        continue
    collection.add(
        documents=[chunk],
        embeddings=[embeddings[i]],
        ids=[doc_id]
    )

# Выбор файла для запроса
print("\nДоступные файлы:")
available_files = [f.name for f in text_files]
for name in available_files:
    print("-", name)

existing_sources = set(available_files)
target_file = input("\nВведите имя файла, по которому искать (или оставьте пустым для всех): ").strip()
if target_file and target_file not in existing_sources:
    print("Файл не найден. Будет выполнен поиск по всем книгам.")
    target_file = ""

# Цикл общения
while True:
    query = input("\nВведите запрос (или 'exit' для выхода): ")
    if query.strip().lower() == "exit":
        print("Выход из программы.")
        break

    query_embedding_result = embed_texts([query])
    if not query_embedding_result:
        print("Ошибка: не удалось получить эмбеддинг запроса. Проверь подключение к API.")
        continue

    query_embed = query_embedding_result[0]

    query_args = {
        "query_embeddings": [query_embed],
        "n_results": 5,
        "include": ["documents"]
    }

    # Выполнение запроса для одной коллекции или всех коллекций
    collections_to_query = []
    if target_file:
        collection_name = Path(target_file).stem.replace(" ", "_").lower()
        collections_to_query = [client.get_or_create_collection(name=collection_name)]
    else:
        collections_to_query = [client.get_or_create_collection(name=Path(f.name).stem.replace(" ", "_").lower()) for f in text_files]

    relevant_chunks = []
    for col in collections_to_query:
        results = col.query(**query_args)
        relevant_chunks.extend(results["documents"][0])

    print("\nРелевантные фрагменты:")
    for doc in relevant_chunks:
        print("\n", doc)

    # Ответ GPT
    answer = ask_gpt(relevant_chunks, query)
    print("\nОтвет GPT:")
    print(answer)
