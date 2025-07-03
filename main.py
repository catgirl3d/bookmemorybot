import argparse
import asyncio
import chromadb
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import concurrent.futures

from tqdm import tqdm

from utils import split_text, embed_texts, ask_gpt

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Загрузка конфигурации
CONFIG_PATH = Path("config.json")
DEFAULT_CONFIG = {
    "batch_size": 10,
    "neighbor_window": 1,
    "search_results": 3,
    "db_path": "db/",
    "data_path": "data/",
    "retry_max_delay": 60
}

def load_config() -> Dict:
    """Загрузка конфигурации из файла или создание дефолтной"""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Ошибка при чтении {CONFIG_PATH}. Используем значения по умолчанию.")
            return DEFAULT_CONFIG
    else:
        # Создаем дефолтный конфиг
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        return DEFAULT_CONFIG

config = load_config()

# Инициализация клиента ChromaDB
client = chromadb.PersistentClient(path=config["db_path"])

def get_file_hash(file_path: Path) -> str:
    """Получение хеша файла для определения изменений"""
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Ошибка при вычислении хеша файла {file_path}: {e}")
        return ""

def expand_chunks_by_neighbors(chunks: List[str], all_chunks: List[str], window: int = 1) -> List[str]:
    """
    Расширяет найденные чанки соседними для лучшего контекста
    
    Args:
        chunks: Список найденных фрагментов
        all_chunks: Полный список всех фрагментов из ОДНОГО документа
        window: Размер окна соседей (сколько фрагментов брать с каждой стороны)
        
    Returns:
        Расширенный список фрагментов
    """
    # Логируем входные данные для отладки
    logger.debug(f"expand_chunks_by_neighbors: получено {len(chunks)} чанков, window={window}")
    logger.debug(f"all_chunks содержит {len(all_chunks)} элементов")
    
    # Если all_chunks пуст или window <= 0, просто возвращаем исходные чанки
    if not all_chunks or window <= 0 or not chunks:
        logger.debug("Пустой список all_chunks или window <= 0, возвращаем исходные чанки")
        return chunks
    
    result = set(chunks)
    chunks_added = 0
    
    # Для каждого чанка находим его индекс в исходном списке
    for chunk in chunks:
        try:
            # Ищем точное совпадение чанка в исходном списке
            idx = all_chunks.index(chunk)
            logger.debug(f"Найден чанк с индексом {idx}")
            
            # Добавляем соседние чанки в пределах заданного окна
            for offset in range(-window, window + 1):
                if offset == 0:  # Пропускаем текущий чанк, он уже в результате
                    continue
                    
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(all_chunks):
                    neighbor = all_chunks[neighbor_idx]
                    if neighbor not in result:
                        result.add(neighbor)
                        chunks_added += 1
                        logger.debug(f"Добавлен соседний чанк с индексом {neighbor_idx}")
        except ValueError:
            logger.debug(f"Не найдено точное совпадение для чанка в all_chunks, пробуем поиск по содержимому")
            
            # Возможно, чанки не полностью совпадают из-за удаления строк или иных изменений
            # Попробуем найти наиболее похожий чанк
            best_match_idx = -1
            best_match_score = 0
            
            # Простой алгоритм поиска наиболее похожего чанка
            for i, candidate in enumerate(all_chunks):
                if len(candidate) > 20 and len(chunk) > 20:  # Проверяем только значимые чанки
                    # Подсчитываем количество общих начальных символов
                    min_len = min(len(candidate), len(chunk))
                    common_prefix = 0
                    for j in range(min_len):
                        if candidate[j] == chunk[j]:
                            common_prefix += 1
                        else:
                            break
                    
                    if common_prefix > best_match_score:
                        best_match_score = common_prefix
                        best_match_idx = i
            
            # Если найдено хорошее совпадение (как минимум 20 общих символов)
            if best_match_score >= 20:
                logger.debug(f"Найден похожий чанк с индексом {best_match_idx} (совпадение {best_match_score} символов)")
                idx = best_match_idx
                
                # Добавляем соседние чанки
                for offset in range(-window, window + 1):
                    if offset == 0:  # Пропускаем текущий чанк
                        continue
                        
                    neighbor_idx = idx + offset
                    if 0 <= neighbor_idx < len(all_chunks):
                        neighbor = all_chunks[neighbor_idx]
                        if neighbor not in result:
                            result.add(neighbor)
                            chunks_added += 1
                            logger.debug(f"Добавлен соседний чанк с индексом {neighbor_idx}")
    
    logger.info(f"Расширение чанков: добавлено {chunks_added} соседних фрагментов")
    return list(result)

async def process_file(file_path: Path, file_hashes: Dict[str, str]) -> Tuple[str, List[str]]:
    """Асинхронная обработка файла и создание/обновление коллекции"""
    logger.info(f"Обработка файла: {file_path.name}")
    file_hash = get_file_hash(file_path)
    collection_name = file_path.stem.replace(" ", "_").lower()

    # Проверка изменений файла
    if file_path.name in file_hashes and file_hashes[file_path.name] != file_hash:
        logger.info("Файл изменён — коллекция будет пересоздана.")
        try:
            client.delete_collection(collection_name)
        except Exception as e:
            logger.error(f"Ошибка при удалении коллекции {collection_name}: {e}")
    elif file_path.name in file_hashes:
        logger.info("Файл не изменён. Пропуск загрузки.")
        # Загружаем существующие чанки из кэша
        cache_path = file_path.with_suffix(".chunks.pkl")
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    chunks = pickle.load(f)
                return file_hash, chunks
            except Exception as e:
                logger.error(f"Ошибка при загрузке кэша чанков: {e}")
        
        # Если кэш не загрузился, продолжаем обычную обработку
    
    # Чтение и разделение текста на чанки
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {file_path}: {e}")
        return file_hash, []

    chunks = split_text(text)
    
    # Сохраняем чанки в кэш
    chunks_cache_path = file_path.with_suffix(".chunks.pkl")
    try:
        with open(chunks_cache_path, "wb") as f:
            pickle.dump(chunks, f)
    except Exception as e:
        logger.error(f"Ошибка при сохранении кэша чанков: {e}")
    
    # Обработка эмбеддингов
    cache_path = file_path.with_suffix(".emb.pkl")
    embeddings = []

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                embeddings = pickle.load(f)
            logger.info("Загружены эмбеддинги из кэша.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке кэша эмбеддингов: {e}")
            # Продолжаем создание новых эмбеддингов

    # Если эмбеддингов нет или их количество не соответствует чанкам, создаем новые
    if len(embeddings) != len(chunks):
        logger.info(f"Создаем эмбеддинги для {len(chunks)} чанков...")
        batch_size = config["batch_size"]
        
        embeddings = []
        for i in tqdm(range(0, len(chunks), batch_size), desc="Создание эмбеддингов"):
            batch = chunks[i:min(i+batch_size, len(chunks))]
            retry_delay = 1
            
            while True:
                try:
                    batch_embeddings = embed_texts(batch)
                    if not batch_embeddings or len(batch_embeddings) != len(batch):
                        raise ValueError(f"Получено {len(batch_embeddings) if batch_embeddings else 0} эмбеддингов для {len(batch)} чанков")
                    
                    embeddings.extend(batch_embeddings)
                    
                    # Сохраняем промежуточные результаты
                    try:
                        with open(cache_path, "wb") as f:
                            pickle.dump(embeddings, f)
                    except Exception as e:
                        logger.warning(f"Не удалось сохранить промежуточный кэш: {e}")
                    
                    retry_delay = 1
                    break
                except Exception as e:
                    logger.warning(f"Ошибка при создании эмбеддингов: {e}. Повтор через {retry_delay} секунд...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, config["retry_max_delay"])

        logger.info("Эмбеддинги сохранены в кэш.")

    # Сохранение в базу данных
    try:
        collection = client.get_or_create_collection(name=collection_name)
        existing_ids = set(collection.get(ids=None)["ids"])
        
        # Добавляем только новые документы
        new_chunks = []
        new_embeddings = []
        new_ids = []
        
        for i, chunk in enumerate(chunks):
            doc_id = f"{file_path.stem}_chunk_{i}"
            if doc_id in existing_ids:
                continue
                
            new_chunks.append(chunk)
            new_embeddings.append(embeddings[i])
            new_ids.append(doc_id)
        
        if new_chunks:
            # Разбиваем на батчи для добавления в базу
            batch_size = 100  # Оптимальный размер батча для ChromaDB
            for i in range(0, len(new_chunks), batch_size):
                end_idx = min(i + batch_size, len(new_chunks))
                collection.add(
                    documents=new_chunks[i:end_idx],
                    embeddings=new_embeddings[i:end_idx],
                    ids=new_ids[i:end_idx]
                )
            logger.info(f"Добавлено {len(new_chunks)} новых чанков в коллекцию {collection_name}")
        else:
            logger.info(f"Все чанки уже существуют в коллекции {collection_name}")
            
    except Exception as e:
        logger.error(f"Ошибка при работе с базой данных: {e}")
    
    return file_hash, chunks

async def process_query(query: str, target_file: str, all_chunks_map: Dict[str, List[str]]) -> None:
    """Обработка запроса пользователя"""
    try:
        # Получение эмбеддинга запроса
        query_embedding_result = embed_texts([query])
        if not query_embedding_result:
            logger.error("Не удалось получить эмбеддинг запроса. Проверьте подключение к API.")
            return

        query_embed = query_embedding_result[0]

        query_args = {
            "query_embeddings": [query_embed],
            "n_results": config["search_results"],
            "include": ["documents", "metadatas", "distances"]  # ids не поддерживается в параметре include
        }

        # Определение коллекций для поиска
        collections_to_query = []
        
        if target_file:
            collection_name = Path(target_file).stem.replace(" ", "_").lower()
            try:
                collections_to_query = [client.get_or_create_collection(name=collection_name)]
                logger.debug(f"Поиск по файлу {target_file}")
            except Exception as e:
                logger.error(f"Ошибка при получении коллекции {collection_name}: {e}")
                return
        else:
            # Поиск по всем коллекциям
            all_files = [f.name for f in Path(config["data_path"]).glob("*.txt")]
            logger.debug(f"Поиск по всем файлам: {all_files}")
            for f in all_files:
                collection_name = Path(f).stem.replace(" ", "_").lower()
                try:
                    collections_to_query.append(client.get_or_create_collection(name=collection_name))
                    logger.debug(f"Добавлена коллекция для поиска: {collection_name}")
                except Exception as e:
                    logger.error(f"Ошибка при получении коллекции {collection_name}: {e}")

        # Поиск релевантных чанков
        relevant_chunks = []
        chunk_file_mapping = {}  # Для отслеживания, из какого файла каждый чанк
        sources = []
        
        # Для каждой коллекции выполняем поиск
        for col in collections_to_query:
            try:
                results = col.query(**query_args)
                
                # Получаем информацию о файле для каждого чанка
                for i, chunk_text in enumerate(results["documents"][0]):
                    # Извлекаем имя файла из имени коллекции
                    source_file = col.name + ".txt"
                    
                    relevant_chunks.append(chunk_text)
                    chunk_file_mapping[chunk_text] = source_file
                    sources.append(source_file)
            except Exception as e:
                logger.error(f"Ошибка при запросе к коллекции {col.name}: {e}")

        logger.info(f"Получено релевантных чанков: {len(relevant_chunks)}")

        # Группируем чанки по файлам для корректного добавления соседей
        file_chunks = {}
        expanded_chunks_by_file = {}  # Группируем расширенные чанки по файлам
        
        for chunk in relevant_chunks:
            file_name = chunk_file_mapping.get(chunk)
            if file_name not in file_chunks:
                file_chunks[file_name] = []
            file_chunks[file_name].append(chunk)
        
        # Сохраняем первоначальный порядок для релевантных чанков
        original_relevant_chunks = relevant_chunks.copy()
        
        # Расширяем чанки с учетом принадлежности к файлам и сохраняем порядок
        for file_name, file_relevant_chunks in file_chunks.items():
            # Получаем все чанки для текущего файла
            all_chunks_for_file = all_chunks_map.get(file_name, [])
            
            if all_chunks_for_file:
                # Для каждого релевантного чанка определяем его соседей
                expanded_file_chunks_ordered = {}
                
                for chunk in file_relevant_chunks:
                    try:
                        # Находим индекс чанка в исходном списке
                        chunk_idx = all_chunks_for_file.index(chunk)
                        
                        # Создаем упорядоченный список с чанком и его соседями
                        ordered_neighbors = []
                        window = config["neighbor_window"]
                        
                        # Добавляем соседей выше
                        for i in range(chunk_idx - window, chunk_idx):
                            if i >= 0:
                                ordered_neighbors.append(all_chunks_for_file[i])
                        
                        # Добавляем сам чанк
                        ordered_neighbors.append(chunk)
                        
                        # Добавляем соседей ниже
                        for i in range(chunk_idx + 1, chunk_idx + window + 1):
                            if i < len(all_chunks_for_file):
                                ordered_neighbors.append(all_chunks_for_file[i])
                        
                        # Сохраняем упорядоченные чанки
                        expanded_file_chunks_ordered[chunk] = ordered_neighbors
                    except ValueError:
                        # Если чанк не найден, пробуем найти наиболее похожий
                        best_match_idx = -1
                        best_match_score = 0
                        
                        for i, candidate in enumerate(all_chunks_for_file):
                            if len(candidate) > 20 and len(chunk) > 20:
                                # Подсчитываем количество общих начальных символов
                                min_len = min(len(candidate), len(chunk))
                                common_prefix = 0
                                for j in range(min_len):
                                    if candidate[j] == chunk[j]:
                                        common_prefix += 1
                                    else:
                                        break
                                
                                if common_prefix > best_match_score:
                                    best_match_score = common_prefix
                                    best_match_idx = i
                        
                        # Если найдено хорошее совпадение
                        if best_match_score >= 20 and best_match_idx >= 0:
                            # Создаем упорядоченный список с чанком и его соседями
                            ordered_neighbors = []
                            window = config["neighbor_window"]
                            
                            # Добавляем соседей выше
                            for i in range(best_match_idx - window, best_match_idx):
                                if i >= 0:
                                    ordered_neighbors.append(all_chunks_for_file[i])
                            
                            # Добавляем сам чанк
                            ordered_neighbors.append(chunk)
                            
                            # Добавляем соседей ниже
                            for i in range(best_match_idx + 1, best_match_idx + window + 1):
                                if i < len(all_chunks_for_file):
                                    ordered_neighbors.append(all_chunks_for_file[i])
                            
                            # Сохраняем упорядоченные чанки
                            expanded_file_chunks_ordered[chunk] = ordered_neighbors
                        else:
                            # Если не смогли найти подобный чанк, просто оставляем его самого
                            expanded_file_chunks_ordered[chunk] = [chunk]
                
                expanded_chunks_by_file[file_name] = expanded_file_chunks_ordered
        
        # Выводим информацию о количестве добавленных чанков
        expanded_chunks_count = sum(len(neighbors) for file_chunks in expanded_chunks_by_file.values() 
                                 for neighbors in file_chunks.values())
        logger.info(f"После расширения: {expanded_chunks_count} чанков (добавлено {expanded_chunks_count - len(relevant_chunks)})")

        # Вывод результатов
        print("\nРелевантные фрагменты:")
        if not expanded_chunks_by_file:
            print("Нет релевантных фрагментов.")
        else:
            # Выводим фрагменты в нужном порядке
            fragment_index = 1
            for original_chunk in original_relevant_chunks:
                file_name = chunk_file_mapping.get(original_chunk, "неизвестный источник")
                
                # Получаем упорядоченные соседи для текущего чанка
                if file_name in expanded_chunks_by_file and original_chunk in expanded_chunks_by_file[file_name]:
                    ordered_fragments = expanded_chunks_by_file[file_name][original_chunk]
                    
                    # Выводим фрагменты в порядке: верхние соседи, сам чанк, нижние соседи
                    for idx, fragment in enumerate(ordered_fragments):
                        is_original = (fragment == original_chunk)
                        marker = "КЛЮЧЕВОЙ ФРАГМЕНТ" if is_original else "соседний фрагмент"
                        print(f"\n--- Фрагмент {fragment_index} ({marker}, источник: {file_name}) ---\n{fragment}")
                        fragment_index += 1
            
            # Уникальные источники
            unique_sources = list(set(sources))
            if unique_sources:
                print("\nИсточники:")
                for source in unique_sources:
                    print(f"- {source}")

        # Собираем все чанки для запроса к GPT
        all_expanded_chunks = []
        for file_chunks in expanded_chunks_by_file.values():
            for ordered_neighbors in file_chunks.values():
                all_expanded_chunks.extend(ordered_neighbors)
        
        # Удаляем дубликаты, сохраняя порядок
        seen = set()
        unique_expanded_chunks = []
        for chunk in all_expanded_chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique_expanded_chunks.append(chunk)

        # Запрос к GPT
        if unique_expanded_chunks:
            print("\nГенерация ответа...")
            
            # МОДИФИЦИРОВАННЫЙ КОД: Используем новый индикатор прогресса на основе потоков
            import threading
            start_time = time.time()
            
            # Определяем, поддерживает ли терминал интерактивный вывод
            is_interactive = sys.stdout.isatty()
            
            # Функция для простого индикатора (работает в любой среде)
            def simple_progress():
                count = 0
                while not progress_done.is_set():
                    elapsed = time.time() - start_time
                    if is_interactive:
                        # Для интерактивного терминала используем символы и замену строки
                        symbols = ["-", "\\", "|", "/"]
                        mins, secs = divmod(int(elapsed), 60)
                        timestr = f"{mins:02d}:{secs:02d}" if mins > 0 else f"{secs}s"
                        sys.stdout.write(f"\rОжидание ответа от GPT {symbols[count % 4]} {timestr}")
                        sys.stdout.flush()
                    else:
                        # Для неинтерактивного терминала просто печатаем точки
                        if count % 5 == 0:  # Каждые ~5 секунд
                            mins, secs = divmod(int(elapsed), 60)
                            timestr = f"{mins:02d}:{secs:02d}" if mins > 0 else f"{secs}s"
                            print(f"Ожидание ответа от GPT... {timestr}")
                    
                    count += 1
                    time.sleep(1.0)
                
                # Очищаем строку в интерактивном режиме
                if is_interactive:
                    sys.stdout.write("\r" + " " * 50 + "\r")
                    sys.stdout.flush()
            
            # Запускаем индикатор прогресса
            progress_done = threading.Event()
            progress_thread = threading.Thread(target=simple_progress)
            progress_thread.daemon = True  # Поток завершится, если основной поток завершится
            progress_thread.start()
            
            try:
                # Выполняем запрос к GPT
                answer = ask_gpt(unique_expanded_chunks, query)
                
                # Останавливаем индикатор прогресса
                progress_done.set()
                progress_thread.join(timeout=1.0)  # Ждем завершения потока не более 1 секунды
                
                # Выводим ответ с четким началом и концом
                print("\n┌────────────────────────────────────────────────────")
                print("│ ОТВЕТ:")
                print("├────────────────────────────────────────────────────")
                
                # Форматируем вывод ответа по строкам для лучшей читаемости
                for line in answer.split('\n'):
                    print(f"│ {line}")
                
                print("└────────────────────────────────────────────────────")
                
                # Выводим информацию о времени выполнения
                elapsed = time.time() - start_time
                print(f"\nВремя выполнения запроса: {elapsed:.2f} секунд")
                
            except Exception as e:
                # В случае ошибки останавливаем индикатор прогресса
                progress_done.set()
                try:
                    progress_thread.join(timeout=1.0)
                except:
                    pass
                
                logger.error(f"Ошибка при запросе к GPT: {e}")
                print("\nНе удалось получить ответ от GPT. Проверьте соединение и настройки API.")
    
    except Exception as e:
        logger.error(f"Общая ошибка при обработке запроса: {e}")
        print(f"Произошла ошибка при обработке запроса: {e}")

async def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Система для семантического поиска в текстовых файлах")
    parser.add_argument("-f", "--file", help="Имя файла для поиска")
    parser.add_argument("-q", "--query", help="Поисковый запрос")
    parser.add_argument("--reindex", action="store_true", help="Принудительное переиндексирование файлов")
    args = parser.parse_args()

    # Проверка существования директорий
    data_path = Path(config["data_path"])
    db_path = Path(config["db_path"])
    
    if not data_path.exists():
        data_path.mkdir(parents=True)
        logger.info(f"Создана директория {data_path}")
    
    if not db_path.exists():
        db_path.mkdir(parents=True)
        logger.info(f"Создана директория {db_path}")

    # Поиск всех текстовых файлов
    text_files = list(data_path.glob("*.txt"))
    if not text_files:
        logger.error(f"Не найдены .txt файлы в директории {data_path}")
        print(f"Не найдены текстовые файлы в директории {data_path}.")
        print("Пожалуйста, добавьте как минимум один .txt файл в эту директорию и перезапустите программу.")
        return

    # Загрузка хешей файлов
    hash_path = data_path / "file_hashes.pkl"
    file_hashes = {}
    
    if hash_path.exists() and not args.reindex:
        try:
            with open(hash_path, "rb") as f:
                file_hashes = pickle.load(f)
        except Exception as e:
            logger.error(f"Ошибка при загрузке хешей файлов: {e}")

    # Обработка файлов
    all_chunks_map = {}
    tasks = []
    
    for file_path in text_files:
        if args.reindex and file_path.name in file_hashes:
            logger.info(f"Принудительное переиндексирование файла {file_path.name}")
            del file_hashes[file_path.name]
            
        task = process_file(file_path, file_hashes)
        tasks.append(task)
    
    # Запускаем задачи параллельно
    results = await asyncio.gather(*tasks)
    
    # Обновляем хеши и собираем чанки
    for file_path, (file_hash, chunks) in zip(text_files, results):
        file_hashes[file_path.name] = file_hash
        all_chunks_map[file_path.name] = chunks
    
    # Сохраняем обновленные хеши
    try:
        with open(hash_path, "wb") as f:
            pickle.dump(file_hashes, f)
    except Exception as e:
        logger.error(f"Ошибка при сохранении хешей файлов: {e}")

    # Выводим доступные файлы
    print("\nДоступные файлы:")
    available_files = [f.name for f in text_files]
    for name in available_files:
        print("-", name)

    # Определяем целевой файл
    target_file = args.file
    
    if args.file and args.file not in available_files:
        logger.warning(f"Файл {args.file} не найден. Будет выполнен поиск по всем файлам.")
        print(f"Файл {args.file} не найден. Будет выполнен поиск по всем файлам.")
        target_file = ""
    
    # Если указан запрос в аргументах, выполняем его и выходим
    if args.query:
        await process_query(args.query, target_file, all_chunks_map)
        return

    # Интерактивный режим
    if not target_file:
        target_input = input("\nВведите имя файла, по которому искать (или оставьте пустым для всех): ").strip()
        if target_input and target_input in available_files:
            target_file = target_input
        else:
            if target_input:
                print("Файл не найден. Будет выполнен поиск по всем файлам.")
            target_file = ""

    # Цикл запросов
    while True:
        query = input("\nВведите запрос (или 'exit' для выхода): ")
        if query.strip().lower() == "exit":
            print("Выход из программы.")
            break

        await process_query(query, target_file, all_chunks_map)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем.")
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}")
        print(f"Произошла критическая ошибка: {e}")
        print("Проверьте файл логов app.log для получения дополнительной информации.")