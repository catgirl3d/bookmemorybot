import json
import logging
import os
from typing import List, Dict
from openai import OpenAI

# Настройка логирования
logger = logging.getLogger(__name__)

# Путь к конфигурационному файлу
CONFIG_PATH = "config.json"

# Дефолтные настройки для GPT
DEFAULT_GPT_CONFIG = {
    "gpt_model": "gpt-4.1-nano",
    "gpt_temperature": 0.1,
    "gpt_max_tokens": 3500
}

def load_or_create_config() -> Dict:
    """
    Загружает существующую конфигурацию или создает новую с дефолтными значениями.
    При необходимости дополняет существующую конфигурацию недостающими параметрами.
    
    Returns:
        Dict: Загруженная или созданная конфигурация
    """
    config = {}
    config_modified = False
    
    # Пытаемся загрузить существующую конфигурацию
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info("Конфигурация успешно загружена из файла.")
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка чтения файла конфигурации: {e}")
            config = {}
    
    # Проверяем, что все нужные настройки GPT присутствуют
    for key, value in DEFAULT_GPT_CONFIG.items():
        if key not in config:
            config[key] = value
            config_modified = True
            logger.info(f"Добавлен отсутствующий параметр в конфигурации: {key}={value}")
    
    # Сохраняем обновленную конфигурацию, если были внесены изменения
    if config_modified:
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            logger.info("Обновленная конфигурация сохранена в файл.")
        except Exception as e:
            logger.error(f"Не удалось сохранить конфигурацию: {e}")
    
    return config

# Загружаем конфигурацию
config = load_or_create_config()

# Настройка API ключа
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def split_text(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Разделяет текст на чанки подходящего размера, стараясь сохранить
    целостность абзацев и предложений.
    
    Args:
        text: Исходный текст для разделения
        max_chunk_size: Максимальный размер чанка в символах
        
    Returns:
        Список чанков текста
    """
    # Разделяем по абзацам
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # Если абзац слишком большой, разделяем его по предложениям
        if len(paragraph) > max_chunk_size:
            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                if current_size + len(sentence) + 1 <= max_chunk_size:
                    current_chunk.append(sentence)
                    current_size += len(sentence) + 1  # +1 for space
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_size = len(sentence)
        else:
            # Проверяем, поместится ли абзац в текущий чанк
            if current_size + len(paragraph) + 2 <= max_chunk_size:  # +2 for newline
                current_chunk.append(paragraph)
                current_size += len(paragraph) + 2
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_size = len(paragraph)
    
    # Добавляем последний чанк
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks

def split_into_sentences(text: str) -> List[str]:
    """
    Разделяет текст на предложения с учетом знаков препинания.
    """
    # Простое разделение по знакам препинания с учетом распространенных сокращений
    import re
    
    # Заменяем сокращения временными маркерами
    abbreviations = [
        r"Mr\.", r"Mrs\.", r"Dr\.", r"Prof\.", r"etc\.", r"i\.e\.", r"e\.g\.",
        r"\d+\.\d+", r"т\.\s*д\.", r"т\.\s*п\.", r"т\.\s*е\."
    ]
    
    temp_text = text
    for i, abbr_pattern in enumerate(abbreviations):
        temp_text = re.sub(abbr_pattern, f"ABBR{i}", temp_text)
    
    # Разделение по знакам препинания
    sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
    temp_sentences = re.split(sentence_endings, temp_text)
    
    # Восстанавливаем сокращения
    sentences = []
    for sent in temp_sentences:
        for i, abbr_pattern in enumerate(abbreviations):
            sent = sent.replace(f"ABBR{i}", abbr_pattern.replace("\\", ""))
        sentences.append(sent.strip())
    
    return [s for s in sentences if s]

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Создает эмбеддинги для списка текстов с использованием API OpenAI.
    
    Args:
        texts: Список текстов для эмбеддинга
        
    Returns:
        Список векторов эмбеддингов
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        return [item.embedding for item in response.data]
    except Exception as e:
        logging.error(f"Ошибка при создании эмбеддингов: {e}")
        return []

def ask_gpt(context_chunks: List[str], query: str) -> str:
    """
    Отправляет запрос к GPT для получения ответа на основе контекстных чанков.
    
    Args:
        context_chunks: Список текстовых фрагментов для контекста
        query: Вопрос пользователя
        
    Returns:
        Ответ от GPT
    """
    if not OPENAI_API_KEY:
        return "Ошибка: API ключ OpenAI не найден. Убедитесь, что переменная окружения OPENAI_API_KEY установлена."
    
    combined_context = "\n\n---\n\n".join(context_chunks)
    
    try:
        system_message = """
        Ты - ассистент для вопросов и ответов на основе предоставленных документов. отвечай развернуто и подробно
        """
        
        user_message = f"""
        Контекст:
        
        {combined_context}
        
        Вопрос: {query}
        """
        
        response = client.chat.completions.create(
            model=config["gpt_model"],
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=config["gpt_temperature"],
            max_tokens=config["gpt_max_tokens"]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Ошибка при получении ответа от GPT: {e}")
        return f"Ошибка при получении ответа: {e}"
