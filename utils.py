import tiktoken
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import os
import logging
import time
import re

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_text(text, max_tokens=500, split_by="hybrid"):
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []

    if split_by == "hybrid":
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            para_tokens = enc.encode(para)
            if len(para_tokens) <= max_tokens:
                chunks.append(para)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', para.strip())
                sentences = [s.strip() for s in sentences if s.strip()]
                current_chunk = []
                current_tokens = []
                for sent in sentences:
                    sent_tokens = enc.encode(sent)
                    if len(current_tokens) + len(sent_tokens) > max_tokens and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_tokens = []
                    current_chunk.append(sent)
                    current_tokens.extend(sent_tokens)
                    if len(current_tokens) >= max_tokens:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_tokens = []
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
    else:
        # Существующая логика для sentence или paragraph
        pass

    return chunks

def embed_texts(texts):
    embeddings = []
    try:
        logging.info(f"Создание эмбеддингов для {len(texts)} фрагментов")
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        embeddings = [r.embedding for r in response.data]
        logging.info("Эмбеддинги успешно созданы")
    except OpenAIError as e:
        logging.error(f"Ошибка при вызове OpenAI API: {e}")
        time.sleep(2)
    except Exception as e:
        logging.error(f"Непредвиденная ошибка: {e}")
    return embeddings

def ask_gpt(relevant_chunks, user_question):
    from openai import OpenAI

    context = "\n\n".join(relevant_chunks)
    messages = [
        {"role": "system", "content": "Ты — ассистент, помогающий анализировать книги и тексты. Отвечай развернуто, содержательно, используя только предоставленные фрагменты. Ты также должен предложить своё видение ситуации на основании этих фрагментов"},
        {"role": "user", "content": f"Вот фрагменты текста:\n\n{context}\n\nВопрос: {user_question}"}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        logging.error(f"Ошибка при запросе к chat API: {e}")
        return "Произошла ошибка при попытке получить ответ."
    except Exception as e:
        logging.error(f"Непредвиденная ошибка: {e}")
        return "Непредвиденная ошибка во время ответа."
