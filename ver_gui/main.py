import pickle
from pathlib import Path
import chromadb
from utils import split_text, embed_texts, ask_gpt
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class MemoryBotGUI:
    def __init__(self, root, ask_fn, book_options):
        self.root = root
        self.ask_fn = ask_fn
        self.root.title("Book Memory Bot")

        ttk.Label(root, text="Выберите книгу:").pack(anchor="w")
        self.book_var = tk.StringVar()
        self.book_menu = ttk.Combobox(root, textvariable=self.book_var, values=book_options, state="readonly")
        self.book_menu.pack(fill="x")

        ttk.Label(root, text="Введите вопрос:").pack(anchor="w", pady=(10,0))
        self.query_entry = ttk.Entry(root)
        self.query_entry.pack(fill="x")

        ttk.Button(root, text="Задать вопрос", command=self.ask_question).pack(pady=10)

        ttk.Label(root, text="Релевантные фрагменты:").pack(anchor="w")
        self.fragments_text = scrolledtext.ScrolledText(root, height=10, wrap=tk.WORD, undo=True)
        self.fragments_text.config(exportselection=False)
        self.fragments_text.pack(fill="both", expand=True)

        ttk.Label(root, text="Ответ GPT:").pack(anchor="w")
        self.answer_text = scrolledtext.ScrolledText(root, height=10, wrap=tk.WORD, undo=True)
        self.answer_text.config(exportselection=False)
        self.answer_text.pack(fill="both", expand=True)

        self.fragments_text.bind("<Control-a>", self.select_all)
        self.fragments_text.bind("<Control-A>", self.select_all)
        self.answer_text.bind("<Control-a>", self.select_all)
        self.answer_text.bind("<Control-A>", self.select_all)

    def select_all(self, event):
        widget = event.widget
        widget.tag_add("sel", "1.0", "end")
        return "break"

    def ask_question(self):
        book = self.book_var.get()
        query = self.query_entry.get().strip()

        if not book or not query:
            messagebox.showwarning("Внимание", "Пожалуйста, выберите книгу и введите вопрос.")
            return

        fragments, answer = self.ask_fn(book, query)

        self.fragments_text.delete("1.0", tk.END)
        self.fragments_text.insert(tk.END, "\n\n".join(fragments))

        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, answer)

client = chromadb.PersistentClient(path="db/")
data_path = Path("data/")
text_files = list(data_path.glob("*.txt"))
assert text_files, "Положи хотя бы один .txt-файл в папку data/"

collections = {}
chunks_map = {}

for file_path in text_files:
    collection_name = file_path.stem.replace(" ", "_").lower()
    collection = client.get_or_create_collection(name=collection_name)
    collections[file_path.name] = collection

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = split_text(text)
    chunks_map[file_path.name] = chunks

    # Генерация эмбеддингов с кэшированием
    cache_path = file_path.with_suffix(".pkl")

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
        print(f"Загружены эмбеддинги из кэша для {file_path.name}")
    else:
        from tqdm import tqdm

        embeddings = []
        for i in tqdm(range(0, len(chunks), 10), desc=f"Создание эмбеддингов для {file_path.name}"):
            batch = chunks[i:i+10]
            batch_embeddings = embed_texts(batch)

            if len(batch_embeddings) != len(batch):
                print(f"⚠️  Пропущено {len(batch) - len(batch_embeddings)} фрагментов в batch {i} из-за ошибки эмбеддинга")
                continue

            embeddings.extend(batch_embeddings)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Созданы и сохранены эмбеддинги для {file_path.name}")

    if len(chunks) != len(embeddings):
        print(f"Предупреждение: для {file_path.name} количество эмбеддингов не совпадает с количеством фрагментов!")
    else:
        for i in range(len(chunks)):
            try:
                collection.add(
                documents=[chunks[i]],
                embeddings=[embeddings[i]],
                ids=[f"{file_path.stem}_chunk_{i}"]
            )
            except Exception as e:
                print(f"Ошибка при добавлении chunk {i} из файла {file_path.name}: {e}")

def expand_chunks_by_neighbors(chunks, all_chunks, window=1):
    result = set(chunks)
    for doc in chunks:
        try:
            idx = all_chunks.index(doc)
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor = all_chunks[idx + offset]
                result.add(neighbor)
        except (ValueError, IndexError):
            continue
    return list(result)

def ask_fn(book_name, query):
    collection = collections[book_name]
    all_chunks = chunks_map[book_name]

    query_embed = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embed], n_results=1, include=["documents"])

    relevant_chunks = results["documents"][0]
    expanded_chunks = expand_chunks_by_neighbors(relevant_chunks, all_chunks)

    answer = ask_gpt(expanded_chunks, query)
    return expanded_chunks, answer

if __name__ == "__main__":
    root = tk.Tk()
    gui = MemoryBotGUI(root, ask_fn, list(collections.keys()))
    root.mainloop()
