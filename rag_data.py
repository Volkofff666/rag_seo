import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from utils import get_embeddings
from PyPDF2 import PdfReader
from docx import Document

def load_document(file_path):
    """Загружает содержимое файла (.txt, .pdf, .docx)."""
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            return " ".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
        else:
            raise ValueError("Неподдерживаемый формат файла. Используйте .txt, .pdf или .docx.")
    except Exception as e:
        raise Exception(f"Ошибка при загрузке файла {file_path}: {e}")

def process_documents(file_paths, collection_name="user-docs", persist_directory="./chroma_db"):
    """Обрабатывает документы и сохраняет их в Chroma."""
    try:
        # Загрузка содержимого всех файлов
        documents = []
        for file_path in file_paths:
            content = load_document(file_path)
            documents.append({"page_content": content, "metadata": {"source": file_path}})

        # Разбиение на куски
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        all_splits = text_splitter.create_documents(
            [doc["page_content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )
        print(f"Всего кусков: {len(all_splits)}")

        # Создание эмбеддингов
        embeddings = get_embeddings()

        # Сохранение в Chroma
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        ids = vector_store.add_documents(all_splits)
        print(f"Сохранено {len(ids)} документов в Chroma.")
        return vector_store
    except Exception as e:
        print(f"Ошибка при обработке документов: {e}")
        return None

if __name__ == "__main__":
    # Пример использования: укажите пути к вашим файлам
    file_paths = ["example.txt"]  # Замените на реальные пути к файлам
    process_documents(file_paths)