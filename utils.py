import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Загрузка переменных окружения
load_dotenv()

# Настройка API-ключа
IO_API_KEY = os.getenv("IOINTELLIGENCE_API_KEY")
if not IO_API_KEY:
    raise ValueError("IOINTELLIGENCE_API_KEY не установлен. Укажите его в .env или переменной окружения.")
BASE_URL = "https://api.intelligence.io.solutions/api/v1/"

def get_embeddings():
    """Создаёт эмбеддинги для IO Intelligence API."""
    try:
        embeddings = OpenAIEmbeddings(
            model="BAAI/bge-multilingual-gemma2",
            api_key=IO_API_KEY,
            base_url=BASE_URL
        )
        return embeddings
    except Exception as e:
        raise Exception(f"Ошибка при создании эмбеддингов: {e}")

def get_llm():
    """Создаёт LLM для IO Intelligence API."""
    try:
        llm = ChatOpenAI(
            model="mistralai/Mistral-Large-Instruct-2411",
            api_key=IO_API_KEY,
            base_url=BASE_URL,
            temperature=0.7,
            max_tokens=500
        )
        return llm
    except Exception as e:
        raise Exception(f"Ошибка при создании LLM: {e}")