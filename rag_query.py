from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from utils import get_embeddings, get_llm

def query_rag(question, collection_name="user-docs", persist_directory="./chroma_db"):
    """Выполняет RAG-запрос к базе Chroma."""
    try:
        # Подключение к Chroma
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=persist_directory
        )

        # Настройка промпта
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Ты полезный ассистент. На основе предоставленного контекста дай точное определение термина или объясни его. Если информации недостаточно, укажи это и предложи общий ответ."),
            ("human", "Контекст: {context}\n\nВопрос: {question}"),
        ])

        # Создание RAG-цепочки
        qa_chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        # Выполнение запроса
        result = qa_chain.invoke({"query": question})
        print("\nОтвет RAG-системы:")
        print(result["result"])
        print("\nИсточники:")
        unique_sources = {(doc.page_content[:100], doc.metadata.get('source')) for doc in result["source_documents"]}
        for content, source in unique_sources:
            print(f"- {content}... (Источник: {source})")
        return result
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return None

if __name__ == "__main__":
    query_rag("Что такое prompt engineering?")