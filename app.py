import streamlit as st
from rag_data import process_documents
from rag_query import query_rag
import os

st.title("RAG-система для SEO-отдела v1.0")

# Загрузка файлов
uploaded_files = st.file_uploader("Загрузите документы (.txt, .pdf, .docx)", accept_multiple_files=True, type=["txt", "pdf", "docx"])
if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        # Сохраняем загруженный файл во временную папку
        temp_dir = "./temp_docs"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    if st.button("Обработать документы"):
        with st.spinner("Обработка документов..."):
            vector_store = process_documents(file_paths)
            if vector_store:
                st.success("Документы успешно обработаны и сохранены")
            else:
                st.error("Ошибка при обработке документов.")

# Поле для ввода вопроса
query = st.text_input("Тут запрос ввести", placeholder="Введите ваш вопрос о документах...")
if st.button("Отправить запрос"):
    with st.spinner("Выполнение запроса..."):
        result = query_rag(query)
        if result:
            st.write("**Ответ:**")
            st.write(result["result"])
            st.write("**Источники:**")
            unique_sources = {(doc.page_content[:100], doc.metadata.get('source')) for doc in result["source_documents"]}
            for content, source in unique_sources:
                st.write(f"- {content}... ({source})")