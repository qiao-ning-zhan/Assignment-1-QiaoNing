import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from docx import Document

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def chunk_document(text: str, chunk_size: int = 1000) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def setup_vector_db():
    chroma_client = chromadb.Client()
    embedding_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="text-embedding-ada-002"
    )
    try:
        return chroma_client.create_collection(name="legal_docs", embedding_function=embedding_func)
    except chromadb.db.base.UniqueConstraintError:
        return chroma_client.get_collection(name="legal_docs", embedding_function=embedding_func)

def preprocess_and_store_document(collection, document: str):
    chunks = chunk_document(document)
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return len(chunks)

def query_document(collection, query: str, k: int = 2) -> dict:
    results = collection.query(query_texts=[query], n_results=k)
    return {
        'documents': results['documents'][0],
        'distances': results['distances'][0],
        'ids': results['ids'][0]
    }

def generate_response(context: str, question: str) -> str:
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful legal assistant. Provide clear and concise answers."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def read_docx(file):
    doc = Document(file)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

def main():
    st.set_page_config(page_title="Legal Assistant", page_icon="⚖️", layout="wide")
    st.title("🤖 Your AI Legal Assistant")

    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = setup_vector_db()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload a legal document", type=["txt", "docx"])
        
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                document = uploaded_file.getvalue().decode("utf-8")
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                document = read_docx(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a .txt or .docx file.")
                return

            with st.spinner("Processing document..."):
                num_chunks = preprocess_and_store_document(st.session_state.vector_db, document)
            st.success(f"✅ Document processed successfully! ({num_chunks} sections analyzed)")

    with col1:
        st.subheader("💬 Chat with Your Legal Assistant")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about the legal document...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.spinner("Searching for relevant information..."):
                results = query_document(st.session_state.vector_db, question)

            with st.chat_message("assistant"):
                context = " ".join(results['documents'])
                response = generate_response(context, question)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            with st.expander("📚 View relevant document sections"):
                for i, (doc, distance) in enumerate(zip(results['documents'], results['distances']), 1):
                    st.markdown(f"**Section {i}** (Relevance: {(1-distance)*100:.1f}%)")
                    st.text(doc[:200] + "..." if len(doc) > 200 else doc)

if __name__ == "__main__":
    main()