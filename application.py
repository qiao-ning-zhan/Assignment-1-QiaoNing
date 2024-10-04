import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from docx import Document

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)  # 粗略估计：平均每个单词约1.3个token

def chunk_document(text: str, chunk_size: int = 1000) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def setup_vector_db():
    persist_directory = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    embedding_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="text-embedding-ada-002"
    )
    try:
        return chroma_client.get_or_create_collection(name="legal_docs", embedding_function=embedding_func)
    except Exception as e:
        st.error(f"Error setting up vector database: {str(e)}")
        return None

def preprocess_and_store_document(collection, document: str, doc_id: str):
    chunks = chunk_document(document)
    collection.add(
        documents=chunks,
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"doc_id": doc_id} for _ in chunks]
    )
    return len(chunks)

def query_document(collection, query: str, k: int = 2) -> dict:
    max_tokens = 6000  # 为其他部分留出一些空间
    results = collection.query(query_texts=[query], n_results=k)
    if not results['documents'][0]:
        return None
    
    documents = results['documents'][0]
    distances = results['distances'][0]
    ids = results['ids'][0]
    metadatas = results['metadatas'][0]
    
    # 限制返回的文档总长度
    total_tokens = 0
    limited_docs = []
    limited_distances = []
    limited_ids = []
    limited_metadatas = []
    
    for doc, dist, id, meta in zip(documents, distances, ids, metadatas):
        doc_tokens = estimate_tokens(doc)
        if total_tokens + doc_tokens > max_tokens:
            break
        total_tokens += doc_tokens
        limited_docs.append(doc)
        limited_distances.append(dist)
        limited_ids.append(id)
        limited_metadatas.append(meta)
    
    return {
        'documents': limited_docs,
        'distances': limited_distances,
        'ids': limited_ids,
        'metadatas': limited_metadatas
    }

def generate_response(context: str, question: str) -> str:
    max_tokens = 8192
    system_prompt = "You are a helpful legal assistant. Provide clear and concise answers."
    
    # 估算当前token数
    estimated_tokens = estimate_tokens(system_prompt) + estimate_tokens(question) + estimate_tokens(context) + 100  # 额外100用于其他开销
    
    # 如果估算的token数超过限制，裁剪上下文
    if estimated_tokens > max_tokens:
        context_tokens = max_tokens - estimate_tokens(system_prompt) - estimate_tokens(question) - 200  # 为安全起见多减200
        context_words = int(context_tokens / 1.3)  # 将token数转换回大致的单词数
        context = ' '.join(context.split()[:context_words])
    
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
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

    if 'vector_db' not in st.session_state or st.session_state.vector_db is None:
        st.session_state.vector_db = setup_vector_db()
        if st.session_state.vector_db is None:
            st.error("Failed to initialize vector database. Please refresh the page and try again.")
            return
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("📄 Document Upload")
        uploaded_files = st.file_uploader("Upload legal documents", type=["txt", "docx"], accept_multiple_files=True)
        st.caption("Limit 200MB per file • TXT, DOCX")
      
        
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                if file.type == "text/plain":
                    document = file.getvalue().decode("utf-8")
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    document = read_docx(file)
                else:
                    st.error(f"Unsupported file type for {file.name}. Please upload .txt or .docx files.")
                    continue

                with st.spinner(f"Processing {file.name}..."):
                    try:
                        num_chunks = preprocess_and_store_document(st.session_state.vector_db, document, file.name)
                        st.success(f"✅ {file.name} processed successfully! ({num_chunks} sections analyzed)")
                        st.session_state.uploaded_files[file.name] = num_chunks
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")

        if st.session_state.uploaded_files:
            st.write("Uploaded Documents:")
            for filename, chunks in st.session_state.uploaded_files.items():
                st.write(f"- {filename} ({chunks} sections)")

    with col1:
        st.subheader("💬 Chat with Your Legal Assistant")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about the legal documents...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.spinner("Searching for relevant information..."):
                try:
                    results = query_document(st.session_state.vector_db, question)
                except Exception as e:
                    st.error(f"Error querying documents: {str(e)}")
                    results = None

            if results:
                with st.chat_message("assistant"):
                    context = " ".join(results['documents'])
                    response = generate_response(context, question)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                with st.expander("📚 View relevant document sections"):
                    for i, (doc, distance, metadata) in enumerate(zip(results['documents'], results['distances'], results['metadatas']), 1):
                        if metadata and 'doc_id' in metadata:
                            st.markdown(f"**Section {i}** (Document: {metadata['doc_id']}, Relevance: {(1-distance)*100:.1f}%)")
                        else:
                            st.markdown(f"**Section {i}** (Relevance: {(1-distance)*100:.1f}%)")
                        st.text(doc[:200] + "..." if len(doc) > 200 else doc)
            else:
                st.error("No relevant information found. Please try a different question or upload more documents.")

if __name__ == "__main__":
    main()