import streamlit as st
import pandas as pd
from typing import List, Dict, Callable
import io
from docx import Document

class ContentExtractor:
    @staticmethod
    def extract_from_txt(file_content: bytes) -> str:
        return file_content.decode("utf-8")

    @staticmethod
    def extract_from_docx(file_content: bytes) -> str:
        doc = Document(io.BytesIO(file_content))
        return ' '.join([para.text for para in doc.paragraphs])

class DocumentAnalyzer:
    def __init__(self):
        self.extractors: Dict[str, Callable] = {
            '.txt': ContentExtractor.extract_from_txt,
            '.docx': ContentExtractor.extract_from_docx
        }
        self.content: str = ""
        self.chunks: List[str] = []

    def load_document(self, file) -> None:
        file_extension = f".{file.name.split('.')[-1].lower()}"
        if file_extension not in self.extractors:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        self.content = self.extractors[file_extension](file.read())

    def split_content(self, chunk_size: int) -> None:
        words = self.content.split()
        self.chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def render_ui():
    st.title("Document Analyzer")

    analyzer = DocumentAnalyzer()

    uploaded_file = st.file_uploader("Choose a document", type=['txt', 'docx'])
    
    if uploaded_file:
        try:
            analyzer.load_document(uploaded_file)
            st.success(f"'{uploaded_file.name}' loaded successfully.")
        except ValueError as e:
            st.error(str(e))

    chunk_size = st.slider("Select chunk size", 50, 500, 100, 50)

    if analyzer.content and st.button("Analyze Document"):
        analyzer.split_content(chunk_size)
        st.write(f"Text split into {len(analyzer.chunks)} chunks.")
        
        for idx, chunk in enumerate(analyzer.chunks, 1):
            st.text_area(f"Chunk {idx}", chunk, height=100)

if __name__ == "__main__":
    render_ui()