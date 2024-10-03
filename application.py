import streamlit as st
from typing import List, Dict
import pandas as pd
from dataclasses import dataclass, field
import io
from docx import Document

@dataclass
class DocumentProcessor:
    content: str = ""
    chunks: List[str] = field(default_factory=list)

    def process(self, file) -> None:
        file_name = file.name.lower()
        if file_name.endswith('.txt'):
            self._process_text(file)
        elif file_name.endswith('.docx'):
            self._process_docx(file)
        else:
            raise ValueError(f"Unsupported file type: {file.name}")
        self._split_content()

    def _process_text(self, file) -> None:
        content = file.getvalue()
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                self.content = content.decode(encoding)
                return
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode the file with supported encodings.")

    def _process_docx(self, file) -> None:
        docx_file = io.BytesIO(file.getvalue())
        doc = Document(docx_file)
        self.content = "\n".join(paragraph.text for paragraph in doc.paragraphs)

    def _split_content(self, chunk_size: int = 1000) -> None:
        words = self.content.split()
        self.chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

class VectorStore:
    def __init__(self, name: str):
        self.name = name
        self.data = pd.DataFrame(columns=['text', 'embedding'])

    def add(self, text: str) -> None:
        new_row = pd.DataFrame({'text': [text], 'embedding': [[0]*10]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)

    def query(self, question: str, k: int = 1) -> List[str]:
        return self.data['text'].head(k).tolist()

class ConversationManager:
    def __init__(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        self.messages.append({"role": "assistant", "content": message})

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

def simulate_chat(messages: List[Dict[str, str]]) -> str:
    return f"Response to: {messages[-1]['content']}"

def main():
    st.set_page_config(page_title="Legal Q&A System", layout="wide")
    st.title("Ask Me Anything About Law")

    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = ConversationManager("You are a legal specialist.")
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None

    uploaded_file = st.file_uploader("Upload your document", type=["txt", "docx"])

    if uploaded_file:
        try:
            doc_processor = DocumentProcessor()
            doc_processor.process(uploaded_file)
            
            vector_store = VectorStore(uploaded_file.name)
            for chunk in doc_processor.chunks:
                vector_store.add(chunk)
            
            st.session_state['vector_store'] = vector_store
            st.success("Document processed and indexed successfully!")
        except ValueError as e:
            st.error(f"Error processing file: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    question = st.text_input("Your Question:", placeholder="Ask about the law...")

    if st.button("Ask") and question:
        if st.session_state['vector_store']:
            relevant_text = st.session_state['vector_store'].query(question)[0]
            prompt = f"""
            Based on the following text extracted from the legislation:
            <extracted text>
            {relevant_text}
            </extracted text>
            Answer the following question:
            <question>
            {question}
            </question>
            Make sure to reference your answer according to the extracted text.
            """
        else:
            prompt = question

        st.session_state['conversation'].add_user_message(prompt)
        response = simulate_chat(st.session_state['conversation'].get_messages())
        st.session_state['conversation'].add_assistant_message(response)

    st.subheader("Conversation History")
    for message in st.session_state['conversation'].get_messages()[1:]:  # Skip system message
        st.text(f"{message['role'].capitalize()}: {message['content']}")

if __name__ == "__main__":
    main()