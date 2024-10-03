import streamlit as st
from dataclasses import dataclass

@dataclass
class ThemeColors:
    background: str = "#FDE5EC"
    primary: str = "#FC5C9C"
    secondary: str = "#C5EBAA"
    text: str = "#333333"
    accent: str = "#FF90BC"

class StreamlitThemer:
    def __init__(self, colors: ThemeColors = ThemeColors()):
        self.colors = colors

    def generate_css(self):
        return f"""
        <style>
        .stApp {{
            background-color: {self.colors.background};
        }}
        .main-title {{
            color: {self.colors.primary};
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            font-family: Arial, sans-serif;
        }}
        .stTextArea textarea {{
            background-color: {self.colors.secondary};
            color: {self.colors.text};
        }}
        .stFileUploader {{
            background-color: {self.colors.accent};
        }}
        .stTextInput input {{
            background-color: {self.colors.secondary};
            color: {self.colors.text};
        }}
        .stButton button {{
            background-color: {self.colors.primary};
            color: white;
        }}
        .custom-box {{
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .custom-box.left {{
            background-color: {self.colors.secondary};
        }}
        .custom-box.right {{
            background-color: {self.colors.accent};
        }}
        </style>
        """

    def apply_theme(self):
        st.markdown(self.generate_css(), unsafe_allow_html=True)

    def custom_header(self, text):
        st.markdown(f'<h1 class="main-title">{text}</h1>', unsafe_allow_html=True)

    def custom_box(self, text, side='left'):
        st.markdown(f'<div class="custom-box {side}">{text}</div>', unsafe_allow_html=True)

def initialize_theme():
    theme = StreamlitThemer()
    theme.apply_theme()
    return theme

# Usage example
if __name__ == "__main__":
    theme = initialize_theme()
    theme.custom_header("Welcome to My App")
    
    col1, col2 = st.columns(2)
    with col1:
        theme.custom_box("This is a left box", "left")
    with col2:
        theme.custom_box("This is a right box", "right")

    st.file_uploader("Upload a file")
    st.text_input("Enter some text")
    st.button("Click me")