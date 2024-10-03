import streamlit as st
from typing import List, Dict, Literal
from dataclasses import dataclass, field
from enum import Enum
import io

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    role: Role
    content: str

class DialogueManager:
    def __init__(self, system_message: str = "You are an AI assistant."):
        self.messages: List[Message] = [Message(Role.SYSTEM, system_message)]

    def add_message(self, role: Role, content: str) -> None:
        self.messages.append(Message(role, content))

    def render_conversation(self) -> None:
        for msg in self.messages[1:]:  # Skip system message
            if msg.role == Role.USER:
                st.markdown(f"**You**: {msg.content}")
            elif msg.role == Role.ASSISTANT:
                st.markdown(f"**Assistant**: {msg.content}")

    def export_conversation(self) -> str:
        buffer = io.StringIO()
        buffer.write("# Conversation Log\n\n")
        for msg in self.messages[1:]:  # Skip system message
            buffer.write(f"**{msg.role.value.capitalize()}**: {msg.content}\n\n")
        return buffer.getvalue()

def create_download_button(dialogue_manager: DialogueManager) -> None:
    content = dialogue_manager.export_conversation()
    st.download_button(
        label="Download Conversation",
        data=content,
        file_name="conversation_log.md",
        mime="text/markdown"
    )

def main():
    st.title("AI Conversation Interface")

    dialogue = DialogueManager()

    user_input = st.text_input("Your message:")
    if st.button("Send") and user_input:
        dialogue.add_message(Role.USER, user_input)
        # Simulated AI response (in a real app, you'd call an AI service here)
        ai_response = f"This is a simulated response to: {user_input}"
        dialogue.add_message(Role.ASSISTANT, ai_response)

    st.subheader("Conversation:")
    dialogue.render_conversation()

    create_download_button(dialogue)

if __name__ == "__main__":
    main()