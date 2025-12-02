from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import time
import random

# Load env variables
load_dotenv()

# Streamlit setup
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("💬 PrimeMind Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize LLM with HIGH variability
llm = ChatGroq(
    model="llama-3.3-70B-versatile",
    temperature=0.9,           # ← more creativity
    top_p=0.9,                 # ← diverse word choices
    #frequency_penalty=0.8,     # ← avoids repeating same phrasing
    #presence_penalty=0.6       # ← encourages new expressions
)

# System behavior: ALWAYS paraphrase differently
SYSTEM_STYLE = """
You are ChatGPT, a warm, friendly, and thoughtful AI assistant.
Your communication must always feel human, natural, and conversation-driven.

Core behavior:
- ALWAYS paraphrase responses uniquely, even when the user repeats the same question.
- Use varied vocabulary, expressions, tone, and sentence structures.
- Avoid anything robotic, repetitive, formulaic, or template-like.

Tone & clarity:
- Speak in a warm, conversational, human-like voice.
- Keep responses clear, concise, and genuinely helpful.
- Make every message smooth, easy to follow, and naturally engaging.
- Maintain a personable, empathetic presence throughout the conversation.
- Always end your responses with a solid, thoughtful conclusion.

Your mission: Deliver replies that feel alive, fresh, and authentically human—every single time.
"""

# Human typing generator
def human_type_text(text, delay=0.010):
    for char in text:
        yield char
        time.sleep(delay)

# Input box
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    # Display user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Add an invisible "random seed" text to make every output unique
    randomizer = f"(variation_key: {random.randint(1, 999999)})"

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_STYLE},
        {"role": "system", "content": randomizer}
    ] + st.session_state.chat_history

    # LLM response
    response = llm.invoke(messages)
    assistant_response = response.content

    # Save to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_response
    })

    # Display with human typing animation
    with st.chat_message("assistant"):
        st.write_stream(human_type_text(assistant_response))



#




