from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import time
import random
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

# Streamlit setup
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("💬 PrimeMind")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize LLM with HIGH variability
llm = ChatGroq(
    #api_key=os.getenv("GROQ_API_KEY"),
    api_key=st.secrets["GROQ_API_KEY"],
    #model="llama-3.3-70B-versatile",
    model="llama-3.1-8b-instant",
    temperature=0.9,           # ← more creativity
    model_kwargs={"top_p": 0.9},                 # ← diverse word choices
    #frequency_penalty=0.8,     # ← avoids repeating same phrasing
    #presence_penalty=0.6       # ← encourages new expressions
)

# System behavior: ALWAYS paraphrase differently
SYSTEM_STYLE = """
You are a helpful, intelligent AI assistant.

Guidelines:
- Provide clear, accurate, and concise responses.
- Focus on correctness and practical usefulness.
- Explain concepts simply when needed, without unnecessary verbosity.
- Maintain a professional, friendly, and human-like tone.
- Adapt your explanations to the user’s level of understanding.
- Avoid repetition, filler, or robotic phrasing.

Your goal is to deliver reliable, easy-to-understand answers that genuinely help the user move forward.
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



