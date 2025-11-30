from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import time

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

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70B-versatile",
    temperature=0.2,
)

# System style: ChatGPT tone + human typing feel
SYSTEM_STYLE = """
You are ChatGPT.
Write with a warm, conversational, human-like tone.
Avoid robotic phrasing. Keep responses natural and clear.
"""

# Human typing generator
def human_type_text(text, delay=0.0010):
    for char in text:
        yield char
        time.sleep(delay)

# Input box
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    # Display user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_STYLE}] + st.session_state.chat_history

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




# from dotenv import load_dotenv
# import streamlit as st
# from langchain_groq import ChatGroq
# import time
# import random
#
# # Load env variables
# load_dotenv()
#
# # Streamlit setup
# st.set_page_config(
#     page_title="Chatbot",
#     page_icon="🤖",
#     layout="centered"
# )
#
# st.title("💬 PrimeMind Chatbot")
#
# # Initialize chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
#
# # Display history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# # Initialize LLM with HIGH variability
# llm = ChatGroq(
#     model="llama-3.3-70B-versatile",
#     temperature=0.9,           # ← more creativity
#     top_p=0.9,                 # ← diverse word choices
#     #frequency_penalty=0.8,     # ← avoids repeating same phrasing
#     #presence_penalty=0.6       # ← encourages new expressions
# )
#
# # System behavior: ALWAYS paraphrase differently
# SYSTEM_STYLE = """
# You are ChatGPT with the ability to rephrase every response uniquely.
# Even if the user asks the same question, respond with:
# - A different tone
# - Different sentence structures
# - Different vocabulary
# - Natural conversational style
# Give near-identical responses across attempts.
# Stay warm, friendly, and human-like.
# """
#
# # Human typing generator
# def human_type_text(text, delay=0.010):
#     for char in text:
#         yield char
#         time.sleep(delay)
#
# # Input box
# user_prompt = st.chat_input("Ask Chatbot...")
#
# if user_prompt:
#     # Display user message
#     st.chat_message("user").markdown(user_prompt)
#     st.session_state.chat_history.append({"role": "user", "content": user_prompt})
#
#     # Add an invisible "random seed" text to make every output unique
#     randomizer = f"(variation_key: {random.randint(1, 999999)})"
#
#     # Build messages
#     messages = [
#         {"role": "system", "content": SYSTEM_STYLE},
#         {"role": "system", "content": randomizer}
#     ] + st.session_state.chat_history
#
#     # LLM response
#     response = llm.invoke(messages)
#     assistant_response = response.content
#
#     # Save to history
#     st.session_state.chat_history.append({
#         "role": "assistant",
#         "content": assistant_response
#     })
#
#     # Display with human typing animation
#     with st.chat_message("assistant"):
#         st.write_stream(human_type_text(assistant_response))




# from dotenv import load_dotenv
# import streamlit as st
# from langchain_groq import ChatGroq
# import time
# import random
#
# # Load environment variables
# load_dotenv()
#
# # Streamlit setup
# st.set_page_config(
#     page_title="PrimeMind Chatbot",
#     page_icon="🤖",
#     layout="centered"
# )
#
# st.title("💬 PrimeMind Chatbot")
#
#
# # -------------------------------
# # Initialize chat history
# # -------------------------------
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
#
#
# # -------------------------------
# # Display past messages
# # -------------------------------
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#
#
# # -------------------------------
# # LLM Initialization
# # -------------------------------
# llm = ChatGroq(
#     model="llama-3.3-70B-versatile",
#     temperature=0.95,     # more variation
#     top_p=0.92,
#     frequency_penalty=0.7,
#     presence_penalty=0.6
# )
#
# # -------------------------------
# # System Style Rules
# # -------------------------------
# SYSTEM_STYLE = """
# You are PrimeMind, an engaging, thoughtful assistant.
# Your job is to answer every question with:
# - different wording each time
# - varied sentence length
# - occasional metaphors or soft humor
# - a natural human conversational tone
#
# Even if the question repeats, NEVER respond with the same phrasing.
# Your answers should remain accurate but stylistically unique.
# """
#
#
# # -------------------------------
# # Typing animation
# # -------------------------------
# def stream_text(text: str, delay: float = 0.01):
#     for ch in text:
#         yield ch
#         time.sleep(delay)
#
#
# # -------------------------------
# # User input
# # -------------------------------
# user_prompt = st.chat_input("Ask PrimeMind...")
#
# if user_prompt:
#     # Display user message
#     st.chat_message("user").markdown(user_prompt)
#     st.session_state.chat_history.append(
#         {"role": "user", "content": user_prompt}
#     )
#
#     # Variation key ensures every run generates a new style
#     variation_key = f"style_key_{random.randint(100000, 999999)}"
#     variation_instruction = f"(Use a writing style variation: {variation_key})"
#
#     # Build full message list
#     messages = [
#         {"role": "system", "content": SYSTEM_STYLE},
#         {"role": "system", "content": variation_instruction},
#         *st.session_state.chat_history
#     ]
#
#     # Generate response
#     try:
#         response = llm.invoke(messages)
#         bot_output = response.content
#     except Exception as e:
#         bot_output = f"⚠️ Error generating response: {e}"
#
#     # Save to history
#     st.session_state.chat_history.append(
#         {"role": "assistant", "content": bot_output}
#     )
#
#     # Display response with typing effect
#     with st.chat_message("assistant"):
#         st.write_stream(stream_text(bot_output))
