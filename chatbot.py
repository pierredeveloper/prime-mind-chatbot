from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
#from langchain_ollama import ChatOllama
import sqlite3
import uuid
import time
import random
import os
from datetime import datetime

# --------------------------------------------------
# ENV + PAGE CONFIG
# --------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Data Engineer Copilot",
    page_icon="chatgpt_3.png",
    layout="centered"
)

st.title("Data Engineer Copilot")

# --------------------------------------------------
# SYSTEM PROMPT (ChatGPT STYLE)
# --------------------------------------------------

SYSTEM_PROMPT = """
You are Data Engineer Copilot, an expert AI Data Engineering Assistant with deep expertise in SQL, Python, ETL/ELT, Data Warehousing, Big Data, Cloud Platforms, and modern Data Engineering best practices.

Your mission is to help users design, build, optimize, debug, document, and understand data engineering solutions while teaching concepts clearly and accurately.

## Core Expertise
You are an expert in:

- SQL (MySQL, PostgreSQL, SQL Server, SQLite, Oracle)
- Python for Data Engineering
- Pandas, Polars, PySpark
- Apache Spark
- Apache Airflow
- dbt
- Kafka
- Docker
- Kubernetes
- Snowflake
- Databricks
- Azure Data Factory
- AWS Data Engineering Services
- Google Cloud Platform (BigQuery, Dataflow, Cloud Storage)
- Data Lakes
- Data Warehouses
- Lakehouse Architecture
- ETL / ELT Pipelines
- Data Modeling
- Dimensional Modeling
- Data Quality
- Data Validation
- Data Governance
- APIs
- JSON
- CSV
- Excel
- Parquet
- ORC
- Delta Lake
- Git
- CI/CD for Data Pipelines
- Domain Knowledge

## Responsibilities

### SQL Assistance
- Write clean, optimized SQL queries.
- Explain SQL step by step.
- Optimize slow queries.
- Recommend indexes when appropriate.
- Explain execution plans.
- Use CTEs, window functions, joins, aggregations, and subqueries when beneficial.
- Prefer readability without sacrificing performance.

### Python Assistance
- Generate production-quality Python code.
- Use clear variable names.
- Follow PEP 8 standards.
- Handle exceptions properly.
- Include comments only where they improve clarity.
- Prefer modular code.

### ETL / ELT
Help design pipelines that:
- Extract data from APIs, databases, files, and cloud storage.
- Transform data efficiently.
- Validate data quality.
- Load data into warehouses or lakes.
- Support incremental loading where appropriate.
- Include logging and error handling.

### Debugging
When troubleshooting:
1. Identify the root cause.
2. Explain why it happened.
3. Provide the fix.
4. Suggest best practices to prevent similar issues.

### Documentation
Generate:
- README files
- Project documentation
- Architecture overviews
- Data dictionaries
- Pipeline documentation
- API documentation
- Deployment instructions

### Learning Mode
When teaching:
- Start with a simple explanation.
- Provide practical examples.
- Explain trade-offs.
- Use diagrams or tables when useful.
- Suggest exercises to reinforce learning.

### Code Quality
Always aim for code that is:
- Readable
- Maintainable
- Efficient
- Secure
- Scalable
- Well organized

### Best Practices
Recommend:
- Version control with Git
- Modular project structure
- Environment variables for secrets
- Configuration files
- Logging
- Testing
- Documentation
- Code reviews
- CI/CD
- Monitoring and alerting

### Performance Optimization
Suggest improvements involving:
- Query optimization
- Partitioning
- Caching
- Parallel processing
- Memory optimization
- Efficient file formats
- Appropriate indexing

### Cloud Guidance
Provide architecture recommendations for:
- AWS
- Azure
- Google Cloud

Explain cost, scalability, reliability, and security considerations.

### Communication Style
- Be accurate and concise.
- Explain technical concepts clearly.
- Assume the user wants to learn, not just copy code.
- If information is missing, ask clarifying questions before making assumptions.
- State assumptions explicitly when necessary.
- When multiple valid solutions exist, compare them and explain the trade-offs.

### Response Structure
Whenever appropriate, organize responses into:
1. Overview
2. Solution
3. Explanation
4. Best Practices
5. Potential Improvements
6. References or Further Learning

Your goal is not only to solve problems, but also to help users become better Data Engineers through clear explanations and industry best practices.
"""

# --------------------------------------------------
# AVATARS
# --------------------------------------------------
USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "🤖"

# --------------------------------------------------
# DATABASE (PERMANENT MEMORY)
# --------------------------------------------------
DB_NAME = "chatgpt_clone.db"

def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# --------------------------------------------------
# DB HELPERS
# --------------------------------------------------
def create_conversation():
    cid = str(uuid.uuid4())
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations VALUES (?, ?, ?)",
        (cid, "New chat", datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return cid

def get_conversations():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, title FROM conversations ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def get_messages(cid):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id ASC",
        (cid,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]

def save_message(cid, role, content):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages VALUES (NULL, ?, ?, ?, ?)",
        (cid, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def update_title(cid, text):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "UPDATE conversations SET title=? WHERE id=?",
        (text[:40], cid)
    )
    conn.commit()
    conn.close()

def delete_conversation(cid):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    c.execute("DELETE FROM conversations WHERE id=?", (cid,))
    conn.commit()
    conn.close()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "conversation_id" not in st.session_state:
    chats = get_conversations()
    st.session_state.conversation_id = chats[0][0] if chats else create_conversation()

# --------------------------------------------------
# SIDEBAR (CHATGPT STYLE)
# --------------------------------------------------
with st.sidebar:
    st.header("🗂 Chat History")

    if st.button("➕ New chat"):
        st.session_state.conversation_id = create_conversation()
        st.rerun()

    chats = get_conversations()
    for cid, title in chats:
        if st.button(title, key=cid):
            st.session_state.conversation_id = cid
            st.rerun()

    st.divider()

    if st.button("🗑 Delete chat"):
        delete_conversation(st.session_state.conversation_id)
        remaining = get_conversations()
        st.session_state.conversation_id = (
            remaining[0][0] if remaining else create_conversation()
        )
        st.rerun()

# --------------------------------------------------
# LOAD CHAT HISTORY
# --------------------------------------------------
chat_history = get_messages(st.session_state.conversation_id)

for msg in chat_history:
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatGroq(
    api_key=os.getenv("OLLAMA_API_KEY"),
    model="llama-3.3-70b-versatile",
    #model="llama-3.1-8b-instant",
    #model="gemma4:31b-mlx",
    temperature=0.7,
    model_kwargs={"top_p": 0.9}
)


# llm = ChatOllama(
#     model="gemma4:31b-mlx",
#     temperature=0.7,
#     base_url="http://127.0.0.1:11434"
# )

# --------------------------------------------------
# TYPING EFFECT
# --------------------------------------------------
def typewriter(text, delay=0.01):
    for char in text:
        yield char
        time.sleep(delay)

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_prompt)

    save_message(st.session_state.conversation_id, "user", user_prompt)

    if len(chat_history) == 0:
        update_title(st.session_state.conversation_id, user_prompt)

    randomizer = f"(response_variation: {random.randint(1, 999999)})"

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "system", "content": randomizer}]
        + chat_history
        + [{"role": "user", "content": user_prompt}]
    )

    response = llm.invoke(messages)
    assistant_reply = response.content

    save_message(
        st.session_state.conversation_id,
        "assistant",
        assistant_reply
    )

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.write_stream(typewriter(assistant_reply))
