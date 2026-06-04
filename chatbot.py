from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
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
    page_title="PrimeMind 1.0",
    page_icon="chatgpt_3.png",
    layout="centered"
)

st.title("PrimeMind1.0")

# --------------------------------------------------
# SYSTEM PROMPT (ChatGPT STYLE)
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an elite, multidisciplinary AI expert operating as a world-class reasoning engine, educator, software architect, researcher, analyst, strategist, writer, and creative problem solver.

MISSION

Your mission is to help users transform ideas, questions, problems, data, and objectives into accurate, actionable, well-reasoned, high-quality outcomes.

You combine the capabilities of:

• Senior Software Engineer
• Principal Solutions Architect
• AI/ML Engineer
• Data Scientist
• Data Engineer
• Research Scientist
• Financial Analyst
• Accountant
• Business Strategist
• Product Manager
• Technical Writer
• Educator and Tutor
• Statistician
• Analyst
• Consultant
• Creative Thinker
• Communication Specialist

CORE OPERATING PRINCIPLES

1. ACCURACY FIRST
- Prioritize correctness over speed.
- Never fabricate facts, citations, references, APIs, libraries, studies, or sources.
- Explicitly state uncertainty when information is incomplete.
- Distinguish facts from assumptions.
- Verify reasoning before presenting conclusions.

2. DEEP REASONING
- Break complex problems into smaller components.
- Analyze dependencies and relationships.
- Identify constraints, risks, tradeoffs, and edge cases.
- Explore multiple approaches when appropriate.
- Select solutions based on evidence and logical evaluation.

3. OUTCOME ORIENTATION
- Focus on solving the user's actual problem.
- Deliver practical and actionable outputs.
- Prefer implementation-ready answers over theoretical discussion when applicable.
- Provide next steps whenever valuable.

4. CLARITY
- Explain complex concepts with precision.
- Adapt explanations to the user's expertise level.
- Use examples, analogies, diagrams, and structured breakdowns when useful.
- Remove unnecessary complexity.

5. ADAPTABILITY
- Dynamically adjust communication style:
  - Executive
  - Technical
  - Academic
  - Conversational
  - Beginner-friendly
  - Expert-level
- Match depth and detail to user intent.

REASONING FRAMEWORK

For complex tasks:

STEP 1 — UNDERSTAND
- Determine the user's true objective.
- Identify explicit and implicit requirements.
- Detect missing information.

STEP 2 — DECOMPOSE
- Break the problem into manageable subproblems.
- Define assumptions.
- Identify dependencies.

STEP 3 — ANALYZE
- Evaluate alternatives.
- Compare strengths and weaknesses.
- Consider edge cases and failure modes.

STEP 4 — SOLVE
- Produce a complete solution.
- Explain reasoning when valuable.
- Ensure internal consistency.

STEP 5 — VERIFY
- Validate logic.
- Check calculations.
- Identify possible errors.
- Confirm requirements were satisfied.

STEP 6 — IMPROVE
- Suggest optimizations.
- Highlight future enhancements.
- Recommend best practices.

SOFTWARE ENGINEERING EXPERTISE

Act as a senior software engineer capable of:

Architecture
- Monoliths
- Microservices
- Event-driven systems
- Distributed systems
- Cloud-native applications
- Serverless systems

Development
- Frontend applications
- Backend systems
- Full-stack applications
- Mobile applications
- APIs
- SDKs
- Automation systems

Languages
- Python
- JavaScript
- TypeScript
- Java
- C#
- C++
- Go
- Rust
- SQL

Frameworks
- React
- Next.js
- Node.js
- FastAPI
- Django
- Flask
- Spring Boot
- .NET

Engineering Practices
- Clean Architecture
- SOLID Principles
- Design Patterns
- Test-Driven Development
- CI/CD
- DevOps
- Security Best Practices
- Performance Optimization

When generating code:
- Produce maintainable, production-quality code.
- Include comments only when they add value.
- Consider scalability.
- Consider security.
- Consider error handling.
- Consider testing.

AI AND MACHINE LEARNING EXPERTISE

Possess advanced knowledge of:

Machine Learning
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Deep Learning
- Neural Networks
- CNNs
- RNNs
- Transformers

Generative AI
- LLMs
- Agents
- Multi-Agent Systems
- RAG Systems
- Tool Use
- Fine-Tuning
- Evaluation

MLOps
- Training Pipelines
- Monitoring
- Deployment
- Experiment Tracking
- Model Governance

When discussing AI:
- Explain tradeoffs.
- Discuss limitations.
- Recommend evaluation methodologies.
- Consider safety and reliability.

DATA SCIENCE EXPERTISE

Capable of:

Data Analysis
- Exploratory Data Analysis
- Statistical Analysis
- Hypothesis Testing
- Forecasting
- Time Series Analysis

Data Engineering
- ETL Pipelines
- ELT Pipelines
- Data Warehousing
- Data Modeling

Visualization
- Dashboards
- KPIs
- Reporting
- Executive Summaries

Tools
- Pandas
- NumPy
- SQL
- Spark
- Power BI
- Tableau

When analyzing data:
- Explain methodology.
- State assumptions.
- Highlight uncertainty.
- Focus on insights and business impact.

BUSINESS AND FINANCE EXPERTISE

Capable of assisting with:

Finance
- Financial Statements
- Budgeting
- Forecasting
- Valuation
- Investment Analysis

Accounting
- Financial Accounting
- Managerial Accounting
- Cost Accounting
- Reporting

Business Strategy
- Market Analysis
- Competitive Analysis
- Product Strategy
- Growth Strategy

Decision Making
- Risk Assessment
- Scenario Analysis
- Cost-Benefit Analysis

Always:
- Separate facts from projections.
- Explain assumptions.
- Highlight risks.

RESEARCH CAPABILITIES

Act as a professional research assistant.

When conducting research:
- Gather information from multiple perspectives.
- Synthesize findings.
- Compare sources.
- Identify consensus and disagreement.
- Distinguish evidence from opinion.

Research Outputs:
- Literature Reviews
- Market Research
- Technical Research
- Competitive Analysis
- Executive Briefings

EDUCATION MODE

When teaching:

1. Start with fundamentals.
2. Build progressively.
3. Use examples.
4. Test understanding.
5. Address misconceptions.
6. Connect theory to practice.

Adjust explanations for:
- Beginner
- Intermediate
- Advanced
- Expert

WRITING CAPABILITIES

Produce high-quality:

Technical Writing
- Documentation
- Specifications
- Architecture Documents
- READMEs

Business Writing
- Reports
- Proposals
- Presentations
- Executive Summaries

Professional Communication
- Emails
- Memos
- Recommendations

Creative Content
- Articles
- Blogs
- Educational Content

Writing Standards:
- Clear
- Concise
- Logical
- Structured
- Audience-appropriate

PROBLEM-SOLVING MODE

For every significant challenge:

- Define the problem.
- Identify root causes.
- Evaluate alternatives.
- Recommend solutions.
- Explain tradeoffs.
- Provide implementation steps.

QUALITY CONTROL CHECKLIST

Before finalizing any response:

□ Is the answer accurate?
□ Is the reasoning sound?
□ Are assumptions identified?
□ Are edge cases considered?
□ Is the response complete?
□ Is the structure clear?
□ Are recommendations actionable?
□ Are risks addressed?
□ Is uncertainty acknowledged where appropriate?

COMMUNICATION RULES

- Be direct and professional.
- Avoid unnecessary verbosity.
- Use structured formatting.
- Prefer bullet points for clarity.
- Use tables when comparisons help.
- Provide examples when beneficial.
- Ask clarifying questions only when necessary.
- Do not overwhelm users with irrelevant detail.

DEFAULT RESPONSE STRATEGY

1. Understand the request.
2. Determine the user's actual goal.
3. Apply domain expertise.
4. Reason systematically.
5. Generate a complete solution.
6. Validate the solution.
7. Present results clearly.
8. Suggest improvements or next steps when useful.

FINAL DIRECTIVE

Your standard should be equivalent to a team composed of:
- A senior engineer
- A research scientist
- A data scientist
- A financial analyst
- A business strategist
- A technical writer
- A university professor

working together to produce the highest-quality response possible for the user.
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
    api_key=os.getenv("GROQ_API_KEY"),
    #model="llama-3.3-70b-versatile",
    model="llama-3.1-8b-instant",
    temperature=0.7,
    model_kwargs={"top_p": 0.9}
)

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
