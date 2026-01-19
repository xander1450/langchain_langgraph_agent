import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Ensure GROQ_API_KEY is set in environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it as an environment variable."
    )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=2
)

@tool
def get_weather(city: str) -> str:
    """Use this to get the weather for a specific city."""
    if "london" in city.lower():
        return "It is rainy and 15°C in London."
    elif "vijayawada" in city.lower():
        return "It is sunny and 32°C in Vijayawada."
    else:
        return "Weather data not available for this city."

@tool
def multiply(a: int, b: int) -> int:
    """Use this to multiply two numbers."""
    return a * b

tools = [get_weather, multiply]

# ---------------------------------------------------------
# STEP 3: CREATE THE AGENT (The Body)
# ---------------------------------------------------------

agent_executor = create_react_agent(llm, tools)

# ---------------------------------------------------------
# STEP 4: STREAMLIT UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Agent Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Agent Chat")
st.caption("Ask me about the weather or math problems!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke(
                {"messages": [("user", prompt)]}
            )
            final_answer = response["messages"][-1].content

        st.markdown(final_answer)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer}
    )
