import streamlit as st
import json
from snowflake.snowpark.functions import ai_complete

# Page config (Community Cloud)
st.set_page_config(
    page_title="Chatbot",
    page_icon=":material/chat:",
    layout="centered"
)

# Connect to Snowflake
@st.cache_resource
def get_session():
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except:
        from snowflake.snowpark import Session
        return Session.builder.configs(st.secrets["connections"]["snowflake"]).create()

session = get_session()

# LLM call with caching
@st.cache_data(show_spinner=False)
def call_llm(prompt: str, model: str = "claude-3-5-sonnet") -> str:
    """Call Snowflake Cortex LLM."""
    df = session.range(1).select(
        ai_complete(model=model, prompt=prompt).alias("response")
    )
    response_raw = df.collect()[0][0]
    response_json = json.loads(response_raw)
    
    if isinstance(response_json, dict) and "choices" in response_json:
        return response_json["choices"][0]["messages"]
    return str(response_json)

# App UI
st.title(":material/chat: Chatbot")

# Initialize chat history
st.session_state.setdefault("messages", [])

# Sidebar
with st.sidebar:
    st.header(":material/settings: Settings")
    model = st.selectbox(
        "Model:",
        ["claude-3-5-sonnet", "llama3.1-70b", "mistral-large"],
        key="model"
    )
    
    if st.button(":material/refresh: Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Build conversation context
    conversation = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}" 
        for m in st.session_state.messages
    ])
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = call_llm(conversation, model)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
