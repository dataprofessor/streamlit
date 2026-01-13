# Streamlit AGENTS.md

> **Building a Streamlit app couldn't be easier. Vibe code your next app with the help of this AGENTS.md file.**

A comprehensive [AGENTS.md](https://agents.md) file that guides AI coding assistants (Cursor, Copilot, Claude, etc.) to build production-ready Streamlit applications. Answer a few questions, get a complete app. It's that simple.

## Quick Start

### Option 1: Quick Build

Reference the file with your instruction:

```
@AGENTS.md build a chatbot using Snowflake Cortex for Community Cloud
```

The AI builds your app immediately, asking only clarifying questions if needed.

### Option 2: Guided Flow

Just reference the file:

```
@AGENTS.md
```

The AI asks you questions one at a time:

1. **"What would you like to build?"** → chatbot, dashboard, data tool
2. **"Where will it run?"** → local, Community Cloud, Snowflake
3. **"What's your data source?"** → CSV, Snowflake, APIs
4. **"Which LLM?"** → Cortex or OpenAI (only if AI app)

Then it builds a complete, deployment-ready app.

## What You Get

Every app created includes:

```
my_app/
├── app.py                      # Main application
├── requirements.txt            # Dependencies
├── README.md                   # Deployment instructions
└── .streamlit/
    └── secrets.toml.example    # Credentials template
```

## Supported App Types

| App Type | Patterns Used |
|----------|---------------|
| Chatbot | Chat interface, streaming, ai_complete |
| Dashboard | Plotly/Altair charts, metrics, caching |
| Data Analysis | File upload, DataFrame styling |
| RAG | Chat + Cortex Search + ai_complete |
| Multipage | st.navigation, session state |

## Deployment Targets

Works across all Streamlit deployment environments:

- ✅ **Local Development**
- ✅ **Streamlit Community Cloud**
- ✅ **Streamlit in Snowflake (SiS)**

The AI automatically handles environment differences (e.g., no `st.set_page_config()` for SiS).

## Installation

### Method 1: Copy to Your Project

Download `AGENTS.md` and place it in your project root:

```bash
curl -O https://raw.githubusercontent.com/<your-repo>/main/AGENTS.md
```

### Method 2: Reference Directly

In Cursor, you can reference the file from anywhere:

```
@/path/to/AGENTS.md build me a dashboard
```

## Key Patterns Included

### Universal Snowflake Connection

```python
@st.cache_resource
def get_session():
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except:
        from snowflake.snowpark import Session
        return Session.builder.configs(st.secrets["connections"]["snowflake"]).create()
```

### Snowflake Cortex LLM (ai_complete)

```python
from snowflake.snowpark.functions import ai_complete

@st.cache_data(show_spinner=False)
def call_llm(prompt: str, model: str = "claude-3-5-sonnet") -> str:
    df = session.range(1).select(
        ai_complete(model=model, prompt=prompt).alias("response")
    )
    response_raw = df.collect()[0][0]
    return json.loads(response_raw)
```

### Chat with Streaming

```python
if prompt := st.chat_input("Message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = st.write_stream(stream_response(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
```

## Where These Patterns Come From

This AGENTS.md is built from real-world experience:

- **[30 Days of AI](https://30daysofai.streamlit.app/)** — A 30-day series teaching AI app development with Streamlit and Snowflake Cortex ([GitHub](https://github.com/streamlit/30daysofai))
- **Years of Streamlit Development** — Patterns from dozens of production apps built 2024-2025

Every pattern has been validated in production.

## Examples

The `examples/` folder contains two apps built entirely using the guided AGENTS.md flow:

### 1. Chatbot (`examples/chatbot_app/`)

A conversational AI chatbot powered by Snowflake Cortex.

**Questions Asked → Answers Given:**

| Question | Answer |
|----------|--------|
| What would you like to build? | a simple chatbot |
| Where will it run? | Community Cloud |
| Which LLM? | Cortex |

**Features:** Chat interface, model selector (Claude/Llama/Mistral), conversation history, response caching.

### 2. Stock Dashboard (`examples/stock_dashboard/`)

A real-time stock dashboard using Yahoo Finance data.

**Questions Asked → Answers Given:**

| Question | Answer |
|----------|--------|
| What would you like to build? | dashboard |
| Where will it run? | Snowflake |
| What's your data source? | yfinance |

**Features:** Candlestick charts, volume analysis, key metrics, time period selector, no `st.set_page_config()` (SiS-compatible).

---

## File Structure

```
streamlit/
├── AGENTS.md                  # The main file (1,100+ lines of patterns)
├── README.md                  # This file
├── blog_using_agents_md.md    # Technical blog post
└── examples/
    ├── chatbot_app/           # Chatbot example
    │   ├── app.py
    │   ├── requirements.txt
    │   └── README.md
    └── stock_dashboard/       # Dashboard example
        ├── app.py
        ├── requirements.txt
        └── README.md
```

## Contributing

Found a pattern that should be included? Open an issue or PR!

## Resources

- [AGENTS.md Standard](https://agents.md)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Snowflake Cortex AI](https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql)
- [Streamlit in Snowflake](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit)

App patterns were borrowed from the #30DaysOfAI challenge
- [30 Days of AI - App](https://30daysofai.streamlit.app/)
- [30 Days of AI - GitHub](https://github.com/streamlit/30daysofai)
---

**Built with ❤️ for the Streamlit community**
