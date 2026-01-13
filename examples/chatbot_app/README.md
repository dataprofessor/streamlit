# Chatbot

> **TLDR:** A simple chatbot powered by Snowflake Cortex AI with conversation memory.

## Features

- Chat interface with message history
- Multiple LLM models (Claude, Llama, Mistral)
- Conversation context (remembers previous messages)
- Response caching for performance

## Run Locally

```bash
# Clone the repo
git clone <your-repo-url>
cd chatbot_app

# Install dependencies
pip install -r requirements.txt

# Add secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your Snowflake credentials

# Run
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add secrets in the dashboard (Settings → Secrets)
5. Deploy!

## Deploy to Streamlit in Snowflake

1. Create a Streamlit app in Snowflake
2. Upload app.py and requirements.txt
3. Run from Snowflake UI

---

Built with [Streamlit](https://streamlit.io) and [Snowflake Cortex](https://docs.snowflake.com/en/user-guide/snowflake-cortex)
