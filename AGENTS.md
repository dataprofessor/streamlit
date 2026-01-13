# AGENTS.md - Streamlit App Builder

## Instructions for AI Agent

**Two modes of operation:**

### Mode 1: User provides instruction
If user says `@AGENTS.md build a chatbot` or similar:
- Use their instruction as the starting point
- Infer reasonable defaults from context
- Ask only 1-2 clarifying questions if critical info is missing (deployment target, LLM provider)
- Then build the app using patterns below

### Mode 2: User just references file
If user only says `@AGENTS.md` with no instruction:
- Ask questions ONE AT A TIME, sequentially
- Start with: **"What would you like to build? (e.g., chatbot, dashboard, data tool)"**
- After they answer, ask: **"Where will it run? (local, Community Cloud, or Snowflake)"**
- After they answer, ask: **"What's your data source? (CSV uploads, Snowflake, APIs, or none)"**
- If they mentioned AI/chat, ask: **"Which LLM? (Snowflake Cortex or OpenAI)"**
- Then build - infer UI components from app type, add caching automatically where beneficial

---

## Pattern Selection Guide

| User Says | Use These Patterns |
|-----------|---------------------|
| chatbot, AI assistant | Chat Interface + Streaming + ai_complete |
| dashboard, visualization | Basic App + Plotly/Altair Charts |
| data analysis, explore data | File Upload + DataFrame Styling |
| RAG, search documents | Chat + Cortex Search + ai_complete |
| multipage | st.navigation + Session State |
| Snowflake, SiS | Omit st.set_page_config + Cortex patterns |

**Always:** Add `@st.cache_data` for LLM calls and data loading automatically.

---

## Files to Create

When building an app, always create these files:

1. **app.py** - Main application
2. **requirements.txt** - Dependencies with versions
3. **README.md** - GitHub-ready documentation (use template below)
4. **.streamlit/secrets.toml.example** - Credentials template (if secrets needed)

### README.md Template

```markdown
# [App Name]

> **TLDR:** [One sentence describing what the app does]

## Features

- [Feature 1]
- [Feature 2]
- [Feature 3]

## Run Locally

\```bash
# Clone the repo
git clone [repo-url]
cd [app-folder]

# Install dependencies
pip install -r requirements.txt

# Add secrets (if needed)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your credentials

# Run
streamlit run app.py
\```

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

Built with [Streamlit](https://streamlit.io)
```

**Note:** Replace bracketed placeholders with actual values. Remove secrets section if app doesn't need credentials.

---

# Reference Patterns

## Overview

Streamlit is a Python framework for building interactive data apps with minimal code. Apps are single Python scripts that run top-to-bottom on each user interaction, with automatic UI updates.

**This guide covers patterns for all deployment targets:**
| Environment | Use Case |
|-------------|----------|
| **Local** | Development, testing, prototyping |
| **Community Cloud** | Public apps, demos, portfolios |
| **Streamlit in Snowflake** | Enterprise apps with Snowflake data |
- Any long-running operations that need progress feedback?

### Quick Start Template Selection

Based on user answers, recommend starting template:

| User Need | Recommended Template |
|-----------|---------------------|
| Simple data dashboard | Basic App + Plotly Charts |
| LLM chatbot | Chat Interface Pattern + Streaming |
| Data analysis tool | File Upload + DataFrame Styling |
| RAG application | Chat + Cortex Search + ai_complete |
| Multipage app | st.navigation with appropriate layout |
| Enterprise Snowflake app | SiS patterns + Cortex AI |

---

## Setup Commands

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Run with custom port
streamlit run app.py --server.port 8501

# Run in headless mode (for testing)
streamlit run app.py --server.headless true
```

## Recommended Project Structure

```
project/
├── app.py                    # Main entry point
├── pages/                    # Multipage app pages
│   ├── home.py
│   ├── dashboard.py
│   └── settings.py
├── utils.py                  # Shared utilities
├── requirements.txt          # Dependencies with versions
├── .streamlit/
│   ├── config.toml          # App configuration
│   └── secrets.toml         # Secrets (never commit!)
├── assets/                   # Static files, images
├── data/                     # Data files (if applicable)
└── AGENTS.md                 # Agent instructions
```

## Code Style Guidelines

### General Python
- Use `snake_case` for functions and variables
- Use type hints for function parameters when clarity is needed
- Prefer f-strings for string formatting
- Keep functions focused and under 50 lines when possible
- Add docstrings to all functions with Args/Returns sections

### Streamlit-Specific
- **Always** use `st.` prefix for all Streamlit functions
- Place `import streamlit as st` as first Streamlit import
- Use descriptive `key` parameters for widgets: `st.text_input("Name:", key="user_name")`
- Use Material icons: `:material/icon_name:` for visual consistency

### Import Order
```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# ... other third-party imports
from utils import helper_function  # Local imports last
```

---

## Core Streamlit Patterns

### Basic App Structure

```python
import streamlit as st

# Page config (Local & Community Cloud only - NOT for SiS)
# Must be first Streamlit command when used
st.set_page_config(
    page_title="My App",
    page_icon=":material/rocket:",
    layout="wide"
)

# Initialize session state
st.session_state.setdefault("data", None)
st.session_state.setdefault("messages", [])

# Sidebar
with st.sidebar:
    st.header(":material/settings: Settings")
    option = st.selectbox("Choose:", ["Option A", "Option B"])

# Main content
st.title(":material/home: My App")
st.write("Description of what the app does")

# App logic here...

```

**Note:** `st.set_page_config()` is **not supported in Streamlit in Snowflake (SiS)**. For SiS apps, omit this call entirely - the page configuration is managed by Snowflake.

### Session State Management

```python
# Method 1: setdefault (preferred - concise)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("counter", 0)

# Method 2: Check existence (explicit)
if "df" not in st.session_state:
    st.session_state.df = None

# Method 3: Safe access with get
value = st.session_state.get("key", default_value)

# Update state
st.session_state.messages.append({"role": "user", "content": text})
st.session_state.counter += 1
```

**Best Practices:**
- Initialize all session state in the main app file (before page imports)
- Use descriptive key names: `uploaded_filename`, `target_column`, `model_results`
- Session state persists across reruns but not across browser restarts

### Caching for Performance

```python
# Cache data (for DataFrames, API responses, computations)
# Use for: Data loading, transformations, API calls, calculations
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load and return a DataFrame with caching."""
    return pd.read_csv(path)

@st.cache_data(ttl=3600)  # Cache expires after 1 hour
def fetch_api_data(endpoint: str) -> dict:
    """Fetch data from API with TTL."""
    response = requests.get(endpoint)
    return response.json()

# Cache LLM API responses (avoids duplicate API calls for same prompts)
@st.cache_data(show_spinner=False)
def call_llm(prompt: str, model: str = "gpt-4") -> str:
    """Call LLM API with caching to avoid redundant calls."""
    import openai
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# For Snowflake Cortex LLM - Method 1: ai_complete (Recommended)
# Works universally across SiS, Community Cloud, and local
import json
from snowflake.snowpark.functions import ai_complete

@st.cache_data(show_spinner=False)
def call_cortex_llm(prompt: str, model: str = "claude-3-5-sonnet") -> str:
    """Call Snowflake Cortex LLM using ai_complete."""
    df = session.range(1).select(
        ai_complete(model=model, prompt=prompt).alias("response")
    )
    response_raw = df.collect()[0][0]
    response_json = json.loads(response_raw)
    
    # Handle different response formats
    if isinstance(response_json, dict) and "choices" in response_json:
        return response_json["choices"][0]["messages"]
    return str(response_json)

# For Snowflake Cortex LLM - Method 2: Raw SQL
@st.cache_data(show_spinner=False)
def call_cortex_llm_sql(prompt: str, model: str = "llama3.1-70b") -> str:
    """Call Snowflake Cortex LLM with SQL."""
    response = session.sql("""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) as response
    """, params=[model, prompt]).collect()[0]['RESPONSE']
    return response

# Cache resources (for connections, models, expensive objects)
# Use for: Database connections, ML models, API clients
@st.cache_resource
def get_database_connection():
    """Get database connection with caching."""
    return create_connection()

@st.cache_resource
def load_model():
    """Load ML model once and reuse."""
    return joblib.load("model.pkl")

# Cache LLM client initialization (not the responses)
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client once and reuse."""
    import openai
    return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
```

**Never cache:**
- User input widgets
- Session state updates
- Mutable objects that get modified after caching

**LLM Caching Tips:**
- Use `@st.cache_data` for LLM responses (same prompt = same response)
- Set `ttl` parameter if responses should expire
- Cache the client with `@st.cache_resource`, responses with `@st.cache_data`

**Snowflake Cortex LLM Notes:**
- `ai_complete` from `snowflake.snowpark.functions` is recommended for universal compatibility
- Works across all environments (SiS, Community Cloud, local) without SSL issues
- Returns JSON that needs parsing with `json.loads()`
- Available models: `claude-3-5-sonnet`, `llama3.1-70b`, `mistral-large`, `claude-sonnet-4-5`

### Layout Components

```python
# Columns
col1, col2, col3 = st.columns([2, 1, 1])  # Proportional widths
with col1:
    st.write("Wide column")
with col2:
    st.metric("Metric", 42)

# Tabs
tab1, tab2 = st.tabs([":material/table_chart: Data", ":material/bar_chart: Charts"])
with tab1:
    st.dataframe(df)
with tab2:
    st.plotly_chart(fig)

# Expander
with st.expander("Show details", expanded=False):
    st.write("Hidden content here")

# Container with border
with st.container(border=True):
    st.subheader(":material/info: Section")
    st.write("Bordered content")

# Sidebar
with st.sidebar:
    st.selectbox("Choose:", options)
```

### Chat Interface Pattern

```python
# Initialize messages
st.session_state.setdefault("messages", [])

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Your message"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_llm_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

### Streaming LLM Responses

Use `st.write_stream()` to display LLM responses as they're generated for a better user experience:

```python
# Method 1: Direct streaming (when LLM returns a generator)
# Works with Snowflake Cortex Complete with stream=True
from snowflake.cortex import Complete

if st.button("Generate"):
    with st.spinner("Generating..."):
        stream_generator = Complete(
            session=session,
            model="claude-3-5-sonnet",
            prompt=user_prompt,
            stream=True  # Enable streaming
        )
        
        # st.write_stream displays chunks as they arrive
        st.write_stream(stream_generator)
```

```python
# Method 2: Custom generator (for APIs that don't return generators)
# Useful for OpenAI, Anthropic, or when you need more control
import time

def stream_llm_response(prompt: str):
    """Generator that yields response chunks."""
    # For OpenAI
    client = get_openai_client()
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if st.button("Generate"):
    with st.spinner("Generating..."):
        st.write_stream(stream_llm_response(user_prompt))
```

```python
# Method 3: Streaming in chat interface
if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Stream the response and capture it
        response = st.write_stream(stream_llm_response(prompt))
    
    # st.write_stream returns the complete text when done
    st.session_state.messages.append({"role": "assistant", "content": response})
```

**Streaming Tips:**
- `st.write_stream()` accepts any generator/iterator that yields strings
- The function returns the complete concatenated response when streaming finishes
- Add small delays (`time.sleep(0.01)`) in custom generators for smoother display
- Use `stream=True` parameter when available (Cortex, OpenAI, Anthropic APIs)

### Status and Progress Feedback

```python
# Spinner for quick operations
with st.spinner("Loading..."):
    result = quick_operation()

# Status for multi-step operations
with st.status("Processing...", expanded=True) as status:
    st.write(":material/psychology: Step 1: Analyzing...")
    result1 = step1()
    
    st.write(":material/flash_on: Step 2: Processing...")
    result2 = step2()
    
    st.write(":material/check_circle: Complete!")
    status.update(label="Success!", state="complete", expanded=False)

# Progress bar for loops
progress = st.progress(0)
status_text = st.empty()

for i, item in enumerate(items):
    status_text.text(f"Processing {i+1}/{len(items)}")
    process(item)
    progress.progress((i + 1) / len(items))

progress.empty()
status_text.empty()
```

### Forms for Batch Input

```python
with st.form("settings_form"):
    st.subheader(":material/settings: Configuration")
    
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120)
    option = st.selectbox("Option", ["A", "B", "C"])
    
    submitted = st.form_submit_button("Save", use_container_width=True)
    
    if submitted:
        st.success(f"Saved: {name}, {age}, {option}")
        # Process form data...
```

### File Upload Pattern

```python
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="Upload a CSV file for analysis"
)

if uploaded_file is not None:
    # Store filename in session state
    st.session_state.uploaded_filename = uploaded_file.name
    
    # Load data
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    
    st.success(f"Loaded {len(df)} rows from {uploaded_file.name}")
    st.dataframe(df.head())
```

### Data Export Pattern

```python
from datetime import datetime

# CSV export
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv = df.to_csv(index=False)

st.download_button(
    label=":material/download: Download CSV",
    data=csv,
    file_name=f"export_{timestamp}.csv",
    mime="text/csv",
    use_container_width=True
)

# Excel export with multiple sheets
import io
output = io.BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    df_main.to_excel(writer, sheet_name='Data', index=False)
    df_stats.to_excel(writer, sheet_name='Statistics', index=False)

st.download_button(
    label=":material/download: Download Excel",
    data=output.getvalue(),
    file_name=f"export_{timestamp}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
```

---

## Multipage App Patterns

### Method 1: pages/ Directory (Simple)

Create files in `pages/` folder - they auto-appear in sidebar. Use numeric prefixes for ordering and descriptive names (avoid emojis in filenames):

```
project/
├── app.py                    # Home page
└── pages/
    ├── 1_dashboard.py
    ├── 2_analytics.py
    └── 3_settings.py
```

### Method 2: st.navigation with Sidebar (Default)

```python
# app.py
import streamlit as st

# Page config (Local & Community Cloud only - omit for SiS)
st.set_page_config(page_title="My App", layout="wide")

# Initialize session state BEFORE navigation
st.session_state.setdefault("df", None)
st.session_state.setdefault("settings", {})

# Define pages with Material icons (not emojis in filenames)
page_home = st.Page("pages/home.py", title="Home", icon=":material/home:")
page_data = st.Page("pages/data.py", title="Data", icon=":material/table_chart:")
page_settings = st.Page("pages/settings.py", title="Settings", icon=":material/settings:")

# Group pages (empty string for top-level, appears without header)
pg = st.navigation({
    "": [page_home],
    "Analysis": [page_data],
    "Config": [page_settings],
})

pg.run()
```

### Method 3: st.navigation with Top Navigation Bar

For a horizontal navigation bar at the top of the page (like BioCurator):

```python
# app.py
import streamlit as st

# Page config (Local & Community Cloud only - omit for SiS)
st.set_page_config(page_title="My App", layout="wide")

# Initialize session state BEFORE navigation
st.session_state.setdefault("df", None)
st.session_state.setdefault("data_processed", False)

# Define pages with Material icons
page_home = st.Page("pages/home.py", title="Home", icon=":material/home:")
page_data = st.Page("pages/data_processing.py", title="Data Processing", icon=":material/table_chart:")
page_viz = st.Page("pages/visualization.py", title="Visualization", icon=":material/bar_chart:")
page_about = st.Page("pages/about.py", title="About", icon=":material/info:")

# Tools section with child pages (appears as dropdown)
tools_pages = [
    st.Page("pages/tools_export.py", title="Export Data", icon=":material/download:"),
    st.Page("pages/tools_settings.py", title="Settings", icon=":material/settings:"),
]

# Navigation with position="top" for horizontal nav bar
pg = st.navigation(
    {
        "": [page_home, page_data, page_viz, page_about],  # Top-level links (no header)
        "Tools": tools_pages,                               # Dropdown menu
    },
    position="top"  # Horizontal navigation bar at top
)

pg.run()
```

**Top Navigation Tips:**
- Use `position="top"` for horizontal navigation bar
- Pages in `""` (empty string) section appear as direct top-level links
- Named sections (e.g., "Tools") appear as dropdown menus
- Keep top-level pages to 5-7 for best UX
- Use descriptive `icon` parameter with Material icons

**Important:** Initialize all session state in the main `app.py` file before `st.navigation()` to ensure data persists across pages.

---

## Visualization Patterns

### Plotly Charts

```python
import plotly.express as px
import plotly.graph_objects as go

# Basic chart
fig = px.bar(df, x="category", y="value", color="group")
fig.update_layout(
    margin=dict(t=20, l=0, r=0, b=0),  # Minimal margins
    showlegend=True
)
st.plotly_chart(fig, use_container_width=True)

# Interactive scatter with hover
fig = px.scatter(
    df, x="x", y="y", color="category",
    hover_data=["additional_info"],
    title="Scatter Plot"
)
st.plotly_chart(fig, use_container_width=True)

# Custom color palette
colors = ["#F8766D", "#7CAE00", "#00BFC4", "#C77CFF"]
fig = px.bar(df, x="x", y="y", color="category",
             color_discrete_sequence=colors)
```

### Altair Charts

```python
import altair as alt

# Basic chart with encoding
chart = alt.Chart(df).mark_bar().encode(
    x=alt.X('category:N', title='Category'),
    y=alt.Y('value:Q', title='Value'),
    color=alt.Color('group:N')
).properties(
    width=400,
    height=300,
    title='Bar Chart'
)

st.altair_chart(chart, use_container_width=True)

# Conditional coloring
chart = alt.Chart(df).mark_bar().encode(
    x='x:Q',
    y='count():Q',
    color=alt.condition(
        alt.datum.x > threshold,
        alt.value('#FF9F40'),  # Above threshold
        alt.value('#66B2FF')   # Below threshold
    )
)
```

### DataFrames with Styling

```python
st.dataframe(
    df,
    column_config={
        "progress": st.column_config.ProgressColumn(
            "Progress",
            format="%.1f%%",
            min_value=0,
            max_value=100
        ),
        "url": st.column_config.LinkColumn("Link"),
        "value": st.column_config.NumberColumn(
            "Value",
            format="$%.2f"
        )
    },
    height=400,
    use_container_width=True
)
```

---

## Environment-Specific Patterns

### Database Connection (Multi-Environment)

```python
# Pattern that works across Local, Community Cloud, and SiS
def get_connection():
    """Get database connection based on environment."""
    try:
        # Streamlit in Snowflake
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except:
        # Local or Community Cloud
        from snowflake.snowpark import Session
        return Session.builder.configs(
            st.secrets["connections"]["snowflake"]
        ).create()

# Cache the connection
@st.cache_resource
def get_session():
    return get_connection()

session = get_session()
```

### Environment Detection

```python
# Detect Streamlit in Snowflake
IS_SIS = False
try:
    import _snowflake
    IS_SIS = True
except ImportError:
    IS_SIS = False

# Use environment-specific features
if IS_SIS:
    # SiS-only features (e.g., Cortex Agents)
    resp = _snowflake.send_snow_api_request(...)
else:
    # Local/Community Cloud fallback
    resp = requests.post(url, json=payload)
```

### Secrets Management

**.streamlit/secrets.toml** (never commit!)
```toml
# API Keys
OPENAI_API_KEY = "sk-..."

# Database connections
[connections.snowflake]
account = "your_account"
user = "your_user"
password = "your_password"
role = "your_role"
warehouse = "COMPUTE_WH"
database = "MY_DB"
schema = "MY_SCHEMA"

# Custom settings
[app]
environment = "development"
debug = true
```

**Usage:**
```python
# Access secrets
api_key = st.secrets["OPENAI_API_KEY"]
db_config = st.secrets["connections"]["snowflake"]
debug = st.secrets["app"]["debug"]
```

---

## Error Handling

### User-Friendly Errors

```python
try:
    result = risky_operation()
    st.success("Operation completed!", icon=":material/check_circle:")
except ValueError as e:
    st.error(f"Invalid input: {e}", icon=":material/error:")
    st.stop()
except ConnectionError as e:
    st.warning("Connection failed. Please try again.", icon=":material/warning:")
except Exception as e:
    st.error(f"Unexpected error: {e}", icon=":material/error:")
    st.stop()
```

### Data Validation

```python
def validate_data(df):
    """Validate uploaded data before processing."""
    errors = []
    warnings = []
    
    # Check minimum size
    if len(df) < 10:
        errors.append(f"Dataset too small ({len(df)} rows). Minimum 10 required.")
    elif len(df) < 100:
        warnings.append(f"Small dataset ({len(df)} rows). Results may be less reliable.")
    
    # Check required columns
    required = ['id', 'value', 'category']
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    
    # Display errors
    if errors:
        for error in errors:
            st.error(f":material/error: {error}")
        st.stop()
    
    if warnings:
        for warning in warnings:
            st.warning(f":material/warning: {warning}")
    
    return True
```

### API Error Handling

```python
import requests

def call_api(endpoint: str, payload: dict) -> dict:
    """Call external API with proper error handling."""
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=30  # Always set timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
        return None
```

---

## Material Icons Reference

Use Material icons with `:material/icon_name:` syntax:

### Common Icons
| Category | Icons |
|----------|-------|
| **Navigation** | `home`, `arrow_back`, `menu`, `settings`, `search` |
| **Actions** | `send`, `play_arrow`, `refresh`, `download`, `upload`, `save`, `delete`, `edit` |
| **Status** | `check_circle`, `error`, `warning`, `info`, `pending` |
| **Data** | `table_chart`, `bar_chart`, `analytics`, `query_stats`, `database` |
| **Content** | `chat`, `code`, `description`, `article`, `folder` |
| **UI** | `visibility`, `build`, `tune`, `filter_list` |

### Usage Examples
```python
st.title(":material/rocket: My App")
st.header(":material/settings: Configuration")
st.button(":material/send: Submit", type="primary")
st.success("Done!", icon=":material/check_circle:")
st.error("Failed!", icon=":material/error:")
st.info("Note:", icon=":material/info:")
```

---

## Common Pitfalls & Solutions

### 1. Duplicate Widget Keys
**Problem:** `DuplicateWidgetID` error
```python
# ❌ Bad
st.text_input("Name:")
st.text_input("Name:")

# ✅ Good
st.text_input("First name:", key="first_name")
st.text_input("Last name:", key="last_name")
```

### 2. Page Config Not First (Local & Community Cloud only)
**Problem:** `StreamlitAPIException`
```python
# ❌ Bad
import streamlit as st
st.write("Hello")
st.set_page_config(...)

# ✅ Good (Local & Community Cloud)
import streamlit as st
st.set_page_config(...)
st.write("Hello")

# ✅ Good (Streamlit in Snowflake - omit page config entirely)
import streamlit as st
st.write("Hello")  # No st.set_page_config() needed
```

**Note:** `st.set_page_config()` is not supported in Streamlit in Snowflake (SiS). Omit it entirely for SiS apps.

### 3. Caching Mutable Objects
**Problem:** Cached data changes unexpectedly
```python
# ❌ Bad - dict gets modified
@st.cache_resource
def get_data():
    return {"count": 0}

# ✅ Good - use cache_data for data
@st.cache_data
def get_data():
    return {"count": 0}
```

### 4. Data Lost Between Pages
**Problem:** Session state not persisting
```python
# ❌ Bad: Initialize in each page
# pages/data.py
if 'df' not in st.session_state:
    st.session_state.df = None

# ✅ Good: Initialize in main app before navigation
# app.py
st.session_state.setdefault("df", None)
pg = st.navigation(...)
pg.run()
```

### 5. Slow Reruns
**Problem:** App reruns slowly
```python
# ❌ Bad: Expensive operation on every rerun
df = load_large_dataset()

# ✅ Good: Cache expensive operations
@st.cache_data
def load_large_dataset():
    return pd.read_csv("large_file.csv")

df = load_large_dataset()
```

### 6. Widget State Reset
**Problem:** Widget values reset unexpectedly
```python
# ❌ Bad: Using return value directly
value = st.slider("Value", 0, 100)
# value resets on any other widget change

# ✅ Good: Use session state for persistence
if "slider_value" not in st.session_state:
    st.session_state.slider_value = 50

value = st.slider(
    "Value", 0, 100,
    value=st.session_state.slider_value,
    key="slider_value"
)
```

---

## Deployment

### Local Development

```bash
streamlit run app.py --server.port 8501 --server.headless false
```

### Streamlit Community Cloud

1. Push code to GitHub (public or private repo)
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Add secrets via dashboard (Settings → Secrets)
4. Deploy

**requirements.txt example:**
```
streamlit>=1.52.0
pandas>=2.0.0
plotly>=5.14.0
numpy>=1.24.0
```

### Streamlit in Snowflake (SiS)

1. Create Streamlit object in Snowflake
2. Upload app files to stage
3. Configure secrets in Snowflake
4. Share with users via URL

**SiS-specific notes:**
- **Do not use `st.set_page_config()`** - not supported in SiS
- Use `from snowflake.snowpark.context import get_active_session`
- `_snowflake` module available for internal APIs
- Secrets managed via Snowflake secrets

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Configuration (.streamlit/config.toml)

```toml
[theme]
primaryColor = "#F8766D"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
port = 8501
```

---

## Security Best Practices

1. **Never commit secrets**
   - Add `.streamlit/secrets.toml` to `.gitignore`
   - Use environment variables or platform secrets management

2. **Validate user inputs**
   ```python
   if user_input:
       cleaned = user_input.strip()[:500]  # Limit length
       # Use parameterized queries for databases
       session.sql("SELECT * FROM t WHERE id = ?", params=[cleaned])
   ```

3. **Handle API errors gracefully**
   ```python
   try:
       result = call_api(prompt)
   except Exception as e:
       st.error(f"API Error: {str(e)}")
       st.stop()
   ```

4. **Limit file uploads**
   - Set `maxUploadSize` in config.toml
   - Validate file types before processing

5. **Use role-based access in Snowflake**
   - Create specific roles for app users
   - Grant minimum required privileges

---

## Testing Checklist

Before deploying, verify:

- [ ] All widgets have unique `key` parameters
- [ ] Session state persists across page navigation
- [ ] `st.set_page_config()` is first Streamlit command (Local/Community Cloud only, omit for SiS)
- [ ] Caching is used for expensive operations
- [ ] Error handling covers edge cases
- [ ] App works with empty/minimal data
- [ ] App handles large datasets gracefully
- [ ] All imports are in requirements.txt
- [ ] Secrets are not committed to repo
- [ ] UI is responsive (use `use_container_width=True`)

---

## Resources

### Official Documentation
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit API Reference](https://docs.streamlit.io/develop/api-reference)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Community Forum](https://discuss.streamlit.io)

### Visualization Libraries
- [Plotly Express](https://plotly.com/python/plotly-express/)
- [Altair](https://altair-viz.github.io/)

### Material Icons
- [Google Material Symbols](https://fonts.google.com/icons)

### Deployment
- [Streamlit Community Cloud](https://share.streamlit.io)
- [Streamlit in Snowflake Docs](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit)

---

## About This File

This AGENTS.md is built from real-world experience:

- **30 Days of AI Challenge** — A 30-day learning series teaching AI app development with Streamlit and Snowflake Cortex. Each day contributed battle-tested patterns for chatbots, RAG systems, streaming responses, and agent orchestration.

- **Years of Streamlit Development** — Patterns refined from building dozens of Streamlit applications over 2024-2025, including data dashboards, cheminformatics tools, ML apps, and enterprise applications deployed to Streamlit in Snowflake.

Every pattern in this file has been validated in production.

---

**Last Updated:** January 2026
**Format:** [AGENTS.md Standard](https://agents.md)

