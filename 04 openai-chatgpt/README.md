# ChatGPT Clone with OpenAI & Streamlit

Build your own ChatGPT-like conversational AI interface using OpenAI's API and Streamlit. This project demonstrates how to create a chat application with persistent conversation history and streaming responses.

## What You'll Learn

- Building conversational AI interfaces
- Managing chat history and session state
- Implementing streaming responses for real-time interaction
- Working with OpenAI's chat completion API
- Creating ChatGPT-like user experiences

## Project Overview

This clone replicates the core functionality of ChatGPT:

- **Real-time conversations** with AI
- **Message history** that persists during the session
- **Streaming responses** for better user experience
- **Clean chat interface** similar to ChatGPT

## Technical Stack

- **OpenAI API**: GPT-3.5-turbo for chat completions
- **Streamlit**: Web framework with built-in chat components
- **Python**: Core programming language
- **Session State**: For maintaining conversation history

## Tutorial Steps

### Step 1: Environment Setup

```bash
# Create conda environment
conda create -n env_langchain1 python=3.10
conda activate env_langchain1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get OpenAI API Key

1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Create a `.env` file and add: `OPENAI_API_KEY=your_api_key_here`

### Step 3: Understanding the Architecture

The application follows this flow:

```python
# 1. Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 2. Manage session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle new user input
if prompt := st.chat_input("What is up?"):
    # Add to history and get AI response
    response = client.chat.completions.create(...)
```

## Running the Application

```bash
streamlit run chatgpt_like_app.py
```

Open http://localhost:8501 and start chatting!

## Key Features Explained

### 1. Session State Management

Streamlit's session state maintains conversation history:

```python
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add messages to history
st.session_state.messages.append({"role": "user", "content": prompt})
```

### 2. Streaming Responses

Real-time response streaming for better UX:

```python
stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=conversation_history,
    stream=True,  # Enable streaming
)
response = st.write_stream(stream)  # Display as it generates
```

### 3. Chat Interface Components

Streamlit's native chat components:

- `st.chat_message()`: Display messages with role-based styling
- `st.chat_input()`: Input field optimized for chat
- `st.write_stream()`: Real-time streaming display

### 4. Conversation Context

The app maintains full conversation context:

```python
messages=[
    {"role": m["role"], "content": m["content"]}
    for m in st.session_state.messages  # Send entire history
]
```

## Learning Objectives

By completing this project, you will:

- Understand OpenAI's chat completion API
- Master Streamlit's session state management
- Implement streaming responses for better UX
- Create conversational AI interfaces
- Handle persistent chat history

## Customization Options

### Model Selection

Change the AI model:

```python
st.session_state["openai_model"] = "gpt-4"  # or "gpt-3.5-turbo"
```

### System Messages

Add personality to your AI:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    *conversation_history
]
```

### UI Enhancements

- Add chat export functionality
- Implement conversation clearing
- Add typing indicators
- Include message timestamps

## Advanced Features to Add

1. **Conversation Export**: Save chat history to files
2. **Multiple Conversations**: Manage different chat sessions
3. **Custom System Prompts**: Define AI personality
4. **Message Rating**: Thumbs up/down for responses
5. **Token Counting**: Track API usage

## Video Tutorial

[Watch the complete tutorial](https://youtu.be/z-moiQlcC6c)

## Troubleshooting

**Common Issues:**

- Rate limiting: Implement retry logic or reduce request frequency
- API key errors: Verify key is valid and has sufficient credits
- Memory issues: Clear session state periodically for long conversations
- Streaming problems: Check network connection and API status

## Security Best Practices

- Never commit API keys to version control
- Use environment variables for sensitive data
- Implement rate limiting for production use
- Validate and sanitize user inputs
