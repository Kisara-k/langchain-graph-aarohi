# Gemini Starter - Basic Q&A Chatbot

Build your first AI-powered question and answer application using Google's Gemini model and LangChain. This starter project demonstrates the fundamental concepts of creating conversational AI applications with modern frameworks.

## What You'll Learn

- Setting up Google Gemini with LangChain
- Creating basic prompt templates for AI interactions
- Building interactive web interfaces with Streamlit
- Understanding the LangChain pipeline architecture
- Implementing simple conversational AI patterns

## Project Overview

This starter application provides:

- **Simple Q&A Interface**: Ask any question and get AI-powered answers
- **Real-time Responses**: Instant AI responses through Streamlit
- **Clean Architecture**: Well-structured LangChain pipeline
- **Easy Deployment**: Ready-to-run Streamlit application
- **Beginner-Friendly**: Perfect introduction to AI app development

## Technical Stack

- **Google Gemini 1.5 Pro**: Advanced language model for generation
- **LangChain**: Framework for building AI applications
- **Streamlit**: Web framework for interactive interfaces
- **Python**: Core programming language
- **dotenv**: Environment variable management

## Quick Start Tutorial

### Step 1: Environment Setup

```bash
# Install required packages
pip install streamlit
pip install langchain-google-genai
pip install python-dotenv
pip install langchain-core
```

### Step 2: API Key Configuration

1. Get your Google AI API key from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
2. Create a `.env` file in this directory
3. Add your API key:

```env
GOOGLE_API_KEY='your_google_api_key_here'
LANGCHAIN_API_KEY='your_langchain_api_key_here'  # Optional for tracking
LANGCHAIN_PROJECT='geminiChatbottutorial'         # Optional for tracking
```

### Step 3: Understanding the Application

The application follows a simple but powerful pattern:

```python
# 1. Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,  # Deterministic responses
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# 2. Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot"),
    ("human", "Question:{question}")
])

# 3. Create processing chain
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# 4. Process user input
response = chain.invoke({'question': user_input})
```

## Running the Application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser and start asking questions!

## Code Architecture Explained

### 1. Model Configuration

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,        # Controls randomness (0 = deterministic)
    max_tokens=None,      # No token limit
    timeout=None,         # No timeout limit
    max_retries=2,        # Retry failed requests twice
)
```

**Key Parameters:**

- **temperature=0**: Ensures consistent, focused responses
- **max_retries=2**: Handles temporary API issues gracefully
- **gemini-1.5-pro**: Uses Google's most capable model

### 2. Prompt Engineering

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot"),
    ("human", "Question:{question}")
])
```

**Prompt Structure:**

- **System Message**: Defines the AI's role and behavior
- **Human Message**: Contains the user's question with template variable
- **Template Variable**: `{question}` is replaced with actual user input

### 3. LangChain Pipeline

```python
chain = prompt | llm | output_parser
```

**Pipeline Flow:**

1. **Prompt**: Formats user input into proper message structure
2. **LLM**: Processes the prompt and generates AI response
3. **Output Parser**: Converts AI response to clean string format

### 4. Streamlit Interface

```python
st.title('Langchain Demo With Gemini')
input_text = st.text_input("Enter your question here")

if input_text:
    st.write(chain.invoke({'question': input_text}))
```

**Interface Components:**

- **Title**: Clear application branding
- **Text Input**: User question entry field
- **Dynamic Response**: Real-time AI response display

## Learning Objectives

By completing this project, you will:

- Understand basic LangChain architecture
- Learn to integrate Google Gemini models
- Master simple prompt engineering techniques
- Build interactive AI applications with Streamlit
- Handle API keys and environment configuration

## Customization Ideas

### 1. Enhanced System Prompts

```python
# Specialized chatbot personalities
prompts = {
    "teacher": "You are a helpful teacher who explains concepts clearly",
    "programmer": "You are an expert programmer who provides code solutions",
    "writer": "You are a creative writer who helps with storytelling",
}
```

### 2. Response Formatting

```python
# Add structured response formatting
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful chatbot. Format your responses as:

    **Answer:** [Your main answer]
    **Explanation:** [Brief explanation]
    **Additional Info:** [Any extra helpful details]
    """),
    ("human", "Question:{question}")
])
```

### 3. Input Validation

```python
# Add input validation and sanitization
def validate_input(text):
    if not text.strip():
        return False, "Please enter a question"
    if len(text) > 500:
        return False, "Question too long (max 500 characters)"
    return True, ""

if input_text:
    is_valid, error_msg = validate_input(input_text)
    if is_valid:
        response = chain.invoke({'question': input_text})
        st.write(response)
    else:
        st.error(error_msg)
```

### 4. Response Caching

```python
# Cache responses for repeated questions
import streamlit as st
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(question):
    return chain.invoke({'question': question})

if input_text:
    response = get_cached_response(input_text)
    st.write(response)
```

## Example Interactions

### General Knowledge

```
Q: What is artificial intelligence?
A: Artificial intelligence (AI) refers to the simulation of human intelligence in machines...
```

### Problem Solving

```
Q: How do I learn programming?
A: Learning programming involves several key steps: 1) Choose a language, 2) Practice regularly...
```

### Creative Tasks

```
Q: Write a short poem about technology
A: In circuits bright and data streams,
Where silicon dreams come alive...
```

## Next Steps

After mastering this basic chatbot:

1. **Add Memory**: Implement conversation history
2. **Multiple Models**: Compare different AI models
3. **Advanced Prompts**: Learn complex prompt engineering
4. **Tool Integration**: Add external tool capabilities
5. **Deployment**: Deploy to cloud platforms

## Performance Optimization

### Response Time Optimization

```python
# Optimize for faster responses
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Faster model variant
    temperature=0.1,
    max_tokens=500,            # Limit response length
)
```

### Error Handling

```python
# Robust error handling
try:
    response = chain.invoke({'question': input_text})
    st.write(response)
except Exception as e:
    st.error(f"Sorry, I encountered an error: {str(e)}")
    st.info("Please try rephrasing your question or check your internet connection.")
```

## Troubleshooting

**Common Issues:**

1. **API Key Errors**

   - Verify your Google API key is correct
   - Check that the API key has proper permissions
   - Ensure the `.env` file is in the correct directory

2. **Import Errors**

   - Install all required packages: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

3. **Streamlit Issues**

   - Clear Streamlit cache: `streamlit cache clear`
   - Restart the application
   - Check port availability (default: 8501)

4. **Slow Responses**
   - Check internet connection
   - Consider using gemini-1.5-flash for faster responses
   - Implement response caching

## Security Best Practices

- Never commit API keys to version control
- Use environment variables for sensitive data
- Implement rate limiting for production use
- Validate and sanitize all user inputs
- Monitor API usage and costs

## Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

This starter project provides the foundation for building more complex AI applications. Start here and gradually add features as you learn!
