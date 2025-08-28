# First GenAI Applications with LangChain & Gemini

This folder contains your first hands-on experience with Generative AI applications using LangChain and Google's Gemini model. You'll build two fundamental AI applications: a Q&A chatbot and a language translator.

## What You'll Learn

- Setting up LangChain with Google Gemini
- Creating prompt templates for AI interactions
- Building Streamlit interfaces for AI applications
- Implementing chat-based Q&A systems
- Creating language translation services

## Applications Included

### 1. AI Q&A Chatbot (`gemini_app_qa.py`)

A simple yet powerful chatbot that can answer questions on any topic using Google's Gemini model.

**Features:**

- Natural language question answering
- Real-time responses
- Clean Streamlit interface
- Context-aware conversations

### 2. Language Translator (`gemini_applanguage_translator.py`)

An intelligent language translator that converts English text to German using AI.

**Features:**

- English to German translation
- AI-powered contextual translation
- Interactive web interface
- Extensible to other language pairs

## Technical Stack

- **LangChain**: Framework for building AI applications
- **Google Gemini**: Large language model for text generation
- **Streamlit**: Web framework for creating interactive apps
- **Python**: Programming language
- **Environment Management**: Virtual environments and dotenv

## Tutorial Steps

### Step 1: Environment Setup

Create a Python virtual environment and install dependencies:

```bash
# Create virtual environment
py -3.10 -m venv myvenv

# Activate environment
myvenv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install --upgrade --quiet langchain-google-genai pillow
pip install streamlit
pip install python-dotenv
```

### Step 2: Get Google API Key

1. Visit [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
2. Generate your Google AI API key
3. Create a `.env` file in this directory
4. Add your API key: `GOOGLE_API_KEY=your_api_key_here`

### Step 3: Understanding the Code Structure

Both applications follow the same LangChain pattern:

```python
# 1. Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 2. Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# 3. Create prompt template
prompt = ChatPromptTemplate.from_messages([...])

# 4. Build the chain
chain = prompt | llm | output_parser

# 5. Process user input
response = chain.invoke(user_input)
```

## Running the Applications

### Q&A Chatbot

```bash
streamlit run gemini_app_qa.py
```

Open http://localhost:8501 and start asking questions!

### Language Translator

```bash
streamlit run gemini_applanguage_translator.py
```

Enter English text and get German translations instantly!

## Key Concepts Explained

### LangChain Components

1. **ChatPromptTemplate**: Structures how we communicate with the AI
2. **ChatGoogleGenerativeAI**: The actual Gemini model interface
3. **StrOutputParser**: Converts AI responses to readable text
4. **Chain**: Connects all components in a pipeline

### Prompt Engineering

The applications demonstrate two types of prompts:

- **System prompts**: Define the AI's role and behavior
- **Human prompts**: Contain user input and specific instructions

## Learning Objectives

By completing this tutorial, you will:

- Understand LangChain's basic architecture
- Know how to integrate Google Gemini with Python
- Create interactive AI applications with Streamlit
- Implement prompt templates for different use cases
- Handle API keys and environment variables securely

## Next Steps

After mastering these basics, you can:

- Experiment with different Gemini models
- Add conversation memory
- Implement more complex prompt templates
- Create multi-turn conversations
- Add error handling and validation

## Video Tutorial

[Watch the complete tutorial](https://youtu.be/VvpuGpXOYrQ)

## Troubleshooting

**Common Issues:**

- API key not found: Ensure `.env` file is in the correct directory
- Module import errors: Check if all packages are installed in the active environment
- Streamlit not starting: Verify Streamlit installation and port availability
