# RAG (Retrieval-Augmented Generation) Demo

Build an intelligent document retrieval and question-answering system using RAG architecture. This project demonstrates how to create AI applications that can understand and answer questions about specific documents or web content with high accuracy.

## What You'll Learn

- Understanding RAG (Retrieval-Augmented Generation) architecture
- Document loading and text chunking strategies
- Vector embeddings and similarity search
- Building retrieval-based Q&A systems
- Combining document retrieval with language generation

## Project Overview

This RAG system provides:

- **Document Intelligence**: Load and process web content automatically
- **Semantic Search**: Find relevant information using AI embeddings
- **Contextual Answers**: Generate accurate responses based on retrieved content
- **Source Attribution**: Know exactly where answers come from
- **Scalable Architecture**: Handle multiple documents efficiently

## Technical Stack

- **LangChain**: RAG framework and document processing
- **OpenAI**: GPT models for generation and embeddings
- **Chroma**: Vector database for similarity search
- **UnstructuredURLLoader**: Web content extraction
- **Streamlit**: Interactive web interface
- **RecursiveCharacterTextSplitter**: Intelligent text chunking

## Tutorial Steps

### Step 1: Environment Setup

```bash
# Create conda environment
conda create -n env_langchain1 python=3.10
conda activate env_langchain1

# Update pip and install packages
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: API Configuration

1. Get OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 3: Understanding RAG Architecture

The RAG pipeline follows these steps:

```python
# 1. Document Loading
loader = UnstructuredURLLoader(urls=['url1', 'url2'])
documents = loader.load()

# 2. Text Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = text_splitter.split_documents(documents)

# 3. Vector Store Creation
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)

# 4. Retrieval Setup
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# 5. Generation Chain
rag_chain = create_retrieval_chain(retriever, qa_chain)
```

## Running the Application

```bash
streamlit run app1.py
```

Visit http://localhost:8501 and start asking questions about your documents!

## RAG Components Explained

### 1. Document Loading

```python
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
```

The loader automatically:

- Fetches content from web URLs
- Extracts text from various formats
- Handles different content types
- Preserves document structure

### 2. Text Chunking

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)
```

Chunking strategies:

- **Fixed Size**: Consistent chunk lengths
- **Semantic Splitting**: Preserve meaning boundaries
- **Overlap**: Maintain context between chunks
- **Recursive**: Multiple splitting strategies

### 3. Vector Embeddings

```python
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings()
)
```

Embeddings enable:

- **Semantic Search**: Find meaning, not just keywords
- **Similarity Matching**: Identify related content
- **Efficient Retrieval**: Fast vector operations
- **Multilingual Support**: Work across languages

### 4. Retrieval Configuration

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)
```

Retrieval options:

- **Similarity Search**: Most semantically similar chunks
- **MMR (Maximum Marginal Relevance)**: Diverse relevant results
- **Threshold Filtering**: Minimum similarity scores
- **Metadata Filtering**: Filter by document properties

### 5. Generation Chain

```python
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

## Learning Objectives

By completing this project, you will:

- Master RAG architecture and components
- Understand vector embeddings and similarity search
- Learn document processing and chunking strategies
- Build retrieval-based Q&A systems
- Implement context-aware AI responses

## Advanced Configuration

### Custom Chunking Strategy

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

### Enhanced Retrieval

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)
```

### Custom Prompts

```python
custom_prompt = """You are a research assistant.
Based on the following context, provide a detailed analysis:

Context: {context}

Question: {input}

Analysis:"""
```

## Enhancement Ideas

1. **Multi-Format Support**: PDFs, Word docs, spreadsheets
2. **Real-Time Updates**: Refresh document index automatically
3. **Source Tracking**: Show which documents provided answers
4. **Confidence Scoring**: Rate answer reliability
5. **Conversation Memory**: Remember previous Q&A context

## Use Cases

- **Customer Support**: Answer questions from documentation
- **Research Assistant**: Query large document collections
- **Legal Analysis**: Search through legal documents
- **Technical Documentation**: Interactive API/code documentation
- **Educational Tools**: Create study aids from course materials

## Performance Optimization

### Chunking Optimization

- **Overlap Strategy**: Balance context preservation and efficiency
- **Semantic Boundaries**: Split at natural language breaks
- **Size Tuning**: Adjust chunk size based on content type

### Retrieval Tuning

- **K Parameter**: Number of chunks to retrieve
- **Similarity Threshold**: Filter low-quality matches
- **Re-ranking**: Post-process retrieval results

### Cost Management

- **Embedding Caching**: Store embeddings to reduce API calls
- **Batch Processing**: Group operations efficiently
- **Model Selection**: Balance quality and cost

## Video Tutorial

[Watch the complete tutorial](https://youtu.be/iA-UhFlIP80)

## Troubleshooting

**Common Issues:**

1. **URL Loading Errors**

   - Check URL accessibility and format
   - Verify network connectivity
   - Handle rate limiting and timeouts

2. **Embedding Generation Fails**

   - Verify OpenAI API key validity
   - Check API usage limits
   - Monitor embedding costs

3. **Poor Retrieval Quality**

   - Adjust chunk size and overlap
   - Tune retrieval parameters (k value)
   - Improve source document quality

4. **Memory Issues**
   - Reduce chunk size for large documents
   - Implement batch processing
   - Use smaller embedding models

## Best Practices

- **API Key Security**: Use environment variables
- **Error Handling**: Implement graceful failure recovery
- **Resource Management**: Monitor API usage and costs
- **Quality Control**: Validate source documents
- **User Experience**: Provide clear feedback and citations
