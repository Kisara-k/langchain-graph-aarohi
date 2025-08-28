# LangChain Agents with Web Search Tools

Build intelligent AI agents that can search the web, gather information, and provide comprehensive answers using multiple tools. This project demonstrates how to create autonomous AI assistants with access to real-time web information.

## What You'll Learn

- Creating intelligent agents with LangChain
- Integrating web search capabilities with SerpAPI
- Building tool-equipped AI assistants
- Agent reasoning and decision-making processes
- Multi-step problem solving with AI

## Project Overview

This system creates AI agents that can:

- **Web Search**: Access real-time information from the internet
- **Tool Selection**: Choose appropriate tools for different tasks
- **Multi-Step Reasoning**: Break down complex queries into steps
- **Information Synthesis**: Combine data from multiple sources
- **Autonomous Operation**: Work independently to find answers

## Technical Stack

- **LangChain Agents**: Framework for building autonomous AI systems
- **OpenAI GPT**: Language model for reasoning and generation
- **SerpAPI**: Google search integration for web queries
- **LangChain Tools**: Pre-built tools for various tasks
- **Python**: Core programming language

## Tutorial Steps

### Step 1: Install Required Packages

```bash
# Install core packages
pip install --upgrade langchain langchain-community google-search-results

# Install OpenAI integration
pip install openai

# Install environment management
pip install python-dotenv
```

### Step 2: API Keys Setup

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

#### Getting API Keys:

1. **OpenAI API Key**:

   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create account and generate API key

2. **SerpAPI Key**:
   - Visit [SerpAPI](https://serpapi.com/)
   - Sign up for free account
   - Get your API key from dashboard

### Step 3: Understanding Agent Architecture

Agents follow this decision-making process:

```python
# 1. Agent receives a question
user_question = "What's the latest news about AI?"

# 2. Agent analyzes what tools it needs
# "I need to search the web for recent AI news"

# 3. Agent selects and uses appropriate tool
search_tool.run("latest AI news 2024")

# 4. Agent processes the results
# Analyze search results and extract key information

# 5. Agent provides comprehensive answer
# Synthesize information into coherent response
```

## Core Components Explained

### 1. Agent Types

#### Zero-Shot React Agent

```python
from langchain.agents import create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate

# Creates agents that can reason about tool usage
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)
```

#### Conversational Agent

```python
from langchain.agents import create_conversational_retrieval_agent

# Maintains conversation history
agent = create_conversational_retrieval_agent(
    llm=llm,
    tools=tools,
    memory=conversation_memory
)
```

### 2. Tool Integration

#### Web Search Tool

```python
from langchain_community.tools import SerpAPIWrapper

# Configure search tool
search = SerpAPIWrapper(
    serpapi_api_key="your_key",
    params={
        "engine": "google",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }
)

tools = [
    Tool(
        name="Search",
        description="Search the web for current information",
        func=search.run
    )
]
```

#### Custom Tools

```python
from langchain.tools import Tool

def calculator(expression):
    """Calculate mathematical expressions"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

calc_tool = Tool(
    name="Calculator",
    description="Perform mathematical calculations",
    func=calculator
)
```

### 3. Agent Execution

```python
from langchain.agents import AgentExecutor

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Show reasoning process
    handle_parsing_errors=True
)

# Run agent
response = agent_executor.invoke({
    "input": "What's the current weather in New York and what's 15% of 1000?"
})
```

## Learning Objectives

By completing this project, you will:

- Understand AI agent architecture and reasoning
- Learn to integrate multiple tools with LangChain
- Master web search integration for AI applications
- Build autonomous problem-solving AI systems
- Implement multi-step reasoning workflows

## Advanced Agent Patterns

### 1. Multi-Tool Workflows

```python
tools = [
    SerpAPIWrapper(),  # Web search
    WikipediaAPIWrapper(),  # Wikipedia search
    LLMMathChain.from_llm(llm),  # Math calculations
    PythonREPLTool(),  # Code execution
]

# Agent can combine multiple tools for complex tasks
```

### 2. Custom Reasoning Prompts

```python
system_message = """You are a research assistant with access to multiple tools.

When answering questions:
1. Break down complex queries into steps
2. Use appropriate tools for each step
3. Synthesize information from multiple sources
4. Provide comprehensive, well-sourced answers

Available tools: {tool_names}
Tool descriptions: {tools}

Question: {input}
Thought process: {agent_scratchpad}"""
```

### 3. Error Handling and Retries

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # Limit reasoning steps
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=True
)
```

## Use Cases and Examples

### 1. Research Assistant

```python
# Query: "Compare the GDP of USA and China in 2023"
# Agent process:
# 1. Search for "USA GDP 2023"
# 2. Search for "China GDP 2023"
# 3. Compare and analyze the data
# 4. Provide comprehensive comparison
```

### 2. News Analyst

```python
# Query: "What are the main tech trends this week?"
# Agent process:
# 1. Search for "technology trends this week"
# 2. Search for "tech news latest"
# 3. Identify common themes
# 4. Summarize key trends
```

### 3. Problem Solver

```python
# Query: "If I invest $10,000 at 5% annual interest, how much will I have in 10 years, and what companies are currently offering similar returns?"
# Agent process:
# 1. Calculate compound interest
# 2. Search for current investment options
# 3. Compare available returns
# 4. Provide investment advice
```

## Advanced Features

### 1. Memory Integration

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,  # Remember last 5 interactions
    return_messages=True
)
```

### 2. Custom Tool Creation

```python
class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather for a location"

    def _run(self, location: str) -> str:
        # Implement weather API call
        return f"Weather in {location}: Sunny, 75°F"
```

### 3. Parallel Tool Execution

```python
# Execute multiple tools simultaneously
async def parallel_search(queries):
    tasks = [search_tool.arun(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return results
```

## Performance Optimization

### 1. Tool Selection Strategy

```python
# Optimize tool descriptions for better selection
tool_descriptions = {
    "search": "Use for current events, recent information, and real-time data",
    "calculator": "Use for mathematical calculations and numerical analysis",
    "wikipedia": "Use for factual, historical, and encyclopedic information"
}
```

### 2. Response Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query):
    return search_tool.run(query)
```

### 3. Rate Limiting

```python
import time
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=10, period=60)  # 10 calls per minute
def rate_limited_search(query):
    return search_tool.run(query)
```

## Video Tutorial

[Watch the complete tutorial](https://youtu.be/Z_OvWHR8C7M)

## Troubleshooting

**Common Issues:**

1. **Tool Selection Problems**

   - Improve tool descriptions
   - Add more specific examples
   - Adjust agent prompt

2. **API Rate Limits**

   - Implement rate limiting
   - Add retry logic
   - Use multiple API keys

3. **Agent Loops**

   - Set max_iterations limit
   - Improve stopping criteria
   - Add timeout mechanisms

4. **Poor Reasoning Quality**
   - Enhance system prompts
   - Use better examples
   - Adjust temperature settings

## Best Practices

### Security

- Store API keys securely
- Validate tool inputs
- Implement usage monitoring
- Add cost controls

### Performance

- Cache frequent queries
- Optimize tool descriptions
- Monitor API usage
- Implement fallback mechanisms

### Quality

- Test with diverse queries
- Monitor agent reasoning
- Collect user feedback
- Iterate on prompts

## Next Steps

After mastering basic agents, explore:

- Multi-agent systems
- Custom tool development
- Advanced reasoning patterns
- Production deployment strategies
- Agent monitoring and analytics
