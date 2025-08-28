# Basic LangChain Agents Foundation

Learn the fundamental concepts of LangChain agents through hands-on examples. This project provides a solid foundation for understanding how AI agents work, make decisions, and solve problems autonomously.

## What You'll Learn

- Core concepts of AI agents and autonomous systems
- LangChain agent framework fundamentals
- Agent reasoning and decision-making processes
- Building your first intelligent agent
- Understanding agent tools and capabilities

## Project Overview

This foundational project introduces:

- **Agent Basics**: Core concepts and terminology
- **Simple Decision Making**: How agents choose actions
- **Tool Integration**: Basic tool usage patterns
- **Reasoning Loops**: Step-by-step problem solving
- **Error Handling**: Managing agent failures gracefully

## Technical Stack

- **LangChain**: Core agent framework
- **OpenAI**: Language model for agent reasoning
- **Python**: Programming environment
- **Jupyter Notebooks**: Interactive development

## Tutorial Steps

### Step 1: Install Core Packages

```bash
# Install essential LangChain components
pip install --upgrade langchain langchain-community

# Install OpenAI integration
pip install openai

# Install environment management
pip install python-dotenv
```

### Step 2: API Key Setup

Create a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### Step 3: Understanding Agent Fundamentals

An agent follows this basic loop:

```python
# 1. Observe: Agent receives input/question
observation = "What is 25 * 4?"

# 2. Think: Agent reasons about what to do
thought = "I need to perform a mathematical calculation"

# 3. Act: Agent selects and uses a tool
action = "Use calculator tool"

# 4. Observe: Agent sees the result
result = "100"

# 5. Respond: Agent provides final answer
response = "25 * 4 equals 100"
```

## Core Concepts Explained

### 1. What Are AI Agents?

AI Agents are autonomous systems that can:

- **Perceive** their environment (receive inputs)
- **Reason** about what actions to take
- **Act** using available tools
- **Learn** from feedback and results

### 2. Agent Components

#### The Language Model (Brain)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0  # Deterministic for consistent reasoning
)
```

#### Tools (Hands)

```python
from langchain.tools import Tool

def simple_calculator(expression):
    """Calculate basic mathematical expressions"""
    try:
        return str(eval(expression))
    except:
        return "Error: Invalid expression"

calculator_tool = Tool(
    name="Calculator",
    description="Performs basic math calculations",
    func=simple_calculator
)
```

#### Memory (Context)

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

### 3. Basic Agent Creation

```python
from langchain.agents import create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate

# Define agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to solve problems."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent
agent = create_openai_functions_agent(
    llm=llm,
    tools=[calculator_tool],
    prompt=prompt
)
```

## Learning Objectives

By completing this project, you will:

- Understand the basic agent architecture
- Learn how agents make decisions
- Create simple tools for agents to use
- Implement basic reasoning loops
- Handle agent errors and edge cases

## Hands-On Exercises

### Exercise 1: Simple Calculator Agent

Create an agent that can perform basic math:

```python
# Step 1: Define the tool
def math_tool(expression):
    """Safely evaluate mathematical expressions"""
    allowed_chars = set('0123456789+-*/.() ')
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression"

    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Error: Could not calculate"

# Step 2: Create the agent
tools = [Tool(
    name="Calculator",
    description="Calculate mathematical expressions like '2+2' or '10*5'",
    func=math_tool
)]

# Step 3: Test the agent
test_questions = [
    "What is 15 + 27?",
    "Calculate 144 divided by 12",
    "What's 25% of 200?"
]
```

### Exercise 2: Text Processing Agent

Build an agent that can manipulate text:

```python
def text_processor(command_and_text):
    """Process text commands like 'uppercase: hello world'"""
    parts = command_and_text.split(':', 1)
    if len(parts) != 2:
        return "Format: command: text"

    command, text = parts[0].strip().lower(), parts[1].strip()

    if command == "uppercase":
        return text.upper()
    elif command == "lowercase":
        return text.lower()
    elif command == "reverse":
        return text[::-1]
    elif command == "count":
        return f"Character count: {len(text)}"
    else:
        return "Available commands: uppercase, lowercase, reverse, count"

text_tool = Tool(
    name="TextProcessor",
    description="Process text with commands like 'uppercase: hello' or 'count: some text'",
    func=text_processor
)
```

### Exercise 3: Decision-Making Agent

Create an agent that makes simple decisions:

```python
def decision_helper(question):
    """Help make simple yes/no decisions"""
    import random

    # Simple decision logic
    positive_words = ['yes', 'good', 'should', 'want', 'like']
    negative_words = ['no', 'bad', 'shouldn\'t', 'don\'t', 'dislike']

    question_lower = question.lower()

    positive_count = sum(1 for word in positive_words if word in question_lower)
    negative_count = sum(1 for word in negative_words if word in question_lower)

    if positive_count > negative_count:
        return "Based on your question, I'd lean towards YES"
    elif negative_count > positive_count:
        return "Based on your question, I'd lean towards NO"
    else:
        return f"It's a tough call! Maybe consider: {random.choice(['pros and cons', 'asking a friend', 'sleeping on it'])}"

decision_tool = Tool(
    name="DecisionHelper",
    description="Help make simple decisions based on yes/no questions",
    func=decision_helper
)
```

## Common Agent Patterns

### 1. Sequential Reasoning

```python
# Agent breaks down complex problems into steps
# Example: "Calculate the area of a circle with radius 5"
# Step 1: Recall formula (π * r²)
# Step 2: Calculate π * 5²
# Step 3: Return result
```

### 2. Tool Selection

```python
# Agent chooses the right tool for the task
# Math question → Calculator tool
# Text question → Text processor tool
# Decision question → Decision helper tool
```

### 3. Error Recovery

```python
# Agent handles tool failures gracefully
# Try tool → Error → Try alternative approach → Success
```

## Simple Use Cases

### 1. Study Assistant

```python
# "What's 15% of 85?"
# Agent: Use calculator → 0.15 * 85 = 12.75
```

### 2. Text Helper

```python
# "Make this uppercase: hello world"
# Agent: Use text processor → HELLO WORLD
```

### 3. Quick Decisions

```python
# "Should I go for a walk? It's sunny outside"
# Agent: Analyze sentiment → YES, sounds like good weather for a walk
```

## Agent Behavior Analysis

### Understanding Agent Thoughts

Monitor what your agent is thinking:

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Shows agent reasoning
    return_intermediate_steps=True
)

# See the agent's thought process
result = agent_executor.invoke({"input": "What's 12 * 8?"})
print("Agent's reasoning steps:")
for step in result['intermediate_steps']:
    print(f"Thought: {step[0]}")
    print(f"Action: {step[1]}")
```

### Common Reasoning Patterns

1. **Direct Tool Use**: Simple questions → Use tool → Return result
2. **Multi-Step**: Complex questions → Break down → Use multiple tools
3. **Clarification**: Unclear questions → Ask for clarification
4. **Fallback**: Tool fails → Try alternative approach

## Video Tutorial

[Watch the complete tutorial](https://youtu.be/FQyrd26U3MU)

## Troubleshooting

**Common Beginner Issues:**

1. **Agent Doesn't Use Tools**

   - Check tool descriptions are clear
   - Verify tool names are descriptive
   - Ensure prompt encourages tool use

2. **Poor Tool Selection**

   - Improve tool descriptions
   - Add usage examples
   - Simplify tool interfaces

3. **Agent Loops or Gets Stuck**

   - Set max_iterations limit
   - Add stopping conditions
   - Improve error handling

4. **Unclear Responses**
   - Enhance system prompts
   - Add response formatting guidelines
   - Test with simpler questions first

## Next Learning Steps

After mastering these basics:

1. **Add More Tools**: Weather, Wikipedia, file operations
2. **Memory Integration**: Remember conversation history
3. **Custom Agents**: Build specialized agent types
4. **Web Integration**: Add internet search capabilities
5. **Multi-Agent Systems**: Multiple agents working together

## Best Practices for Beginners

### Start Simple

- Begin with one tool
- Test with basic questions
- Gradually add complexity

### Debug Effectively

- Use verbose mode to see reasoning
- Test tools independently first
- Add logging for troubleshooting

### Safety First

- Validate all tool inputs
- Use safe evaluation methods
- Implement proper error handling

### Iterate and Improve

- Start with basic functionality
- Collect example use cases
- Refine based on testing resultshttps://youtu.be/FQyrd26U3MU

## Environment setup:

    	Install packages

    	pip install --upgrade langchain langchain-community


    	#### Provides access to the OpenAI GPT models. You’ll also need an OpenAI API key, which you can get by signing up at OpenAI.
    	pip install openai


    	#### To manage environment variables using a .env file.
    	pip install python-dotenv

    	In the same directory as your script, create a .env file and add your API keys.


    	Create .env file and  paste these environment variables
