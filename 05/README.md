# SQL Database AI Query Engine

Transform natural language questions into SQL queries and get instant answers from your database. This project combines Google Gemini AI with MySQL to create an intelligent database assistant that understands plain English queries.

## What You'll Learn

- Converting natural language to SQL queries using AI
- Integrating LangChain with SQL databases
- Working with MySQL and SQLAlchemy
- Building database-aware AI applications
- Creating business intelligence tools with AI

## Project Overview

This application demonstrates Text-to-SQL capabilities:

- **Natural Language Processing**: Ask questions in plain English
- **SQL Generation**: AI converts questions to SQL queries
- **Database Execution**: Automatically runs queries on real data
- **Result Display**: Shows both the generated SQL and results

## Technical Stack

- **Google Gemini Pro**: AI model for text-to-SQL conversion
- **LangChain**: Framework for building the query chain
- **MySQL**: Database management system
- **SQLAlchemy**: Python SQL toolkit and ORM
- **Streamlit**: Web interface for the application
- **Retail Sales Dataset**: Sample data for testing

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

### Step 2: Database Setup

#### Install MySQL

1. Download from [MySQL Official Site](https://dev.mysql.com/downloads/installer/)
2. Install MySQL Server and MySQL Workbench
3. Set up root user with password

#### Create Database

```bash
# Navigate to database folder
cd database/

# Import the SQL file
mysql -u root -p < retail_sales.sql
```

### Step 3: API Configuration

1. Get API key from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
2. Create `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### Step 4: Understanding the Code Architecture

```python
# 1. Database Connection
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
db = SQLDatabase(engine, sample_rows_in_table_info=3)

# 2. AI Model Setup
llm = GoogleGenerativeAI(model="gemini-pro")

# 3. Create SQL Query Chain
chain = create_sql_query_chain(llm, db)

# 4. Process Natural Language Query
sql_query = chain.invoke({"question": user_question})
result = db.run(sql_query)
```

## Running the Application

```bash
streamlit run myapp2.py
```

Visit http://localhost:8501 and start asking questions about your data!

## Sample Database Schema

The retail sales database contains:

```sql
CREATE TABLE sales_tb (
    TransactionID INT,
    Date DATE,
    CustomerID VARCHAR(10),
    Gender VARCHAR(10),
    Age INT,
    ProductCategory VARCHAR(50),
    Quantity INT,
    PriceperUnit DECIMAL(10,2),
    TotalAmount DECIMAL(10,2)
);
```

## Example Queries You Can Ask

### Basic Questions

- "How many total sales do we have?"
- "What are the different product categories?"
- "Show me all transactions from December 2023"

### Analytical Questions

- "What is the average transaction amount by gender?"
- "Which product category generates the most revenue?"
- "Who are our top 5 customers by total spending?"

### Complex Queries

- "What is the monthly sales trend for Electronics?"
- "Show me the age distribution of customers buying Beauty products"
- "Which gender spends more on average, and in which categories?"

## Learning Objectives

By completing this project, you will:

- Understand Text-to-SQL conversion with AI
- Learn LangChain's SQL chain components
- Master database integration with AI applications
- Implement error handling for SQL operations
- Create business intelligence tools

## Key Components Explained

### 1. SQL Database Integration

```python
# Sample rows help AI understand table structure
db = SQLDatabase(engine, sample_rows_in_table_info=3)
```

### 2. LangChain SQL Chain

```python
# Automatically generates SQL from natural language
chain = create_sql_query_chain(llm, db)
```

### 3. Error Handling

```python
try:
    response = chain.invoke({"question": question})
    result = db.run(response)
except ProgrammingError as e:
    st.error(f"An error occurred: {e}")
```

## Advanced Features to Implement

1. **Query History**: Save and recall previous queries
2. **Data Visualization**: Generate charts from query results
3. **Query Optimization**: Suggest better SQL queries
4. **Multiple Tables**: Join queries across related tables
5. **Export Results**: Download query results as CSV/Excel

## Business Use Cases

- **Sales Analytics**: Quick insights into sales performance
- **Customer Analysis**: Understanding customer behavior patterns
- **Inventory Management**: Track product performance
- **Financial Reporting**: Generate business reports from natural language
- **Data Exploration**: Non-technical users can query databases

## Video Tutorial

[Watch the complete tutorial](https://youtu.be/425N7n86QGw)

## Troubleshooting

**Common Issues:**

1. **Database Connection Errors**

   - Verify MySQL is running
   - Check credentials in the code
   - Ensure database exists

2. **SQL Generation Issues**

   - AI might generate invalid SQL for complex queries
   - Add more context to your questions
   - Check table schema matches expectations

3. **API Errors**
   - Verify Google API key is valid
   - Check internet connection
   - Monitor API usage limits

## Security Considerations

- Use environment variables for database credentials
- Implement SQL injection protection
- Validate AI-generated queries before execution
- Add user authentication for production use
- Monitor and log all database queries
