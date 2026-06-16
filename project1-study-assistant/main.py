from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are an expert teacher and interviewer.

Explain the topic: {topic}

Format your response EXACTLY in the following sections:

# Summary
Provide a concise explanation suitable for a beginner.

# Important Concepts
List at least 5 important concepts and explain each one briefly.

# Interview Questions
Provide 5 interview questions with answers.

# Quiz
Create 5 multiple-choice questions (MCQs) with 4 options each and indicate the correct answer.

Make the content clear, structured, and easy to understand.
"""
)

# LCEL Chain
chain = prompt | llm

# Inspect LangChain Objects
print("\n=== LangChain Components ===")
print("Prompt Type:", type(prompt))
print("LLM Type:", type(llm))
print("Chain Type:", type(chain))

# User Input
topic = input("\nEnter a topic: ")

# Invoke Chain
response = chain.invoke(
    {"topic": topic}
)

# Output
print("\n" + "=" * 80)
print(response.content)
print("=" * 80)