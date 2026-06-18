from dotenv import load_dotenv
import os

from typing import List
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

# ==========================================
# Pydantic Schemas
# ==========================================

class FlashCard(BaseModel):
    question: str
    answer: str


class FlashCardSet(BaseModel):
    flashcards: List[FlashCard]

# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile"
)

# ==========================================
# Structured Output
# ==========================================

structured_llm = llm.with_structured_output(
    FlashCardSet
)

# ==========================================
# Prompt Template
# ==========================================

prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are an expert teacher.

Generate exactly 5 educational flashcards about:

{topic}

Each flashcard must contain:

- question
- answer

Keep answers concise and beginner friendly.
"""
)

# ==========================================
# LCEL Chain
# ==========================================

chain = prompt | structured_llm

# ==========================================
# User Input
# ==========================================

topic = input("Enter Topic: ")

# ==========================================
# Invoke Chain
# ==========================================

response = chain.invoke(
    {"topic": topic}
)

# ==========================================
# Output
# ==========================================

print("\n" + "=" * 80)
print(f"FLASHCARDS FOR: {topic}")
print("=" * 80)

for idx, card in enumerate(response.flashcards, start=1):
    print(f"\nFlashcard {idx}")
    print(f"Q: {card.question}")
    print(f"A: {card.answer}")

print("\n" + "=" * 80)