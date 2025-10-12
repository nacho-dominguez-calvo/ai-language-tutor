from openai import OpenAI
import os
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize model (use a cheap model)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create a reusable prompt template
SYSTEM_PROMPT = """You are a helpful AI assistant. You should answer concisely and with short output.
If you don't know the answer, just say that you don't know, don't try to make up an answer. If the user asks you to make up an answer, refuse.
Use at most 50 words in your answer. Use only information from the documents provided by the user.
If there is a clash between the documents, say that there is a clash and provide both answers.
If there is a clash between the documents and your prior knowledge, always use the documents, but say that there is a clash.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{user_input}")
])

# Define function to get response
def get_response(prompt: str) -> str:
    """Send a message to a cheap model using LangChain and return the reply."""
    chain = prompt_template | llm
    response = chain.invoke({"user_input": prompt})
    return response.content
