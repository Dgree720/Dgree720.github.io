# 18_2RAG02.py
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

llm = OllamaLLM(model='gemma3:1b', temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}"),])

chain = prompt | llm

print(chain.invoke({"input": "Please introduce Pforzheim"}))