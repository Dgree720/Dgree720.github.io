# 18_1RAG01.py
# pip install langchain langchain-ollama pypdf langchain-community fastembed chromadb

from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3:1b", temperature=0.7)

print(llm.invoke("Hello, please introduce yourself"))
