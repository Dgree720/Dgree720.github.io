# 18_1RAG01s.py

from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3:1b", temperature=0.7)

for chunk in llm.stream("Hello, please introduce yourself"):
    print(chunk)
