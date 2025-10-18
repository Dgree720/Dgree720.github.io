# 18_3RAG03.py
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

llm = OllamaLLM(model="gemma3:1b", temperature=0.7)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a content manager with extensive SEO knowledge.Your task is to write an article based on a given title",
        ),
        ("user", "{input}"),
    ]
)

chain = prompt | llm

print(chain.invoke({"input": "How does software change the world"}))
