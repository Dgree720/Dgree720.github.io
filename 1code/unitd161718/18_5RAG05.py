# streamlit run 18_5RAG05.py
# pip install langchain-community

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Prompt 模板與對話歷史插槽
from langchain_core.runnables import RunnableWithMessageHistory  # 可記憶對話的 chain 包裝器
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # 使用 Streamlit 儲存對話歷史
from langchain_core.output_parsers import StrOutputParser  # 將輸出格式化為純文字

llm = OllamaLLM(model='gemma3:1b', temperature=0.7)

# ----------------- Initialize the prompt  -----------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# ----------------- Assemble the chain  -----------------
chain = prompt | llm | StrOutputParser()

# ----------------- Establish chat memory -----------------
chat_history = StreamlitChatMessageHistory()
# Package into a Chain retains memory，use session_id to Identify the user
chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,  # no matter what session_id is provided, it will always return the same single instance of chat_history
    input_messages_key="input",
    history_messages_key="history"
)

# ----------------- Streamlit -----------------
st.title("🧠 Chatbot with Memory (Gemma + LCEL) 18_5RAG05"+ '|Student ID|')
# Display history
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if user_input := st.chat_input("Ask anything..."):
    st.chat_message("human").write(user_input)

    with st.spinner("Generating..."):
        response = chain_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "user-session-001"}}
        )

    st.chat_message("ai").write(response)
