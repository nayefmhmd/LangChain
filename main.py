from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral")
response = llm.invoke("Explain LangChain in one sentence.")
print(response.content)