from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="gpt-oss API")

llm = ChatOllama(model="gpt-oss:20b", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(q: Query):
    answer = await chain.ainvoke({"question": q.question})
    return {"answer": answer}

@app.post("/chat/stream")
async def chat_stream(q: Query):
    async def gen():
        async for chunk in chain.astream({"question": q.question}):
            yield chunk
    return StreamingResponse(gen(), media_type="text/plain")