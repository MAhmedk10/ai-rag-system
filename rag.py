import os,requests
import chainlit as cl
from agents import Agent,AsyncOpenAI,OpenAIChatCompletionsModel,RunConfig,Runner, function_tool
from openai.types.responses import ResponseTextDeltaEvent
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", # Using gemini-2.0-flash model
    openai_client=provider
)

runcofig = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True # Disable tracing for cleaner runs
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="RETRIEVAL_DOCUMENT", # Specify task type for better embeddings
    google_api_key=gemini_api_key # Use the loaded Gemini API key
)

index_name = "agentic-rag"
pc_key = "pcsk_3BWZjc_XN1oG5bGCidctJr7Uudz9P8fickAMyYq5vJzxMG2N3fVfd8CRxjrKUoVnv6DM2"

pc = Pinecone(api_key=pc_key)
# if index_name not in pc.list_indexes():
#     pc.create_index(
#        index_name,
#         dimension=768,
#         metric="cosine", # Cosine similarity for embedding comparison
#         spec=ServerlessSpec(
#         cloud="aws",
#         egion="us-east-1"
#     ))
index = pc.Index(index_name)

loader = PyMuPDFLoader("mybook.pdf")
book = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks = splitter.split_documents(book)
texts = [chunk.page_content for chunk in chunks]

vectors = embeddings.embed_documents(texts=texts)
pinecone_data = [(f"id_{i}", vectors[i], {"text": texts[i]}) for i in range(len(texts))]
# Split into batches of 100
batch_size = 100
for i in range(0, len(pinecone_data), batch_size):
    batch = pinecone_data[i:i+batch_size]
    index.upsert(batch)


@function_tool
def embed_query(query:str):
    query_vectors = embeddings.embed_query(query)
    res = index.query(vector=query_vectors,top_k=3,include_metadata=True)
    retrieved_chunks = [match["metadata"]["text"] for match in res["matches"]]
    context = "\n\n".join(retrieved_chunks)
    return context


agent = Agent(name="Ahmed's Support Agent",instructions="You are a helpful AI instructor. If the user asks any question related to the book *Rich Dad Poor Dad* or about maths, use the `embed_query` tool to embed the question, retrieve relevant context from the document, and then answer using that context."
,tools = [embed_query])

@cl.on_chat_start
async def handle_chat():
    cl.user_session.set("history",[])
    await cl.Message(content = "Hello! I'm Ahmed's Support Agent.How may I help you?").send()


@cl.on_message
async def main(message: cl.Message):
    history= cl.user_session.get("history") 
    history.append({"role":"user","content":message.content})
    msg = cl.Message(content="")

    result = Runner.run_streamed(
    agent,
    input= history,
    run_config=runcofig
)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data,ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role":"assistant","content":result.final_output})
    cl.user_session.set("history",history)


