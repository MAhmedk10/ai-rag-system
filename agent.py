# import os,requests
# import chainlit as cl
# from agents import Agent,AsyncOpenAI,OpenAIChatCompletionsModel,RunConfig,Runner, function_tool
# from openai.types.responses import ResponseTextDeltaEvent


# from dotenv import load_dotenv,find_dotenv
# load_dotenv(find_dotenv())

# gemini_api_key = os.getenv("GEMINI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")

# provider = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=provider
# )

# runcofig = RunConfig(
#     model=model,
#     model_provider=provider,
#     tracing_disabled=True
# )

# data = {
#     "Ali": 85,
#     "Sara": 90,
#     "Ahmed": 78,
#     "Zara": 88,
#     "Bilal": 92,
#     "Fatima": 80,
#     "Omar": 76,
#     "Aisha": 89,
#     "Hassan": 95,
#     "Noor": 83,
#     "Arham chussar":69
# }

# @function_tool("getMarks")
# def getMarks(name:str):
#     nam = name.capitalize()
#     if nam in data:
#         return data.get(nam,"name is not in the data!")
    
# @function_tool
# def getInfo(url):
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             return response.text[0:]  # Limit response to avoid excessive text
#         else:
#             return f"Error fetching data: {response.status_code}"
#     except Exception as e:
#         return f"Failed to retrieve information: {str(e)}"

# agent1 = Agent(name="Ahmeds support Agent",instructions="You are a helping instructor,if user give you link call the getinfo function ad retrieve the information ")#tools=[getMarks,getInfo]

# @cl.on_chat_start
# async def handle_chat():
#     cl.user_session.set("history",[])
#     await cl.Message(content = "Hello! I'm Ahmed's Support Agent.How may I help you?").send()
    

# @cl.on_message
# async def main(message: cl.Message):
#     history= cl.user_session.get("history") 
#     history.append({"role":"user","content":message.content})
#     msg = cl.Message(content="")


#     result = Runner.run_streamed(
#         agent1,
#         input = history,
#         run_config=runcofig   
# )
#     async for event in result.stream_events():
#         if event.type == "raw_response_event" and isinstance(event.data,ResponseTextDeltaEvent):
#             await msg.stream_token(event.data.delta)

#     history.append({"role":"assistant","content":result.final_output})
#     cl.user_session.set("history",history)
