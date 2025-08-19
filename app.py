import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType,AgentExecutor, create_react_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain import hub
from dotenv import load_dotenv
load_dotenv()
import os
 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# if GROQ_API_KEY:
#     print("GROQ_API_KEY loaded successfully!")
#     print(f"Your key starts with: {GROQ_API_KEY[:4]}...") 
# else:
#     print("Error: GROQ_API_KEY not found.")

from langchain_groq import ChatGroq
model=ChatGroq(model="llama3-70b-8192",groq_api_key=GROQ_API_KEY,streaming=True
)
 
#Arxiv and WikiPedia Tool 
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=500)
arxiv_tool=ArxivQueryRun( name="Arxiv",
    description="Use this tool to search for scientific papers and research articles on ArXiv.",
    api_wrapper=arxiv_wrapper)

wikipedia_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wikipedia_tool=WikipediaQueryRun(name="Wikipedia",
    description="Use this tool to look up general knowledge, events, people, and concepts on Wikipedia.",
    api_wrapper=wikipedia_wrapper)

search_tool = DuckDuckGoSearchRun(name="DuckDuckGo Search",
    description="Use this tool for any general web searches to find information on the internet.")

tools = [wikipedia_tool, arxiv_tool, search_tool]

prompt_template = hub.pull("hwchase17/react")
llm=model
# Create the agent using the recommended create_react_agent function
agent = create_react_agent(llm, tools, prompt_template)

# Create the AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True # See agent's thoughts in the terminal
)

st.title("Langchain - Chat with Search")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"Assistant","content":"Hi ,I am a Chatbot who can search the web .How can I help you ?"}
    ]

for msg in st.session_state.messages :
    st.chat_message(msg['role']).write(msg['content'])

llm = model

if prompt:=st.chat_input(placeholder="What is Machine Learning ?"):

    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = agent_executor.invoke(
        {"input": prompt}, # Pass only the user's latest prompt
        {"callbacks": [st_cb]}
         )
        final_response = response.get("output", "Sorry, I encountered an error.")
        
        st.session_state.messages.append({"role":"Assistant","content":final_response})
        st.write(response) 