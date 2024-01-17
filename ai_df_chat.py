import streamlit as st
import tabulate
import pandas as pd
import datetime
import numpy as np
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

st.subheader('Chat with CSV data using OpenAI ChatGPT')

api_key = "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"
#os.environ['OPENAI_API_KEY'] = api_key
    # file uploader widget
uploaded_file = st.sidebar.file_uploader('Upload a dataframe in csv:', type='csv', accept_multiple_files=False)
    # add data button widget
# add_data = st.sidebar.button('Add Data')

if uploaded_file is not None:
    st.sidebar.success('File uploaded successfully.')
    df = pd.read_csv(uploaded_file)
    # llm = OpenAI(temperature=0)
    llm = ChatOpenAI(model_name='gpt-4', temperature=0, openai_api_key=api_key)
    agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=True)
    
    
    #st.sidebar.dataframe(df.head())

q = st.text_input('Ask a question about your data:')    
submit_q = st.button('Submit')
    
if submit_q: 
    standard_answer = "Answer only based on the text you received as input. Don't search external sources. " \
                        "If you can't answer then return `I DONT KNOW`."
    q = f"{q} {standard_answer}"

    answer = agent.run(q)

    # text area widget for the LLM answer
    st.text_area(label = "ChatGPT Response:", value=answer)

st.divider()
if uploaded_file is not None:
    st.write('Data Preview')
    st.dataframe(df.head(10))

# run the app: streamlit run ./df_doc_chat.py

