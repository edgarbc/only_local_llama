from pandasai.llm.local_llm import localLLM 
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe

# Create an instance of the local LLM class
llm = localLLM(
    api_base = 'https://localhost:11434/v1',
    model_name = 'llama3',
)

# Create a streamlit app
st.title('Local LLM Example')

# upload a file
uploaded_file = st.file_uploader("Choose a csv file", type=['csv'])

if uploaded_file is not None:
    # Read the file
    data = pd.read_csv(uploaded_file)
    # Display some data
    st.write(data.head(3))
    # Create a SmartDataframe object
    sdf = SmartDataframe(data, config = {'llm': llm})

    # get a prompt from the user
    prompt = st.text_area('Enter a prompt:')
    if st.button('Generate response'):
        if prompt:
            with st.spinner('Generating response...'):
                # Get the response
                response = sdf.chat(prompt)
                st.write(response)
       