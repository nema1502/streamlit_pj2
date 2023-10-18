import os
from dotenv import find_dotenv, load_dotenv
import openai
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain

load_dotenv(find_dotenv())

openai_api_key = st.text_input("Enter OpenAI API Key")

st.title("Tweet Generator")
st.header("Generate Your Next Tweet with Tweet Bot")

st.text("1. Describe Your Tweet (Or Copy and Paste and Existing One)")
prompt = st.text_area("", height=14)


st.text("2. Select Your Voice")
voice_options = st.multiselect(label="",
                               options=["Casual",
                                        "Prefessional",
                                        "Funny"])
# st.write("You've Selected: ", voice_options)
llm = OpenAI(temperature="0.9")

tweet_one_templage = PromptTemplate(
    input_variables=["description", "option"],
    template="""Write me a very {option} tweet that is based on this description: {description}. 
    Include two appropriate emojis and hashtags at the end of the tweet"""
)

tweet_two_templage = PromptTemplate(
    input_variables=["description", "option"],
    template="""Write me a very {option} tweet that is based on {description}. 
    Include two appropriate emojis and hashtags at the end of the tweet"""
)

tweet_one_chain = LLMChain(llm=llm, prompt=tweet_one_templage,
                           verbose=True, output_key="tweet_one")
tweet_two_chain = LLMChain(llm=llm, prompt=tweet_two_templage,
                           verbose=True, output_key="tweet_two")

# combine both chains into a Sequence!
sequential_chain = SequentialChain(
    chains=[tweet_one_chain, tweet_two_chain],
    input_variables=["description", "option"],
    output_variables=["tweet_one", "tweet_two"]
)

if st.button("Generate Tweets") and prompt and voice_options:
    response = sequential_chain({"description":prompt, 
                                 "option":voice_options})
    
    #print(f"Tweet 1: {response['tweet_one']} -- Tweet2:{response['tweet_two']}")
    st.info(response['tweet_one'])
    st.divider()
    st.info(response['tweet_two'])
    
