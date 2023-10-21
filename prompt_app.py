import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


#Import Kaggle data: https://www.kaggle.com/datasets/crowdflower/political-social-media-posts?resource=download
#Retrieve Data
data = pd.read_csv('political_social_media.csv', encoding_errors= "ignore")

#LangChain Crash Course: Build a AutoGPT app in 25 minutes!: https://www.youtube.com/watch?v=MlK6SIjcjE8
#Use to run Streamlit: python -m streamlit run prompt_app.py
#Finetuning for Tone: https://blog.langchain.dev/chat-loaders-finetune-a-chatmodel-in-your-voice/

# App Framework
st.title('Political Banter')
st.header('Your go to generative AI solution for producing intelligent and informed political messaging.')
st.subheader('Using this tool is as simple as telling the Political Banter tool what you want it to write about.')
st.text('Political Banter was created by finetuning an OpenAi chatGPT model based on a Kaggle database of Tweets by politicians from across the United States. Additional promting was also used to guide the algorithm to craft a catchy political content in the form of a headline, press release, tweet, facebook post, and instagram post.')
prompt = st.text_input('What Political Issue Should I Wite About?')

#Few Shot Prompts
# The few shot prompts will guide the algorithm to craft a catchy political headline.
headline_examples = [
  {
    "question": "Can you write me a political headline?",
    "answer":
"""
topic: Politics Unveiled: A Closer Look at the Game of Power and Influence
"""
  },
  {
    "question": "Can you write me a political headline?",
    "answer":
"""
topic: Political Turmoil: Challenges and Controversies Shape the World of Politics
"""
  },
  {
    "question": "Can you write me a political headline?",
    "answer":
"""
topic: Political Landscape Shifts as New Policies and Candidates Emerge
"""
  },
  {
    "question": "Can you write me a political headline?",
    "answer":
"""
topic: Politics: Shaping Society and Nation Through Governance and Policy-Making
"""
  },
  {
    "question": "Can you write me a political headline?",
    "answer":
"""
topic: Breaking News: Political Turmoil Shakes the Nation
"""
  },
   {
    "question": "Can you write me a political headline?",
    "answer":
"""
topic: Political Turmoil Grips Nation as Controversial Policies Divide Citizen
"""
  },
  {
    "question": "Can you write me a political headline?",
    "answer":
"""
topic: Political Landscape Shifts as New Policies Take Center Stage
"""
  },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    headline_examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=1
)

#Prompt Templates
#The prompt templates will determine the app's output.
headline_template = PromptTemplate(
    input_variables = ["question","answer"],
    template = 'question: {question} \n {answer}'
)

headline_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=headline_template,
    suffix="Prompt: {topic}",
    input_variables=["topic"]

)

press_template = PromptTemplate(
    input_variables = ["headline", "wikipedia_research"],
    template = 'I want you to act as a politician. You will research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop a press release about what happened during various periods of history. My first suggestion request is: write me a press release based on this headline: {headline} while leveraging this wikipedia research : {wikipedia_research}'
)

twitter_template = PromptTemplate(
    input_variables = ["press_release"],
    template = 'write me a twitter post based on this press release written by a politician: {press_release}'
)

facebook_template = PromptTemplate(
    input_variables = ["twitter"],
    template = 'write me a facebook post based on this twitter post tweeted by a politician: {twitter}'
)

instagram_template = PromptTemplate(
    input_variables = ["facebook"],
    template = 'write me an instagram post based on this facebook post written by a politician: {facebook}'
)

# #Memory
# headline_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
# press_memory = ConversationBufferMemory(input_key="headline", memory_key="chat_history")
# twitter_memory = ConversationBufferMemory(input_key="press_release", memory_key="chat_history")
# facebook_memory = ConversationBufferMemory(input_key="twitter", memory_key="chat_history")
# instagram_memory = ConversationBufferMemory(input_key="facebook", memory_key="chat_history")

#LLMs
llm = ChatOpenAI(temperature=0.5, model = "ft:gpt-3.5-turbo-0613:personal::84XCwFjs")
headline_chain = LLMChain(llm=llm, prompt=headline_prompt, verbose = True, output_key = "headline")
press_chain = LLMChain(llm=llm, prompt=press_template, verbose = True, output_key = "press_release")
twitter_chain = LLMChain(llm=llm, prompt=twitter_template, verbose = True, output_key = "twitter")
facebook_chain = LLMChain(llm=llm, prompt=facebook_template, verbose = True, output_key = "facebook")
instagram_chain = LLMChain(llm=llm, prompt=instagram_template, verbose = True, output_key = "instagram")

wiki = WikipediaAPIWrapper()

# Return response to prompt
if prompt:
    headline = headline_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    press_release = press_chain.run(headline=headline,wikipedia_research=wiki_research)
    twitter = twitter_chain.run(press_release=press_release,headline=headline,wikipedia_research=wiki_research)
    facebook = facebook_chain.run(twitter=twitter,headline=headline,wikipedia_research=wiki_research)
    instagram = instagram_chain.run(facebook=facebook,wikipedia_research=wiki_research)
    col1, col2, col3, col4 = st.columns([3,1,1,1])
    st.write("Headline: " + headline)
    col1.write("Press Release: " + press_release)
    col2.write("Tweet: " + twitter)
    col3.write("Facebook Post: " + facebook)
    col4.write("Instagram Post: " + instagram)

    # with st.expander("Headline History"):
    #     st.info(headline_memory.buffer)
    # with st.expander("Press Release History"):
    #     st.info(press_memory.buffer)
    # with st.expander("Wikipedia Research History"):
    #     st.info(wiki_research)
    # with st.expander("Tweet History"):
    #     st.info(twitter_memory.buffer)
    # with st.expander("Facebook Post History"):
    #     st.info(facebook_memory.buffer)
    # with st.expander("Instagram Post History"):
    #     st.info(instagram_memory.buffer)