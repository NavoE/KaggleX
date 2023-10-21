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
from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['GOOGLE_CSE_ID'] = st.secrets['GOOGLE_CSE_ID']
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

#Import Kaggle data: https://www.kaggle.com/datasets/crowdflower/political-social-media-posts?resource=download
#Retrieve Data
data = pd.read_csv('political_social_media.csv', encoding_errors= "ignore")

#LangChain Crash Course: Build a AutoGPT app in 25 minutes!: https://www.youtube.com/watch?v=MlK6SIjcjE8
#Use to run Streamlit: python -m streamlit run prompt_app.py
#Finetuning for Tone: https://blog.langchain.dev/chat-loaders-finetune-a-chatmodel-in-your-voice/

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
    input_variables = ["headline", "wikipedia_research","google"],
    template = 'I want you to act as a politician. You will research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop a press release about what happened during various periods of history. My first suggestion request is: write me a press release based on this headline: {headline} while leveraging this wikipedia research: {wikipedia_research} and google research: {google}'
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

#Memory
#Saves the chat history for the session
headline_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history", return_messages = True)
press_memory = ConversationBufferMemory(input_key="headline", memory_key="chat_history", return_messages = True)
twitter_memory = ConversationBufferMemory(input_key="press_release", memory_key="chat_history", return_messages = True)
facebook_memory = ConversationBufferMemory(input_key="twitter", memory_key="chat_history", return_messages = True)
instagram_memory = ConversationBufferMemory(input_key="facebook", memory_key="chat_history", return_messages = True)

#LLMs
#Runs the Generative AI model using LangChain
llm = ChatOpenAI(temperature=0.5, model = "ft:gpt-3.5-turbo-0613:personal::84XCwFjs")
headline_chain = LLMChain(llm=llm, prompt=headline_prompt, verbose = True, output_key = "headline",memory=headline_memory)
press_chain = LLMChain(llm=llm, prompt=press_template, verbose = True, output_key = "press_release",memory=press_memory)
twitter_chain = LLMChain(llm=llm, prompt=twitter_template, verbose = True, output_key = "twitter",memory=twitter_memory)
facebook_chain = LLMChain(llm=llm, prompt=facebook_template, verbose = True, output_key = "facebook",memory=facebook_memory)
instagram_chain = LLMChain(llm=llm, prompt=instagram_template, verbose = True, output_key = "instagram",memory=instagram_memory)

# App Framework
#Color Palette 1: https://coolors.co/palette/cc8b86-f9eae1-7d4f50-d1be9c-aa998f
#Color Palette 2: https://coolors.co/palette/e8d1c5-eddcd2-fff1e6-f0efeb-eeddd3-edede8
st.image('Logo/Political Banter-logos_transparent.png')
st.header('The Next Generation of Political Tech')
st.write('Learn more about how Political Banter was developed in the side bar!')
prompt = st.text_input('What Political Issue Should I Write About?')

with st.sidebar:
  st.title('Political Banter')
  st.header('Using this tool is as simple as telling the Political Banter tool what you want it to write about.')
  st.markdown('Political Banter was created by finetuning an OpenAi chatGPT model based on a Kaggle database of Tweets by politicians from across the United States. Additional promting was also used to guide the algorithm to craft a catchy political content in the form of a headline, press release, tweet, facebook post, and instagram post.')

#Returns response to prompt: What Political Issue Should I Write About?
#Uses expanders and tabs to separate topics and data
if prompt:
    search = GoogleSearchAPIWrapper()
    tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
    )
    wiki = WikipediaAPIWrapper()
    headline = headline_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    google_research = tool.run(prompt)
    press_release = press_chain.run(headline=headline,wikipedia_research=wiki_research,google=google_research)
    twitter = twitter_chain.run(press_release=press_release,headline=headline,wikipedia_research=wiki_research,google=google_research)
    facebook = facebook_chain.run(twitter=twitter,headline=headline,wikipedia_research=wiki_research,google=google_research)
    instagram = instagram_chain.run(facebook=facebook,wikipedia_research=wiki_research,google=google_research)
    st.write("Headline: " + headline)
    # with st.expander("Headline History"):
    #   st.info(headline_memory.buffer)
    with st.expander("Press Release"):
      st.write(press_release)
    # with st.expander("Press Release History"):
    #     st.info(press_memory.buffer)
    with st.expander("Tweet"):
      st.write(twitter)
    # with st.expander("Tweet History"):
    #     st.info(twitter_memory.buffer)
    with st.expander("Facebook Post"):
      st.write(facebook)
    # with st.expander("Facebook Post History"):
    #     st.info(facebook_memory.buffer)
    with st.expander("Instagram Post"):
      st.write(instagram)
    # with st.expander("Instagram Post History"):
    #     st.info(instagram_memory.buffer)
    with st.expander("Google Research"):
        st.info(google_research)
    with st.expander("Wikipedia Research"):
        st.info(wiki_research)
    st.write("This data was used to fine tune the GenerativeAI Model used to build the Political Banter App and can be found at: https://www.kaggle.com/datasets/crowdflower/political-social-media-posts?resource=download")
    st.write(data)
    