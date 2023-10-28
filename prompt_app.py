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
#Prompt Templates for Enhanced Outputs
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

#Prompt Templates for Baseline Outputs
headline_prompt2 = PromptTemplate(
    input_variables = ["input"],
    template = 'I want you to act as a politician. You will research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop a press release about what happened during various periods of history. My first suggestion request is: write me a press release based on this headline: {input}'
)

press_template2 = PromptTemplate(
    input_variables = ["headline2"],
    template = 'I want you to act as a politician. You will research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop a press release about what happened during various periods of history. My first suggestion request is: write me a press release based on this headline: {headline2}'
)

twitter_template2 = PromptTemplate(
    input_variables = ["press_release2"],
    template = 'write me a twitter post based on this press release written by a politician: {press_release2}'
)

facebook_template2 = PromptTemplate(
    input_variables = ["twitter2"],
    template = 'write me a facebook post based on this twitter post tweeted by a politician: {twitter2}'
)

instagram_template2 = PromptTemplate(
    input_variables = ["facebook2"],
    template = 'write me an instagram post based on this facebook post written by a politician: {facebook2}'
)

#Memory
#Saves the chat history for the session - not used
# headline_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history", return_messages = True)
# press_memory = ConversationBufferMemory(input_key="headline", memory_key="chat_history", return_messages = True)
# twitter_memory = ConversationBufferMemory(input_key="press_release", memory_key="chat_history", return_messages = True)
# facebook_memory = ConversationBufferMemory(input_key="twitter", memory_key="chat_history", return_messages = True)
# instagram_memory = ConversationBufferMemory(input_key="facebook", memory_key="chat_history", return_messages = True)

#LLMs
#Runs the Generative AI model using LangChain using fine-tuned data and few shot prompting
llm = ChatOpenAI(temperature=0.5, model = "ft:gpt-3.5-turbo-0613:personal::84XCwFjs")
headline_chain = LLMChain(llm=llm, prompt=headline_prompt, verbose = True, output_key = "headline")
press_chain = LLMChain(llm=llm, prompt=press_template, verbose = True, output_key = "press_release")
twitter_chain = LLMChain(llm=llm, prompt=twitter_template, verbose = True, output_key = "twitter")
facebook_chain = LLMChain(llm=llm, prompt=facebook_template, verbose = True, output_key = "facebook")
instagram_chain = LLMChain(llm=llm, prompt=instagram_template, verbose = True, output_key = "instagram")

#Runs the Generative AI model using LangChain using basic model and limited prompting
llm2 = ChatOpenAI(temperature=0.5)
headline_chain2 = LLMChain(llm=llm2, prompt=headline_prompt2, verbose = True, output_key = "headline2",)
press_chain2 = LLMChain(llm=llm2, prompt=press_template2, verbose = True, output_key = "press_release2")
twitter_chain2 = LLMChain(llm=llm2, prompt=twitter_template2, verbose = True, output_key = "twitter2")
facebook_chain2 = LLMChain(llm=llm2, prompt=facebook_template2, verbose = True, output_key = "facebook2")
instagram_chain2 = LLMChain(llm=llm2, prompt=instagram_template2, verbose = True, output_key = "instagram2")

#Create function to generate fine-tuned content
def generate_fine(prompt):
    st.write("Your content is being generated. I am checking a number of sources and crafting an optimal solution for you - please give me a moment.")
    #Returns response to prompt: What Political Issue Should I Write About?
    #Creates wikipedia and google search instances
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

    #Feeds prompts into OpenAI LLM chains
    headline = headline_chain.run(prompt)
    press_release = press_chain.run(headline=headline,wikipedia_research=wiki_research,google=google_research)
    twitter = twitter_chain.run(press_release=press_release,headline=headline)
    facebook = facebook_chain.run(twitter=twitter,headline=headline)
    instagram = instagram_chain.run(facebook=facebook,headline=headline)

    #Uses session state to store variables
    st.session_state.headline_id = headline
    st.session_state.press_id = press_release
    st.session_state.twitter_id = twitter
    st.session_state.facebook_id = facebook
    st.session_state.instagram_id = instagram
    return headline, press_release, twitter, facebook, instagram, google_research, wiki_research

#Create function to generate default content
def generate_default(prompt):
  st.write("Your content is being generated. I am checking a number of sources and crafting an optimal solution for you - please give me a moment.")
  #Returns response to prompt: What Political Issue Should I Write About?
  #Feeds prompts into OpenAI LLM chains
  headline2 = headline_chain2.run(prompt)
  press_release2 = press_chain2.run(headline2=headline2)
  twitter2 = twitter_chain2.run(press_release2=press_release2,headline2=headline2)
  facebook2 = facebook_chain2.run(twitter2=twitter2,headline2=headline2)
  instagram2 = instagram_chain2.run(facebook2=facebook2,headline2=headline2)

  #Uses session state to store variables
  st.session_state.headline2_id = headline2
  st.session_state.press2_id = press_release2
  st.session_state.twitter2_id = twitter2
  st.session_state.facebook2_id = facebook2
  st.session_state.instagram2_id = instagram2
  return headline2, press_release2, twitter2, facebook2, instagram2



#App Framework
#Introduces app and ingests prompt provided by user
#Color Palette 1: https://coolors.co/palette/cc8b86-f9eae1-7d4f50-d1be9c-aa998f
#Color Palette 2: https://coolors.co/palette/e8d1c5-eddcd2-fff1e6-f0efeb-eeddd3-edede8
st.image('Logo/Political Banter-logos_transparent.png')
st.header('The Next Generation of Political Tech')
st.write('Learn more about Political Banter in the side bar!')
prompt = st.text_input('What Political Issue Should I Write About?')

#Creates session state for fine-tuned
if 'headline' not in st.session_state:
  st.session_state.headline_id = None

if 'press' not in st.session_state:
  st.session_state.press_id = None

if 'twitter' not in st.session_state:
  st.session_state.twitter_id = None

if 'facebook' not in st.session_state:
  st.session_state.facebook_id = None

if 'instagram' not in st.session_state:
  st.session_state.instagram_id = None

#Creates session state for baseline
if 'headline2' not in st.session_state:
  st.session_state.headline2_id = None

if 'press2' not in st.session_state:
  st.session_state.press2_id = None

if 'twitter2' not in st.session_state:
  st.session_state.twitter2_id = None

if 'facebook2' not in st.session_state:
  st.session_state.facebook2_id = None

if 'instagram2' not in st.session_state:
  st.session_state.instagram2_id = None

#Creates sidebar
with st.sidebar:
  st.title('About Political Banter')
  st.header('Using this tool is as simple as telling Political Banter what political issues you want it to write about.')
  st.markdown('Political Banter was created by finetuning an OpenAI chatGPT model based on a Kaggle database of Tweets by politicians from across the United States. Additional promting was also used to guide the algorithm to craft catchy political content in the form of a headline, press release, tweet, facebook post, and instagram post.')
  st.write('Learn more about the Kaggle dataset that was used to inform the tone and voice of Political Banter via the following link: https://www.kaggle.com/datasets/crowdflower/political-social-media-posts?resource=download')

#Creates radio button widget 
model = st.radio(
  "Which GenAI model would you like to use?",
  ["Fine-Tuned OpenAI Model","Default OpenAI Model"],
  captions = ["Includes Fine-Tuned OpenAI Model and Few Shot Prompts","Uses Default OpenAI Model and Basic Prompts"]
)

st.write('Please do not change the settings until after the content is generated. Otherwise, your content will not be generated.')

#Creates tabs to separate app features
tab1, tab2, tab3 = st.tabs(['Political Banter','Default','Data'])

#Selects which model to run and generate on tab 1
with tab1:
  if model == "Fine-Tuned OpenAI Model":
    #Creates button for generating content
    finebutton = st.button("Generate Fine-Tuned Content", type='primary')
    #Runs button to generate content
    if finebutton:
      if prompt:
          headline, press_release, twitter, facebook, instagram, google_research, wiki_research = generate_fine(prompt)
          #Adds returned results to tab 1 and uses expanders to separate topics
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
  else:
      st.write("Please select the Fine-Tuned OpenAI Model setting to generate new content")

#Selects which model to run and generate on tab 2
with tab2:
  if model == "Default OpenAI Model":
    #Creates button for generating content
    defbutton = st.button("Generate Default Content", type='primary')
    #Runs button to generate content
    if defbutton:
      if prompt:
        headline2, press_release2, twitter2, facebook2, instagram2 = generate_default(prompt)
        #Adds returned results to tab 1 and uses expanders to separate topics
        with tab2:
          st.write("Headline: " + headline2)
          # with st.expander("Headline History"):
          #   st.info(headline_memory.buffer)
          with st.expander("Press Release"):
            st.write(press_release2)
          # with st.expander("Press Release History"):
          #     st.info(press_memory.buffer)
          with st.expander("Tweet"):
            st.write(twitter2)
          # with st.expander("Tweet History"):
          #     st.info(twitter_memory.buffer)
          with st.expander("Facebook Post"):
            st.write(facebook2)
          # with st.expander("Facebook Post History"):
          #     st.info(facebook_memory.buffer)
          with st.expander("Instagram Post"):
            st.write(instagram2)
          # with st.expander("Instagram Post History"):
          #     st.info(instagram_memory.buffer)
  else:
    st.write("Please select the Default OpenAI Model setting to generate new content")
    

#Adds data table to tab 2
with tab3: 
  st.write("This data was ingested into the fine tuning GenerativeAI Model used to build the Political Banter App and can be found at: https://www.kaggle.com/datasets/crowdflower/political-social-media-posts?resource=download")
  st.write(data)
