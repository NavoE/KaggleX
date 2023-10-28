import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['GOOGLE_CSE_ID'] = st.secrets['GOOGLE_CSE_ID']
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

#Import Kaggle data: https://www.kaggle.com/datasets/crowdflower/political-social-media-posts?resource=download
#Retrieve Data
data = pd.read_csv('political_social_media.csv', encoding_errors= "ignore")

#LangChain Crash Course: Build a AutoGPT app in 25 minutes!: https://www.youtube.com/watch?v=MlK6SIjcjE8
#Use to run Streamlit: python -m streamlit run prompt_app.py
#Finetuning for Tone: https://blog.langchain.dev/chat-loaders-finetune-a-chatmodel-in-your-voice/

#Memory
#Saves the chat history for the session - not used
# headline_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history", return_messages = True)
# press_memory = ConversationBufferMemory(input_key="headline", memory_key="chat_history", return_messages = True)
# twitter_memory = ConversationBufferMemory(input_key="press_release", memory_key="chat_history", return_messages = True)
# facebook_memory = ConversationBufferMemory(input_key="twitter", memory_key="chat_history", return_messages = True)
# instagram_memory = ConversationBufferMemory(input_key="facebook", memory_key="chat_history", return_messages = True)

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

if 'google' not in st.session_state:
  st.session_state.google_id = None

if 'wiki' not in st.session_state:
  st.session_state.wiki_id = None

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


def finestate(headline, press_release, twitter, facebook, instagram, google_research, wiki_research):
  #Uses session state to store fine-tuned variables
  st.session_state.headline_id = headline
  st.session_state.press_id = press_release
  st.session_state.twitter_id = twitter
  st.session_state.facebook_id = facebook
  st.session_state.instagram_id = instagram
  st.session_state.google_id = google_research
  st.session_state.wiki_id = wiki_research

  #Adds returned results to tab 1 and uses expanders to separate topics
  with st.expander("Press Release"):
    st.write(st.session_state.press_id)
  # with st.expander("Press Release History"):
  #     st.info(press_memory.buffer)
  with st.expander("Tweet"):
    st.write(st.session_state.twitter_id)
  # with st.expander("Tweet History"):
  #     st.info(twitter_memory.buffer)
  with st.expander("Facebook Post"):
    st.write(st.session_state.facebook_id)
  # with st.expander("Facebook Post History"):
  #     st.info(facebook_memory.buffer)
  with st.expander("Instagram Post"):
    st.write(st.session_state.instagram_id)
  # with st.expander("Instagram Post History"):
  #     st.info(instagram_memory.buffer)
  with st.expander("Google Research"):
      st.info(st.session_state.google_id)
  with st.expander("Wikipedia Research"):
      st.info(st.session_state.wiki_id)
  return st.session_state

def defstate(headline2, press_release2, twitter2, facebook2, instagram2):
  #Uses session state to store default variables
  st.session_state.headline2_id = headline2
  st.session_state.press2_id = press_release2
  st.session_state.twitter2_id = twitter2
  st.session_state.facebook2_id = facebook2
  st.session_state.instagram2_id = instagram2

  #Adds returned results to tab 2 and uses expanders to separate topics
  with st.expander("Press Release"):
    st.write(st.session_state.press2_id)
  # with st.expander("Press Release History"):
  #     st.info(press_memory.buffer)
  with st.expander("Tweet"):
    st.write(st.session_state.twitter2_id)
  # with st.expander("Tweet History"):
  #     st.info(twitter_memory.buffer)
  with st.expander("Facebook Post"):
    st.write(st.session_state.facebook2_id)
  # with st.expander("Facebook Post History"):
  #     st.info(facebook_memory.buffer)
  with st.expander("Instagram Post"):
    st.write(st.session_state.instagram2_id)
  # with st.expander("Instagram Post History"):
  #     st.info(instagram_memory.buffer)
  return st.session_state
  
#Create function to generate fine-tuned content
def generate_fine(prompt):
    st.write("Your content is being generated. I am checking a number of sources and crafting an optimal solution for you - please give me a moment.")
    #Returns response to prompt: What Political Issue Should I Write About?
    #Runs the Generative AI model using LangChain using fine-tuned data and few shot prompting
    from Finetuned import headline_prompt, press_template, twitter_template, facebook_template, instagram_template
    llm = ChatOpenAI(temperature=0.5, model = "ft:gpt-3.5-turbo-0613:personal::84XCwFjs")
    headline_chain = LLMChain(llm=llm, prompt=headline_prompt, verbose = True, output_key = "headline")
    press_chain = LLMChain(llm=llm, prompt=press_template, verbose = True, output_key = "press_release")
    twitter_chain = LLMChain(llm=llm, prompt=twitter_template, verbose = True, output_key = "twitter")
    facebook_chain = LLMChain(llm=llm, prompt=facebook_template, verbose = True, output_key = "facebook")
    instagram_chain = LLMChain(llm=llm, prompt=instagram_template, verbose = True, output_key = "instagram")

    #Creates wikipedia and google search instances
    search = GoogleSearchAPIWrapper()
    tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
    )
    wiki = WikipediaAPIWrapper()
    wiki_research = wiki.run(prompt)
    google_research = tool.run(prompt)

    #Feeds prompts into OpenAI LLM chains
    headline = headline_chain.run(prompt)
    st.write("Headline: " + headline)
    # with st.expander("Headline History"):
    #   st.info(headline_memory.buffer)
    press_release = press_chain.run(headline=headline,wikipedia_research=wiki_research,google=google_research)
    twitter = twitter_chain.run(press_release=press_release,headline=headline)
    facebook = facebook_chain.run(twitter=twitter,headline=headline)
    instagram = instagram_chain.run(facebook=facebook,headline=headline)
    return headline, press_release, twitter, facebook, instagram, google_research, wiki_research

#Create function to generate default content
def generate_default(prompt):
  st.write("Your content is being generated. I am checking a number of sources and crafting an optimal solution for you - please give me a moment.")
  #Returns response to prompt: What Political Issue Should I Write About?
  #Runs the Generative AI model using LangChain using basic model and limited prompting
  from Baseline import headline_prompt2, press_template2, twitter_template2, facebook_template2, instagram_template2
  llm2 = ChatOpenAI(temperature=0.5, model='gpt-3.5-turbo')
  headline_chain2 = LLMChain(llm=llm2, prompt=headline_prompt2, verbose = True, output_key = "headline2",)
  press_chain2 = LLMChain(llm=llm2, prompt=press_template2, verbose = True, output_key = "press_release2")
  twitter_chain2 = LLMChain(llm=llm2, prompt=twitter_template2, verbose = True, output_key = "twitter2")
  facebook_chain2 = LLMChain(llm=llm2, prompt=facebook_template2, verbose = True, output_key = "facebook2")
  instagram_chain2 = LLMChain(llm=llm2, prompt=instagram_template2, verbose = True, output_key = "instagram2")

  #Feeds prompts into OpenAI LLM chains
  headline2 = headline_chain2.run(prompt)
  st.write("Headline: " + headline2)
  # with st.expander("Headline History"):
  #   st.info(headline_memory.buffer)
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
          finestate(headline, press_release, twitter, facebook, instagram, google_research, wiki_research)
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
        defstate(headline2, press_release2, twitter2, facebook2, instagram2)
  else:
    st.write("Please select the Default OpenAI Model setting to generate new content")
    

#Adds data table to tab 2
with tab3: 
  st.write("This data was ingested into the fine tuning GenerativeAI Model used to build the Political Banter App and can be found at: https://www.kaggle.com/datasets/crowdflower/political-social-media-posts?resource=download")
  st.write(data)
