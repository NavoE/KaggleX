from langchain.prompts import PromptTemplate

#Prompt Templates for Baseline Outputs
headline_prompt2 = PromptTemplate(
    input_variables = ["input"],
    template = 'I want you to act as a politician. You will research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop a headline about what happened during various periods of history. My first suggestion request is: write me a headline based on this input: {input}'
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
