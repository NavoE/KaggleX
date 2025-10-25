from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_community.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# --- Few-shot examples (match on "topic", show target "headline") ---
headline_examples = [
    {"topic": "Inflation relief bill",
     "headline": "Congress Debates Inflation Relief Bill Amid Budget Standoff"},
    {"topic": "Wildfire response funding",
     "headline": "Governor Announces Boost to Wildfire Response and Prevention"},
    {"topic": "Public safety and policing",
     "headline": "Mayor Unveils Community-First Plan to Improve Public Safety"},
    {"topic": "Small business tax credits",
     "headline": "State Expands Tax Credits to Help Small Businesses Hire"},
    {"topic": "School infrastructure upgrades",
     "headline": "District Launches Plan to Modernize Aging School Facilities"},
    {"topic": "Clean energy jobs",
     "headline": "Clean Energy Initiative Aims to Create Thousands of Local Jobs"},
    {"topic": "Veterans services expansion",
     "headline": "New Investments Expand Health and Housing Support for Veterans"},
]

# Select the most relevant example based on the user's "topic"
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=headline_examples,
    embeddings=OpenAIEmbeddings(),   # <-- instance, not class
    vectorstore_cls=Chroma,
    k=1,
    input_keys=["topic"],            # <-- the field used from the runtime input
)

# How each example is rendered in the few-shot prompt
example_prompt = PromptTemplate(
    input_variables=["topic", "headline"],
    template="User topic: {topic}\nGood headline: {headline}"
)

# Final few-shot prompt the chain will use
headline_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix=(
        "User topic: {topic}\n"
        "Write ONE news-style headline only (≤80 chars), factual, specific, and non-clickbait. "
        "Avoid ALL CAPS; no extra text."
    ),
    input_variables=["topic"],
)

# Other templates (unchanged except clearer guardrails if you want them)
press_template = PromptTemplate(
    input_variables=["headline", "wikipedia_research", "google"],
    template=(
        "I want you to act as a politician. Write a factual press release based on:\n"
        "Headline: {headline}\nWikipedia notes: {wikipedia_research}\nGoogle notes: {google}\n\n"
        "Structure: dateline, 1–2 sentence lead, 2–3 short paragraphs with facts/background/next steps, "
        "one short quote, a brief CTA + boilerplate. Avoid speculation."
    ),
)

twitter_template = PromptTemplate(
    input_variables=["press_release"],
    template="Write ONE tweet (≤280 chars), factual and respectful, based on this: {press_release}"
)

facebook_template = PromptTemplate(
    input_variables=["twitter"],
    template=(
        "Turn this into a short Facebook post (3–5 sentences, line breaks ok), "
        "with a clear next step and up to 2 hashtags: {twitter}"
    ),
)

instagram_template = PromptTemplate(
    input_variables=["facebook"],
    template=(
        "Create an Instagram caption (2–4 short lines, warm tone, ≤150 words) based on: {facebook}\n"
        "End with 3–5 relevant hashtags; avoid emoji spam."
    ),
)
