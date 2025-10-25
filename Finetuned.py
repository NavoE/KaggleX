from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# --- Stronger, more realistic few-shot examples (topic → headline) ---
headline_examples = [
    {"topic": "Inflation relief bill",
     "headline": "Congress Weighs Inflation Relief Bill as Budget Talks Tighten"},
    {"topic": "Wildfire response funding",
     "headline": "Governor Proposes Surge Funding for Wildfire Response and Prevention"},
    {"topic": "Public safety and policing",
     "headline": "Mayor Details Community Partnership Plan to Improve Public Safety"},
    {"topic": "Small business tax credits",
     "headline": "State Expands Hiring Tax Credits to Support Small Businesses"},
    {"topic": "School infrastructure upgrades",
     "headline": "District Unveils Plan to Modernize Aging School Facilities"},
    {"topic": "Clean energy jobs",
     "headline": "Clean Energy Initiative Targets Thousands of New Local Jobs"},
    {"topic": "Veterans services expansion",
     "headline": "New Investments Expand Health and Housing Support for Veterans"},
]

# Select the most relevant example based on the user's "topic"
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=headline_examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=1,
    input_keys=["topic"],
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
        "Write ONE newsroom-style headline ONLY (no extra text).\n"
        "Rules:\n"
        "• ≤80 characters; active voice; specific; no clickbait; avoid ALL CAPS.\n"
        "• Include actor + action + object; add locale/timeframe if clear.\n"
        "• If facts are uncertain or allegations exist, use careful framing "
        "  (e.g., “addresses report”, “proposes”, “considers”).\n"
        "• Subtly mirror a pragmatic, credible political voice (no slogans)."
    ),
    input_variables=["topic"],
)

# Press release with stronger credibility + “researchy” discipline
press_template = PromptTemplate(
    input_variables=["headline", "wikipedia_research", "google"],
    template=(
        "You are a responsible politician writing a factual press release.\n"
        "Source inputs (may be incomplete; DO NOT fabricate):\n"
        "• Headline: {headline}\n"
        "• Wikipedia notes: {wikipedia_research}\n"
        "• Google notes: {google}\n\n"
        "Write a newsroom-ready release with this structure:\n"
        "1) DATELINE (CITY, ST — Month Day, Year —).\n"
        "2) Lead: 1–2 sentences summarizing the concrete news.\n"
        "3) Body: 2–3 short paragraphs with verifiable facts, relevant background, next steps.\n"
        "4) Quote: one short, human quote (First Last, Title) in a measured, authentic voice.\n"
        "5) CTA + boilerplate: brief call to action and 1–2 sentence campaign boilerplate.\n\n"
        "Credibility & research guardrails:\n"
        "• Do NOT invent dates, numbers, or sources. If specifics are unclear, explicitly note that the\n"
        "  campaign is reviewing public information and will update the public.\n"
        "• Attribute carefully using hedges like “according to public reports” or “state data indicate,”\n"
        "  only when supported by the notes above; never fabricate citations.\n"
        "• Avoid legal conclusions/defamation; keep tone respectful and factual.\n"
        "• ≤300 words total; concise and readable."
    ),
)

# Platform posts with clear constraints
twitter_template = PromptTemplate(
    input_variables=["press_release"],
    template=(
        "From this press release, write ONE tweet (≤280 chars):\n"
        "{press_release}\n\n"
        "Constraints: factual, respectful, no speculation; include at most one brief proof point if present; "
        "0–2 concise hashtags; 0–1 emoji (optional); optional short CTA with example.com."
    ),
)

facebook_template = PromptTemplate(
    input_variables=["twitter"],
    template=(
        "Turn this into a short Facebook post (3–5 sentences, line breaks ok):\n"
        "{twitter}\n\n"
        "Add one concrete next step (event/site/contact) with a placeholder link; up to 2 specific hashtags; "
        "avoid ALL CAPS & emoji clutter; keep it factual and acknowledge uncertainty when needed."
    ),
)

instagram_template = PromptTemplate(
    input_variables=["facebook"],
    template=(
        "Create an Instagram caption based on this:\n"
        "{facebook}\n\n"
        "2–4 short lines; warm, human tone; ≤150 words; if noting a proof point, keep it brief and neutral; "
        "end with 3–5 relevant, non-generic hashtags; 0–2 tasteful emojis max (optional)."
    ),
)
