from langchain_core.prompts import PromptTemplate

# ── Baseline, safer & sharper ───────────────────────────────────────────────

headline_prompt2 = PromptTemplate(
    input_variables=["input"],
    template=(
        "You are a cautious, facts-first politician crafting a news-style headline.\n"
        "Context: {input}\n\n"
        "Requirements:\n"
        "• 1 headline only (no preamble).\n"
        "• ≤ 80 characters, active voice, specific, non-clickbait, no ALL CAPS.\n"
        "• Do NOT assert unverified allegations as fact; prefer neutral framing "
        "  like “responds to allegations” or “addresses report”.\n"
        "• If details are unclear, reflect uncertainty (e.g., “statement on” / “responds to”)."
    )
)

press_template2 = PromptTemplate(
    input_variables=["headline2"],
    template=(
        "You are a responsible politician preparing a press release based on this headline:\n"
        "“{headline2}”.\n\n"
        "Write a clear, factual press release with the following structure:\n"
        "1) Dateline and location (e.g., “RICHMOND, VA — [date] —”).\n"
        "2) 1–2 sentence lead that summarizes the news without speculation.\n"
        "3) 2–3 short paragraphs with verifiable facts, relevant background, and what happens next.\n"
        "4) One short quote from the principal (first and last name as “Candidate Lastname”).\n"
        "5) A brief call to action (event, website, or contact) and a 1–2 sentence campaign boilerplate.\n\n"
        "Guardrails:\n"
        "• Do NOT invent data, dates, or sources. If facts are unknown, state that the campaign is "
        "  reviewing available information and will update the public.\n"
        "• Be respectful; avoid legal conclusions and defamatory statements.\n"
        "• Keep total length under ~300 words."
    )
)

twitter_template2 = PromptTemplate(
    input_variables=["press_release2"],
    template=(
        "Create ONE tweet (≤280 chars) summarizing this press release, suitable for a politician:\n"
        "{press_release2}\n\n"
        "Rules:\n"
        "• Clear, factual, and respectful; no speculation.\n"
        "• At most 2 concise hashtags; no emoji spam (0–1 emoji max, optional).\n"
        "• If appropriate, add a short CTA with a placeholder link like example.com."
    )
)

facebook_template2 = PromptTemplate(
    input_variables=["twitter2"],
    template=(
        "Turn this tweet into a Facebook post in an accessible, professional tone:\n"
        "{twitter2}\n\n"
        "Requirements:\n"
        "• 3–5 sentences with short paragraphs (line breaks allowed).\n"
        "• Add one concrete next step (event, site, or contact) with a placeholder link.\n"
        "• Up to 2 hashtags; avoid all caps and excessive emojis.\n"
        "• Keep it factual; if claims are unverified, acknowledge ongoing review."
    )
)

instagram_template2 = PromptTemplate(
    input_variables=["facebook2"],
    template=(
        "Write an Instagram caption derived from the following Facebook post:\n"
        "{facebook2}\n\n"
        "Requirements:\n"
        "• 2–4 short lines; mobile-friendly.\n"
        "• Warm, human tone; avoid heavy policy jargon.\n"
        "• 3–5 relevant, non-generic hashtags at the end.\n"
        "• ≤150 words; 0–2 tasteful emojis (optional), no emoji flood."
    )
)
