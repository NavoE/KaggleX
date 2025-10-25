from langchain_core.prompts import PromptTemplate

# ── Baseline, safer & sharper ───────────────────────────────────────────────

headline_prompt2 = PromptTemplate(
    input_variables=["input"],
    template=(
        "You are a cautious, facts-first politician crafting a news-style headline.\n"
        "Context: {input}\n\n"
        "Requirements:\n"
        "• Output ONE headline only (no preamble or quotes around it).\n"
        "• ≤ 80 characters, active voice, specific, non-clickbait, no ALL CAPS.\n"
        "• Include the key actor + action + object; add locale or timeframe if clear.\n"
        "• If allegations/unclear facts: reflect uncertainty (e.g., “addresses report”, “responds to”).\n"
        "• Subtly mirror the likely voice implied by the context (measured, credible, not slogan-y)."
    )
)

press_template2 = PromptTemplate(
    input_variables=["headline2"],
    template=(
        "You are a responsible politician preparing a credible press release based on this headline:\n"
        "“{headline2}”.\n\n"
        "Write a factual, newsroom-ready release with this structure:\n"
        "1) Dateline and location (e.g., “RICHMOND, VA — [Month Day, Year] —”).\n"
        "2) 1–2 sentence lead that summarizes the news without speculation.\n"
        "3) Two short paragraphs: concrete facts, relevant background, what happens next.\n"
        "4) One short, human quote from the principal (format: “First Last, Candidate for X, said…”),\n"
        "   voiced to match a pragmatic, respectful political tone.\n"
        "5) Brief CTA (event/site/contact) and a 1–2 sentence campaign boilerplate.\n\n"
        "Credibility & research discipline:\n"
        "• Do NOT invent dates, numbers, or hidden sources. If specifics are unknown, say the campaign\n"
        "  is reviewing available information and will update the public.\n"
        "• If you reference public information, frame it carefully (e.g., “according to public reports”\n"
        "  or “state data indicate”), without fabricating citations.\n"
        "• Avoid legal conclusions and defamatory language; keep it measured and verifiable.\n"
        "• ≤ 300 words total."
    )
)

twitter_template2 = PromptTemplate(
    input_variables=["press_release2"],
    template=(
        "Create ONE tweet (≤280 chars) summarizing this press release for a politician:\n"
        "{press_release2}\n\n"
        "Rules:\n"
        "• Clear, factual, respectful; no speculation.\n"
        "• If a proof point is present, include exactly one (brief) and hedge if uncertain.\n"
        "• 0–2 concise hashtags; 0–1 emoji max (optional); include a short CTA with example.com if natural."
    )
)

facebook_template2 = PromptTemplate(
    input_variables=["twitter2"],
    template=(
        "Turn this tweet into a Facebook post in an accessible, professional tone:\n"
        "{twitter2}\n\n"
        "Requirements:\n"
        "• 3–5 sentences with short paragraphs; add one concrete next step (event/site/contact) with a placeholder link.\n"
        "• Up to 2 specific hashtags (no generic #politics); avoid ALL CAPS and emoji clutter.\n"
        "• Keep it factual; if claims are unverified, acknowledge ongoing review.\n"
        "• Subtly reflect the speaker’s likely voice (calm, solutions-focused)."
    )
)

instagram_template2 = PromptTemplate(
    input_variables=["facebook2"],
    template=(
        "Write an Instagram caption derived from this Facebook post:\n"
        "{facebook2}\n\n"
        "Requirements:\n"
        "• 2–4 short lines; mobile-friendly; warm, human tone (no heavy jargon).\n"
        "• If there’s a proof point, keep it brief and neutral; do not invent details.\n"
        "• End with 3–5 relevant, non-generic hashtags; ≤150 words total.\n"
        "• 0–2 tasteful emojis max (optional); avoid emoji spam."
    )
)
