# utils/prompts.py

ENRICHMENT_PROMPT_TEMPLATE = """
Your task is to convert a raw chat log into a structured, "Intelligent Log" in Markdown format.
Your output MUST start with a YAML frontmatter block containing a concise, 1-2 sentence summary.
Then, for each turn in the conversation, create a separate Markdown section.
Each section must include the speaker and the raw text.
Most importantly, below the raw text, add structured tags to capture key information. Use the format `[type: tag_type] [key: value]...`.

Available Tag Types:
- `[type: decision]`: For explicit decisions made. (e.g., `[object: Project X] [outcome: approved]`)
- `[type: action_item]`: For tasks assigned. (e.g., `[subject: Bob] [predicate: is_responsible_for] [object: updating roadmap]`)
- `[type: fact]`: For objective statements. (e.g., `[subject: Q4 code freeze] [predicate: is_approaching]`)
- `[type: sentiment]`: For emotional tone. (e.g., `[subject: Bob] [value: hesitant]`)
- `[type: contrasting_viewpoint]`: To link two differing opinions. (e.g., `[subject: Bob's Hesitation] [object: Alice's Proposal]`)
- `[type: rationale]`: To explain the "why" behind an action or statement. (e.g., `[subject: Alice's Proposal] [value: "market opportunity"]`)

Your entire output must be the structured Markdown log. Do not add any other commentary.
IMPORTANT: Process the *entire* chat log from beginning to end, right up to the `=== END OF LOG ===` marker. Do not stop early, even if the text contains words like "stop" or "end".

Raw Chat Log to Process:
{chat_log_string}
=== END OF LOG ===
"""
