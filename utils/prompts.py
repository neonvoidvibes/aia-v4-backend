# utils/prompts.py

ENRICHMENT_PROMPT_TEMPLATE = """
**PRIMARY DIRECTIVE: You are a document conversion engine. Your only function is to convert a raw chat log into a single, complete, structured Markdown document. You MUST process the entire input log from start to finish without any truncation.**

**OUTPUT SPECIFICATION:**

1.  **YAML Frontmatter:**
    - The document MUST begin with a YAML frontmatter block.
    - This block MUST be enclosed by `---` delimiters.
    - It MUST contain a single key: `summary`, with a 1-2 sentence summary of the ENTIRE conversation from start to finish.

2.  **Conversation Turns:**
    - After the frontmatter, create a `### Turn X` heading for each message in the log.
    - Each turn section MUST contain `**Speaker:**` and `**Raw Text:**` fields.
    - The `**Raw Text:**` MUST be the verbatim content of the message.

3.  **Structured Tagging:**
    - Below the `**Raw Text:**` for each turn, you MUST add structured tags.
    - Tags MUST follow the format: `[type: tag_type] [key: value]...`
    - **Available Tag Types:** `decision`, `action_item`, `fact`, `sentiment`, `contrasting_viewpoint`, `rationale`.

**EXAMPLE OF CORRECT OUTPUT:**
```markdown
---
summary: "A user and an assistant discuss adding a feature, weighing market opportunity against timeline concerns."
---

### Turn 1
**Speaker:** User
**Raw Text:** I think we should add the new social sharing feature to the Q3 launch.
[type: action_item] [subject: team] [predicate: should_add] [object: social sharing feature]

### Turn 2
**Speaker:** Assistant
**Raw Text:** I'm concerned about the timeline. The code freeze is approaching.
[type: sentiment] [subject: Assistant] [value: concerned]
[type: fact] [subject: code freeze] [predicate: is_approaching]
```

**EXECUTION INSTRUCTIONS:**
- Adhere strictly to the `OUTPUT SPECIFICATION` and `EXAMPLE`.
- Do not add any commentary, apologies, or text outside of the specified Markdown structure.
- Your single task is to convert the following log. Begin now.

**Raw Chat Log to Process:**
{chat_log_string}
=== END OF LOG ===
"""

ENRICHMENT_CONTINUATION_PROMPT_TEMPLATE = """
**PRIMARY DIRECTIVE: Continue converting a raw chat log into a structured Markdown document, starting from Turn {start_turn_number}.**

**OUTPUT SPECIFICATION:**
- You MUST continue the `### Turn X` sequence, starting with the provided number.
- Each turn section MUST contain `**Speaker:**` and `**Raw Text:**` fields.
- You MUST add structured tags below the `**Raw Text:**` for each turn.
- Do NOT generate a YAML frontmatter block.

**EXAMPLE OF CORRECT CONTINUATION OUTPUT (if start_turn_number is 21):**
```markdown
### Turn 21
**Speaker:** User
**Raw Text:** Okay, that makes sense. Let's proceed.
[type: decision] [subject: User] [outcome: proceed]

### Turn 22
**Speaker:** Assistant
**Raw Text:** Great. I will update the project plan.
[type: action_item] [subject: Assistant] [predicate: will_update] [object: project plan]
```

**EXECUTION INSTRUCTIONS:**
- Adhere strictly to the `OUTPUT SPECIFICATION` and `EXAMPLE`.
- Do not add any commentary, apologies, or text outside of the specified Markdown structure.
- Your single task is to convert the following log chunk. Begin now.

**Raw Chat Log Chunk to Process:**
{chat_log_string}
=== END OF LOG ===
"""
