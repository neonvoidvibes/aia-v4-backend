# utils/prompts.py

ENRICHMENT_PROMPT_TEMPLATE = """
**CRITICAL INSTRUCTION: YOUR TASK IS TO CONVERT A RAW CHAT LOG INTO A STRUCTURED MARKDOWN DOCUMENT.**

**Rule 1: Output Format**
Your entire output MUST be a single Markdown document. It MUST start with a **YAML frontmatter block** (YAML code enclosed by `---` lines) containing a `summary` field. After the frontmatter, create a `### Turn X` section for each message. Each turn MUST contain `**Speaker:**` and `**Raw Text:**` fields.

**Rule 2: Content Processing**
You MUST process the *entire* chat log provided, from the beginning to the `=== END OF LOG ===` marker. Do NOT truncate, summarize, or omit any part of the conversation.

**Rule 3: Tagging**
For each turn, you MUST add structured tags below the `**Raw Text:**` to capture key information. Use the format `[type: tag_type] [key: value]...`.

**Available Tag Types:**
- `[type: decision]`: For explicit decisions made.
- `[type: action_item]`: For tasks assigned.
- `[type: fact]`: For objective statements.
- `[type: sentiment]`: For emotional tone.
- `[type: contrasting_viewpoint]`: To link two differing opinions.
- `[type: rationale]`: To explain the "why" behind an action or statement.

**EXAMPLE OUTPUT:**
```markdown
---
summary: "The user proposes a new feature, and the assistant raises a concern about the project timeline."
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

**DO NOT ADD ANY COMMENTARY. YOUR RESPONSE MUST BE ONLY THE MARKDOWN DOCUMENT.**

**Raw Chat Log to Process:**
{chat_log_string}
=== END OF LOG ===
"""
