# aia-v4-backend/utils/prompts.py

CORE_MEMORY_CLASSIFIER_PROMPT = """
You are a highly intelligent text analysis agent. Your task is to determine if the following text snippet contains a "core memory" about the user.

**Context:** Core memories are foundational facts, principles, or structures that should not lose relevance over time. We are flagging them so they are exempt from a time-decay algorithm that reduces the importance of older memories. Temporary states, transient opinions, or short-term tasks should NOT be flagged.

**Instructions:**
Analyze the user's text below. Does it contain a core, long-term, foundational fact, principle, or structure related to the user or their work?

**Examples of Core Memories (Flag as 'true'):**
- "My primary leadership style is servant leadership." (A core professional philosophy)
- "I've identified that my key development area for this year is strategic thinking." (A long-term personal development goal)
- "My team consists of three engineers: Alice, Bob, and Carol." (A stable team structure)
- "Our organization follows a matrix management structure." (A fundamental organizational design)
- "A core principle I follow is 'seek first to understand, then to be understood'." (An enduring personal or professional value)
- "I believe that complex systems are best influenced through small, high-leverage interventions." (A foundational belief about systems/complexity)

**Examples of Non-Core/Temporary Memories (Flag as 'false'):**
- "I'm feeling overwhelmed by my workload this week." (A temporary emotional/physical state)
- "I need to prepare for the team meeting tomorrow." (A short-term task)
- "I'm currently reading a book on systems thinking." (A current activity, not the core belief itself)
- "The team seemed disengaged during today's stand-up." (A transient observation)

Based on this, analyze the following text. Respond with only the word 'true' or 'false'.

**User Text:**
"{user_text}"
"""

ENRICHMENT_PROMPT_TEMPLATE = """
You are a sophisticated data enrichment agent. Your task is to process a raw chat log and transform it into a structured, machine-readable JSON format.

**Instructions:**
1.  **Summarize the Conversation:** Create a concise, one-sentence summary of the entire conversation's main topic or goal.
2.  **Identify Key Entities:** Extract and categorize key entities mentioned (e.g., people, projects, technologies, key dates).
3.  **Extract Action Items:** List any explicit tasks, to-dos, or action items mentioned.
4.  **Identify Key Decisions:** Document any significant decisions made during the conversation.
5.  **Detect User Sentiment:** Analyze the overall sentiment of the user throughout the conversation (e.g., positive, negative, neutral, mixed).
6.  **Format as JSON:** Present the entire output as a single, valid JSON object. Do not include any text or formatting outside of the JSON structure.

**Raw Chat Log:**
```json
{chat_log_json}
```

**JSON Output Structure:**
{{
  "summary": "A brief, one-sentence summary of the conversation.",
  "entities": {{
    "people": ["Person A", "Person B"],
    "projects": ["Project X"],
    "technologies": ["Python", "React"],
    "dates": ["2023-10-27"]
  }},
  "action_items": [
    "Schedule a follow-up meeting.",
    "Send the report to the team."
  ],
  "decisions": [
    "The team will adopt the new framework for Project X."
  ],
  "user_sentiment": "neutral"
}}
"""

ENRICHMENT_CONTINUATION_PROMPT_TEMPLATE = """
You are a data enrichment agent continuing a task. You previously processed a large chat log and generated a partial JSON output. The process was interrupted.

Your task is to complete the JSON object based on the remaining part of the chat log.

**Instructions:**
1.  **Review the Partial JSON:** You will be given the JSON you have generated so far.
2.  **Analyze the Remaining Log:** You will be given the rest of the raw chat log.
3.  **Complete the JSON:** Continue populating the JSON object based on the new information. Do NOT repeat information already present in the partial JSON. Your output should be only the closing part of the JSON object, starting from the first incomplete or new field.

**Partial JSON Generated So Far:**
```json
{partial_json}
```

**Remaining Raw Chat Log:**
```json
{remaining_chat_log_json}
```

**Your Output:**
Continue the JSON from where it left off. For example, if the partial JSON ended mid-way through "action_items", your output might look like this:
```json
    "Send the report to the team."
  ],
  "decisions": [
    "The team will adopt the new framework for Project X."
  ],
  "user_sentiment": "neutral"
}}
```
"""
