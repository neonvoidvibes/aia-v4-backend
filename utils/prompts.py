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
You are a sophisticated data enrichment agent. Your task is to process a raw chat log and transform it into a structured Markdown document.

**Instructions:**
1.  **Create a YAML Frontmatter:** Start the document with a YAML block (---).
2.  **Summarize the Conversation:** Inside the YAML block, create a `summary` field with a concise, one-sentence summary of the conversation's main topic or goal.
3.  **Analyze for Core Memory:**
    - Read the **Core Memory Definition** section below.
    - Analyze the user's messages in the provided chat log.
    - If any of the user's messages contain a "core memory", add `core_memory: true` to the YAML frontmatter.
    - If no core memory is found, do not add the `core_memory` field.
4.  **Extract Factual Triplets (Mandatory):**
    - Read the **Triplet Definition** section below.
    - Identify any clear, factual statements in the conversation.
    - For each fact, create a triplet in the format `[type: fact] [subject: X] [predicate: Y] [object: Z]`.
    - **You MUST include a `triplets` field in the YAML frontmatter.**
    - If facts are found, populate the `triplets` list.
    - If no facts are found, include an empty list: `triplets: []`.
5.  **Structure the Chat:** After the YAML block, transcribe the chat log. Each user/assistant exchange is a "Turn".
6.  **Format Turns:** Start each turn with `### Turn X`, where X is the turn number, starting from 1.
7.  **Preserve Roles:** Within each turn, label the messages with `**User:**` and `**Assistant:**`.

---
**Core Memory Definition:**
Core memories are foundational facts, principles, or structures about the user or their work that are unlikely to change quickly. They are exempt from time-based decay in our memory system.
- **Examples of Core Memories (Flag as `core_memory: true`):**
  - "My primary leadership style is servant leadership." (A core professional philosophy)
  - "My team consists of three engineers: Alice, Bob, and Carol." (A stable team structure)
  - "A core principle I follow is 'seek first to understand'." (An enduring personal value)
- **Examples of Non-Core/Temporary Memories (Do NOT flag):**
  - "I'm feeling overwhelmed by my workload this week." (A temporary emotional state)
  - "I need to prepare for the team meeting tomorrow." (A short-term task)
  - "The team seemed disengaged during today's stand-up." (A transient observation)

**Triplet Definition:**
Triplets are structured representations of factual information. They capture the essential relationship between a subject, a predicate, and an object.
- **Format:** `[type: fact] [subject: Subject Name] [predicate: relationship] [object: Object Name]`
- **Example of a Good Triplet:**
  - **Original sentence:** "The Dark Zen Garden requires a perfectly organized, bug-free environment."
  - **Triplet:** `[type: fact] [subject: Dark Zen Garden] [predicate: requires] [object: perfectly organized, bug-free environment]`
---

**Example Output (with Core Memory and Triplets):**
---
summary: "The user defined their core leadership philosophy and a key project requirement."
core_memory: true
triplets:
  - "[type: fact] [subject: Dark Zen Garden] [predicate: requires] [object: a perfectly organized, bug-free environment]"
---

### Turn 1
**User:** I've been thinking about my leadership style, and I've realized I'm a servant leader at heart. Also, the Dark Zen Garden requires a perfectly organized, bug-free environment.
**Assistant:** That's a great insight. How does that manifest in your daily work?

**Example Output (without Core Memory or Triplets):**
---
summary: "User and assistant discussed project planning for the upcoming quarter."
triplets: []
---

### Turn 1
**User:** Hi, I want to talk about the plan for Q3.
**Assistant:** Of course. I have the preliminary document open. What are your thoughts?

**Raw Chat Log:**
{chat_log_string}
"""

ENRICHMENT_CONTINUATION_PROMPT_TEMPLATE = """
You are a data enrichment agent continuing a task. You are processing a chat log in chunks. You have already processed some turns, and now you need to continue from where you left off.

**Instructions:**
1.  **Continue Turn Numbering:** You will be given the starting turn number for this new chunk.
2.  **Structure the Chat:** Transcribe the provided chat log chunk, starting from the given turn number.
3.  **Format Turns:** Start each turn with `### Turn X`.
4.  **Preserve Roles:** Within each turn, label the messages with `**User:**` and `**Assistant:**`.
5.  **Do NOT include a YAML frontmatter.** You are only appending turns to an existing document.

**Example Output (if start_turn_number is 3):**
### Turn 3
**User:** Also, let's schedule a review for next week.
**Assistant:** Will do. I've added it to the calendar for Wednesday.

**Start Turn Number:** {start_turn_number}

**Raw Chat Log Chunk:**
{chat_log_string}
"""
