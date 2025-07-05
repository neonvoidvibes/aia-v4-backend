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
