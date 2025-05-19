import re
import os
import logging
from typing import Optional, Any
from anthropic import Anthropic, APIError # Import Anthropic specific error

logger = logging.getLogger(__name__)

# Pre-compile regex for efficiency
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
# Basic phone regex - this might need significant improvement for international/Swedish numbers
PHONE_REGEX = re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}(?:[-.\s]?\d{1,4})?")

def anonymize_transcript_chunk(
    text_chunk: str,
    llm_client: Optional[Anthropic], # Made client optional for fallback
    pii_model_name: str,
    language_hint: str = "sv"
) -> str:
    """
    Anonymizes PII in a text chunk.
    Uses regex for emails/phones, and an LLM for names and other PII.
    """
    if not text_chunk:
        return ""

    # Step 1: Regex-based filtering for emails and phone numbers
    try:
        text_after_regex = EMAIL_REGEX.sub("[EMAIL_REDACTED]", text_chunk)
        text_after_regex = PHONE_REGEX.sub("[PHONE_REDACTED]", text_after_regex)
        redacted_count_regex = text_chunk.count("@") - text_after_regex.count("@") # Rough count
        logger.debug(f"PII Filter: Regex applied. Approx {redacted_count_regex} email/phone patterns potentially redacted.")
    except Exception as e:
        logger.error(f"PII Filter: Error during regex redaction: {e}")
        text_after_regex = text_chunk # Fallback to original if regex fails badly

    # Step 2: LLM-based PII Redaction (for names primarily)
    if not llm_client:
        logger.warning("PII Filter: LLM client not provided. Skipping LLM-based PII redaction. Returning regex-filtered text.")
        return text_after_regex

    # Construct the prompt for the LLM
    # IMPORTANT: This prompt needs to be carefully crafted and tested.
    # It instructs the LLM to only output the redacted text.
    prompt = f"""Your sole task is to identify and redact specific Personally Identifiable Information (PII) from the provided text chunk. The text is primarily in {language_hint}.

PII Categories for Redaction:
1.  **Personal Names:** Replace clear instances of full names or first names used in a way that identifies a specific person with "[PERSON_REDACTED]". Be conservative: do NOT redact common nouns, verbs, or adjectives, even if they could coincidentally be names (e.g., "System", "Hope", "Mark"). Only redact if the context strongly suggests it's a person's name.
2.  **Email Addresses:** Replace with "[EMAIL_REDACTED_BY_AI]". (Primary regex should catch most of these).
3.  **Phone Numbers:** Replace with "[PHONE_REDACTED_BY_AI]". (Primary regex should catch most of these).
4.  **Physical Addresses:** Replace specific street addresses, cities if uniquely identifiable and not generic, etc., with "[ADDRESS_REDACTED]".
5.  **Organization Names:** Replace specific, non-generic organization names (e.g., "Acme Corp") with "[ORGANIZATION_REDACTED]". Do not redact generic terms like "the company" or "our team".

Output Requirements:
-   **You MUST return ONLY the processed text.**
-   Do NOT include any conversational introduction, explanation, apologies, or summaries.
-   Do NOT include the "Text to process:" or "---" markers from this prompt in your output.
-   Preserve original line breaks and general formatting of the input text.
-   If no PII from the categories above is found, return the original text chunk unmodified.

Text to process:
{text_after_regex}
""" # Removed the "---" and "Redacted text:" from the end of the prompt to avoid model confusion.

    try:
        logger.info(f"PII Filter: Calling LLM ({pii_model_name}) for name/PII redaction. Language hint: {language_hint}. Chunk size: {len(text_after_regex)}")
        
        # For Claude, the instruction is better within the user message.
        # System prompt is optional here.
        response = llm_client.messages.create(
            model=pii_model_name,
            max_tokens=len(text_after_regex) + 200, # Allow some overhead for placeholders
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1, # Low temperature for more deterministic output
        )

        if response.content and isinstance(response.content, list) and len(response.content) > 0:
            llm_redacted_text = response.content[0].text.strip()
            
            # Additional cleanup for potential "---" artifacts if the model still adds them despite instruction
            llm_redacted_text = re.sub(r"^-+$\n?", "", llm_redacted_text, flags=re.MULTILINE).strip()

            if llm_redacted_text != text_after_regex: # Check if LLM made changes
                 logger.info(f"PII Filter: LLM successfully redacted PII. Original (post-regex) len: {len(text_after_regex)}, LLM output len: {len(llm_redacted_text)}")
            else:
                 logger.debug("PII Filter: LLM made no changes to the regex-filtered text.")
            return llm_redacted_text
        else:
            logger.warning(f"PII Filter: LLM returned no content or unexpected response format. Response: {response}")
            return text_after_regex # Fallback

    except APIError as e: # More specific Anthropic error
        logger.error(f"PII Filter: Anthropic APIError during PII redaction: {e}", exc_info=True)
        return text_after_regex # Fallback
    except Exception as e:
        logger.error(f"PII Filter: Unexpected error during LLM PII redaction: {e}", exc_info=True)
        return text_after_regex # Fallback

if __name__ == '__main__':
    # Basic test (requires ANTHROPIC_API_KEY and model access)
    logging.basicConfig(level=logging.INFO)
    logger.info("Running PII filter test...")
    
    sample_client = None
    try:
        api_key_test = os.getenv("ANTHROPIC_API_KEY")
        if api_key_test:
            sample_client = Anthropic(api_key=api_key_test)
            logger.info("Test: Anthropic client initialized.")
        else:
            logger.warning("Test: ANTHROPIC_API_KEY not set. LLM redaction will be skipped in test.")
    except Exception as e:
        logger.error(f"Test: Failed to initialize Anthropic client: {e}")

    test_model = os.getenv("PII_REDACTION_MODEL_NAME", "claude-3-haiku-20240307")

    texts_to_test = [
        ("Svenska: Hej, mitt namn är Kalle Anka och min e-post är kalle.anka@exempel.se. Ring mig på 070-123 45 67.", "sv"),
        ("English: Hello, I'm Mickey Mouse, my email is mickey@example.com and phone +1 (555) 123-4567. I live at 123 Main St, Anytown.", "en"),
        ("Mixed: Hej David, can you call Maria Svensson at 08-7654321? Her email: maria.svensson@example.com. Also, Mr. John Smith is waiting.", "sv"),
        ("No PII: This is a regular sentence without any personal data.", "en"),
        ("Svenska utan PII: Det här är en vanlig mening.", "sv"),
        ("Lång text: Detta är en längre text med ett namn Eva Lundström och en e-post eva.lundstrom@example.com mitt i, samt telefonnummer 0700-000000. Vi hoppas att Eva är nöjd.", "sv")
    ]

    for text, lang in texts_to_test:
        print(f"\n--- Testing (lang: {lang}) ---")
        print(f"Original: {text}")
        redacted = anonymize_transcript_chunk(text, sample_client, test_model, language_hint=lang)
        print(f"Redacted: {redacted}")

    logger.info("PII filter test finished.")