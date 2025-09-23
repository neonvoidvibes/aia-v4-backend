# utils/llm_api_utils.py
import os
import logging
import threading
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from openai import OpenAI, APIError as OpenAI_APIError, APIStatusError as OpenAI_APIStatusError, APIConnectionError as OpenAI_APIConnectionError
from anthropic import Anthropic, APIStatusError as AnthropicAPIStatusError, AnthropicError, APIConnectionError as AnthropicAPIConnectionError
from groq import Groq, APIError as GroqAPIError, APIConnectionError as GroqAPIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception_type
import time

logger = logging.getLogger(__name__)

# --- Circuit Breaker and Custom Exception Classes (can be shared or defined here) ---
class CircuitBreaker:
    STATE_CLOSED = "CLOSED"
    STATE_OPEN = "OPEN"
    STATE_HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int, recovery_timeout: int, name: str):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self.failure_count = 0
        self.state = self.STATE_CLOSED
        self.last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        logger.info(f"Circuit Breaker '{name}' initialized: Threshold={failure_threshold}, Timeout={recovery_timeout}s")

    def is_open(self) -> bool:
        with self._lock:
            if self.state == self.STATE_OPEN:
                if self.last_failure_time and (time.time() - self.last_failure_time > self.recovery_timeout):
                    self.state = self.STATE_HALF_OPEN
                    logger.warning(f"Circuit Breaker '{self.name}': State changed to HALF_OPEN. Allowing a test request.")
                    return False
                return True
            return False

    def record_failure(self):
        with self._lock:
            if self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_OPEN
                self.last_failure_time = time.time()
                logger.error(f"Circuit Breaker '{self.name}': Failure in HALF_OPEN state. Tripping back to OPEN.")
            else:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = self.STATE_OPEN
                    self.last_failure_time = time.time()
                    logger.error(f"Circuit Breaker '{self.name}': Failure threshold ({self.failure_threshold}) reached. Tripping to OPEN state for {self.recovery_timeout}s.")
    
    def record_success(self):
        with self._lock:
            if self.state == self.STATE_HALF_OPEN:
                logger.info(f"Circuit Breaker '{self.name}': Success in HALF_OPEN state. Resetting to CLOSED.")
            elif self.state == self.STATE_CLOSED and self.failure_count > 0:
                 logger.info(f"Circuit Breaker '{self.name}': Success recorded, resetting failure count.")
            self.failure_count = 0
            self.state = self.STATE_CLOSED

class CircuitBreakerOpen(Exception):
    pass

# Instantiate global circuit breakers
anthropic_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60, name="Anthropic")
gemini_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60, name="Gemini")
openai_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60, name="OpenAI")
groq_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60, name="Groq")


# --- Retry Strategies ---
def log_retry_error(retry_state):
    logger.warning(f"Retrying API call (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}")

retry_strategy_anthropic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error,
    retry=(retry_if_exception_type((AnthropicAPIStatusError, AnthropicAPIConnectionError)))
)

retry_strategy_gemini = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error,
    retry=(retry_if_exception_type((
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        google_exceptions.InternalServerError
    )))
)

retry_strategy_openai = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error,
    retry=(retry_if_exception_type((OpenAI_APIStatusError, OpenAI_APIConnectionError)))
)

retry_strategy_groq = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error,
    retry=(retry_if_exception_type((GroqAPIError, GroqAPIConnectionError)))
)

# --- Thread-safe lock for Gemini's global config ---
gemini_config_lock = threading.Lock()


# --- LLM API Call Functions ---

@retry_strategy_anthropic
def _call_anthropic_stream_with_retry(model: str, max_tokens: int, system: str, messages: List[Dict[str, Any]], api_key: str):
    if anthropic_circuit_breaker.is_open():
        raise CircuitBreakerOpen(f"Assistant ({anthropic_circuit_breaker.name}) is temporarily unavailable due to upstream issues.")

    if not api_key:
        raise ValueError("API key for Anthropic is missing.")
    
    transient_anthropic_client = Anthropic(api_key=api_key)
    return transient_anthropic_client.messages.stream(model=model, max_tokens=max_tokens, system=system, messages=messages)

@retry_strategy_gemini
def _call_gemini_stream_with_retry(model_name: str, max_tokens: int, system_instruction: str, messages: List[Dict[str, Any]], api_key: str, temperature: float):
    if gemini_circuit_breaker.is_open():
        raise CircuitBreakerOpen(f"Assistant ({gemini_circuit_breaker.name}) is temporarily unavailable due to upstream issues.")

    if not api_key:
        raise ValueError("API key for Google Generative AI is missing.")

    with gemini_config_lock:
        original_global_key = os.getenv('GOOGLE_API_KEY')
        try:
            if api_key != original_global_key:
                genai.configure(api_key=api_key)
            
            # Define safety settings to be less restrictive, similar to the non-streaming version.
            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
            model = genai.GenerativeModel(
                model_name=model_name, 
                system_instruction=system_instruction,
                safety_settings=safety_settings  # Add safety settings here
            )
            generation_config = {"max_output_tokens": max_tokens, "temperature": temperature}
            gemini_messages = [{'role': 'model' if msg['role'] == 'assistant' else 'user', 'parts': [msg['content']]} for msg in messages]
            
            return model.generate_content(gemini_messages, stream=True, generation_config=generation_config)
        finally:
            if api_key != original_global_key and original_global_key:
                genai.configure(api_key=original_global_key)

@retry_strategy_openai
def _call_openai_stream_with_retry(model_name: str, max_tokens: int, system_instruction: str, messages: List[Dict[str, Any]], api_key: str, temperature: float):
    if openai_circuit_breaker.is_open():
        raise CircuitBreakerOpen(f"Assistant ({openai_circuit_breaker.name}) is temporarily unavailable due to upstream issues.")

    if not api_key:
        raise ValueError("API key for OpenAI is missing.")
    
    client = OpenAI(api_key=api_key)
    openai_messages = [{"role": "system", "content": system_instruction}] + messages
    
    stream = client.responses.stream(
        model=model_name,
        input=openai_messages,
        max_output_tokens=max_tokens
    )
    return stream

@retry_strategy_groq
def _call_groq_stream_with_retry(model_name: str, max_tokens: int, system_instruction: str, messages: List[Dict[str, Any]], api_key: str, temperature: float):
    if groq_circuit_breaker.is_open():
        raise CircuitBreakerOpen(f"Assistant ({groq_circuit_breaker.name}) is temporarily unavailable.")
    if not api_key:
        raise ValueError("API key for Groq is missing.")
    
    client = Groq(api_key=api_key)
    
    groq_messages = [{"role": "system", "content": system_instruction}] + messages

    stream = client.chat.completions.create(
        model=model_name,
        messages=groq_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )
    return stream

@retry_strategy_groq
def _call_groq_non_stream_with_retry(model_name: str, max_tokens: int, system_instruction: str, messages: List[Dict[str, Any]], api_key: str, temperature: float, reasoning_effort: Optional[str] = None):
    if groq_circuit_breaker.is_open():
        raise CircuitBreakerOpen(f"Assistant ({groq_circuit_breaker.name}) is temporarily unavailable.")
    if not api_key:
        raise ValueError("API key for Groq is missing.")

    client = Groq(api_key=api_key)

    groq_messages = [{"role": "system", "content": system_instruction}] + messages

    # Prepare request parameters
    request_params = {
        "model": model_name,
        "messages": groq_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }

    # Add reasoning_effort if specified
    if reasoning_effort is not None:
        request_params["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**request_params)

    logger.debug(f"Groq API response object: {response}")
    logger.debug(f"Groq choices length: {len(response.choices) if response.choices else 0}")
    if response.choices:
        logger.debug(f"Groq choice[0]: {response.choices[0]}")
        logger.debug(f"Groq message content: '{response.choices[0].message.content}'")
        logger.debug(f"Groq finish reason: {response.choices[0].finish_reason}")

    # For reasoning models, extract only the content, not the reasoning
    message = response.choices[0].message
    content = message.content

    # Log reasoning separately for debugging (but don't return it)
    if hasattr(message, 'reasoning') and message.reasoning:
        logger.debug(f"Reasoning (not returned): {message.reasoning}")

    logger.debug(f"Returning content: '{content}' (type: {type(content)})")
    return content

@retry_strategy_gemini
def _call_gemini_non_stream_with_retry(model_name: str, max_tokens: int, system_instruction: str, messages: List[Dict[str, Any]], api_key: str, temperature: float):
    if gemini_circuit_breaker.is_open():
        raise CircuitBreakerOpen(f"Assistant ({gemini_circuit_breaker.name}) is temporarily unavailable due to upstream issues.")

    if not api_key:
        raise ValueError("API key for Google Generative AI is missing.")

    with gemini_config_lock:
        original_global_key = os.getenv('GOOGLE_API_KEY')
        try:
            if api_key != original_global_key:
                genai.configure(api_key=api_key)
            
            # Define safety settings to be less restrictive, as title generation is a low-risk task.
            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
            model = genai.GenerativeModel(
                model_name=model_name, 
                system_instruction=system_instruction,
                safety_settings=safety_settings
            )
            generation_config = {"max_output_tokens": max_tokens, "temperature": temperature}
            gemini_messages = [{'role': 'model' if msg['role'] == 'assistant' else 'user', 'parts': [msg['content']]} for msg in messages]
            
            response = model.generate_content(gemini_messages, stream=False, generation_config=generation_config)
            
            # Safely extract text, even if the finish reason is MAX_TOKENS or other non-standard reasons.
            try:
                return response.text
            except ValueError:
                logger.warning(f"Could not use `response.text` accessor (finish_reason: {response.candidates[0].finish_reason}). Extracting content manually.")
                if response.parts:
                    return "".join(part.text for part in response.parts)
                # If there are no parts, it's a genuinely empty response or a safety block.
                logger.error("Gemini response was blocked or empty with no content parts.")
                return ""
        finally:
            if api_key != original_global_key and original_global_key:
                genai.configure(api_key=original_global_key)
