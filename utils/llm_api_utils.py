# utils/llm_api_utils.py
import os
import logging
import threading
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from openai import OpenAI, APIError as OpenAI_APIError, APIStatusError as OpenAI_APIStatusError, APIConnectionError as OpenAI_APIConnectionError
from anthropic import Anthropic, APIStatusError as AnthropicAPIStatusError, AnthropicError, APIConnectionError as AnthropicAPIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception_type

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
            
            model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
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
    
    transient_openai_client = OpenAI(api_key=api_key)
    openai_messages = [{"role": "system", "content": system_instruction}] + messages
    
    stream = transient_openai_client.chat.completions.create(
        model=model_name, messages=openai_messages, max_tokens=max_tokens, temperature=temperature, stream=True
    )
    return stream

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
            
            model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
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
