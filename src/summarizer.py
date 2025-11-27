"""
Streaming multiturn conversation for document analysis.

Supports:
- Anthropic Claude (via anthropic library)
- OpenAI GPT (via openai library)
- Google Gemini (via google-genai library)
- Streaming responses
- Multiturn conversations with context
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Literal
from dataclasses import dataclass, field
import os


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class ConversationHistory:
    """Manages conversation history for context."""
    messages: list[Message] = field(default_factory=lambda: [])
    system_prompt: str | None = None

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history."""
        self.messages.append(Message(role="assistant", content=content))

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    def get_message_count(self) -> int:
        """Get number of messages in history."""
        return len(self.messages)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def stream_response(
        self,
        history: ConversationHistory,
        **kwargs: Any
    ) -> Iterator[str]:
        """
        Stream a response from the LLM.

        Args:
            history: Conversation history with context
            **kwargs: Additional provider-specific arguments

        Yields:
            Text chunks from the streaming response
        """
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (default: ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-5-20250929)
            max_tokens: Maximum tokens in response (default: 4096)
        """
        from anthropic import Anthropic

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens

    def stream_response(
        self,
        history: ConversationHistory,
        **kwargs: Any
    ) -> Iterator[str]:
        """Stream response from Claude."""
        from anthropic.types import MessageParam

        # Build messages in Anthropic format
        messages: list[MessageParam] = []
        for msg in history.messages:
            role_str: str = msg.role
            if role_str == "system":
                # Skip system messages as they go in the system parameter
                continue
            messages.append(
                MessageParam(role=role_str, content=msg.content)  # type: ignore[typeddict-item]
            )

        # Create streaming request with proper system parameter handling
        stream_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": messages,
        }

        if history.system_prompt is not None:
            stream_kwargs["system"] = history.system_prompt

        with self.client.messages.stream(**stream_kwargs) as stream:
            for text in stream.text_stream:
                yield text


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.1-2025-11-13",
        max_tokens: int = 4096
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (default: OPENAI_API_KEY env var)
            model: Model to use (default: gpt-5.1-2025-11-13)
            max_tokens: Maximum tokens in response (default: 4096)
        """
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens

    def stream_response(
        self,
        history: ConversationHistory,
        **kwargs: Any
    ) -> Iterator[str]:
        """Stream response from GPT."""
        from openai.types.chat import ChatCompletionMessageParam, ChatCompletionChunk

        # Build messages in OpenAI format
        messages: list[ChatCompletionMessageParam] = []

        # Add system prompt if provided
        if history.system_prompt:
            messages.append({
                "role": "system",
                "content": history.system_prompt
            })

        # Add conversation history
        for msg in history.messages:
            role_str: str = msg.role
            if role_str == "system":
                # System messages already added above
                continue
            messages.append({
                "role": role_str,  # type: ignore[typeddict-item]
                "content": msg.content
            })

        # Create streaming request
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )

        chunk: ChatCompletionChunk
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content is not None:
                yield delta_content


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3-pro-preview",
        max_tokens: int = 4096
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key (default: GOOGLE_API_KEY env var)
            model: Model to use (default: gemini-3-pro-preview)
            max_tokens: Maximum tokens in response (default: 4096)
        """
        from google import genai

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens

    def stream_response(
        self,
        history: ConversationHistory,
        **kwargs: Any
    ) -> Iterator[str]:
        """Stream response from Gemini."""
        # Build contents in Gemini format
        contents: list[str] = []

        # Add system instruction as first message if provided
        system_instruction: str | None = history.system_prompt

        # Add conversation history
        for msg in history.messages:
            # Gemini uses 'user' and 'model' roles
            role: str = "model" if msg.role == "assistant" else msg.role
            contents.append(f"{role}: {msg.content}")

        # Combine all messages into a single prompt
        combined_prompt: str = "\n\n".join(contents)

        # Create streaming request
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=combined_prompt,
            config={
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
                "system_instruction": system_instruction,
            }
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text


class DocumentSummarizer:
    """
    Streaming multiturn conversation for document analysis.

    Supports summarization, question answering, and analysis of extracted text.
    """

    def __init__(
        self,
        provider: LLMProvider,
        document_text: str,
        system_prompt: str | None = None
    ):
        """
        Initialize document summarizer.

        Args:
            provider: LLM provider to use (Anthropic, OpenAI, or Gemini)
            document_text: Extracted text from document
            system_prompt: Optional system prompt to customize behavior
        """
        self.provider = provider
        self.document_text = document_text

        # Initialize conversation history with document context
        default_system: str = (
            "You are an expert document analyst. Your role is to help users understand, "
            "analyze, and extract insights from documents.\n\n"
            "Guidelines:\n"
            "- Base all answers strictly on the provided document content\n"
            "- If information is not in the document, clearly state that\n"
            "- Provide specific quotes or references when relevant\n"
            "- Be concise but thorough in explanations\n"
            "- Identify key themes, patterns, and relationships in the content\n"
            "- For summaries, prioritize the most important information\n"
            "- Use clear structure (bullet points, numbered lists) when appropriate"
        )

        self.history = ConversationHistory(
            system_prompt=system_prompt or default_system
        )

        # Add document context as initial context
        self._add_document_context()

    def _add_document_context(self) -> None:
        """Add document text as context to conversation."""
        user_context_message: str = (
            f"Please analyze this document:\n\n"
            f"<document>\n{self.document_text}\n</document>\n\n"
            f"Be ready to answer questions, provide summaries, and analyze the content."
        )

        assistant_acknowledgment: str = (
            "I've reviewed the document and I'm ready to help you analyze it. "
            "I can provide summaries, answer specific questions, identify key themes, "
            "extract important details, or help you understand any part of the content. "
            "What would you like to know?"
        )

        self.history.add_user_message(user_context_message)
        self.history.add_assistant_message(assistant_acknowledgment)

    def ask(self, question: str, **kwargs: Any) -> Iterator[str]:
        """
        Ask a question about the document with streaming response.

        Args:
            question: Question to ask about the document
            **kwargs: Additional provider-specific arguments

        Yields:
            Text chunks from the streaming response
        """
        # Add user question to history
        self.history.add_user_message(question)

        # Get streaming response
        full_response: list[str] = []
        for chunk in self.provider.stream_response(self.history, **kwargs):
            full_response.append(chunk)
            yield chunk

        # Add complete response to history
        self.history.add_assistant_message("".join(full_response))

    def summarize(self, style: str = "concise", **kwargs: Any) -> Iterator[str]:
        """
        Generate a summary of the document with streaming response.

        Args:
            style: Summary style - 'concise', 'detailed', or 'bullet-points'
            **kwargs: Additional provider-specific arguments

        Yields:
            Text chunks from the streaming response
        """
        prompts: dict[str, str] = {
            "concise": (
                "Provide a concise summary of this document in 2-4 paragraphs. "
                "Focus on the main purpose, key points, and most important conclusions or takeaways."
            ),
            "detailed": (
                "Provide a comprehensive summary of this document that covers:\n"
                "1. The main purpose and context\n"
                "2. All major topics and their key details\n"
                "3. Important arguments, evidence, or data presented\n"
                "4. Conclusions and implications\n"
                "Organize your summary with clear sections."
            ),
            "bullet-points": (
                "Create a bullet-point summary of this document. Include:\n"
                "- Main topic/purpose (1-2 bullets)\n"
                "- Key points (5-8 bullets covering the most important information)\n"
                "- Main conclusions or takeaways (1-3 bullets)\n"
                "Keep each bullet concise and focused on essential information only."
            ),
        }

        prompt: str = prompts.get(style, prompts["concise"])
        yield from self.ask(prompt, **kwargs)

    def clear_history(self) -> None:
        """Clear conversation history while keeping document context."""
        self.history.clear()
        self._add_document_context()

    def get_message_count(self) -> int:
        """Get number of messages in conversation history."""
        return self.history.get_message_count()


def create_summarizer(
    document_text: str,
    provider_name: Literal["anthropic", "openai", "gemini"] = "anthropic",
    **provider_kwargs: Any
) -> DocumentSummarizer:
    """
    Factory function to create a DocumentSummarizer with specified provider.

    Args:
        document_text: Extracted text from document
        provider_name: LLM provider to use ('anthropic', 'openai', or 'gemini')
        **provider_kwargs: Arguments passed to provider constructor

    Returns:
        Configured DocumentSummarizer instance

    Example:
        >>> from extractors import extract_text
        >>> text = extract_text("document.pdf")
        >>> summarizer = create_summarizer(text, provider_name="anthropic")
        >>>
        >>> # Get a summary
        >>> for chunk in summarizer.summarize(style="concise"):
        ...     print(chunk, end="", flush=True)
        >>>
        >>> # Ask questions
        >>> for chunk in summarizer.ask("What are the main conclusions?"):
        ...     print(chunk, end="", flush=True)
    """
    provider: LLMProvider

    if provider_name == "anthropic":
        provider = AnthropicProvider(**provider_kwargs)
    elif provider_name == "openai":
        provider = OpenAIProvider(**provider_kwargs)
    elif provider_name == "gemini":
        provider = GeminiProvider(**provider_kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

    return DocumentSummarizer(
        provider=provider,
        document_text=document_text
    )
