"""Tests for src/summarizer.py and src/main.py — all providers mocked, no real API calls."""

import pytest
from unittest.mock import MagicMock, patch

import src.main as main_module
from src.summarizer import (
    Message,
    ConversationHistory,
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    DocumentSummarizer,
    create_summarizer,
)


# ---------------------------------------------------------------------------
# src/main.py
# ---------------------------------------------------------------------------

class TestMainModule:
    def test_main_function_runs(self, capsys):
        main_module.main()
        out = capsys.readouterr().out
        assert "document-scanner-summarizer" in out


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def test_add_user_message(self):
        history = ConversationHistory()
        history.add_user_message("Hello")
        assert history.messages[0].role == "user"
        assert history.messages[0].content == "Hello"

    def test_add_assistant_message(self):
        history = ConversationHistory()
        history.add_assistant_message("Hi there")
        assert history.messages[0].role == "assistant"
        assert history.messages[0].content == "Hi there"

    def test_clear(self):
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi")
        history.clear()
        assert history.messages == []

    def test_get_message_count(self):
        history = ConversationHistory()
        assert history.get_message_count() == 0
        history.add_user_message("Hello")
        assert history.get_message_count() == 1
        history.add_assistant_message("Hi")
        assert history.get_message_count() == 2

    def test_system_prompt_stored(self):
        history = ConversationHistory(system_prompt="Be helpful")
        assert history.system_prompt == "Be helpful"

    def test_no_system_prompt_defaults_none(self):
        history = ConversationHistory()
        assert history.system_prompt is None


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    def _make_stream(self, tokens: list[str]) -> MagicMock:
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(tokens)
        return mock_stream

    def test_init_explicit_key(self):
        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider = AnthropicProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        mock_cls.assert_called_once_with(api_key="test-key")

    def test_init_from_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider = AnthropicProvider()
        assert provider.api_key == "env-key"

    def test_init_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("anthropic.Anthropic"):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicProvider()

    def test_stream_response_basic(self):
        mock_stream = self._make_stream(["Hello", " world"])
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="fake")

        history = ConversationHistory(system_prompt="Be helpful")
        history.add_user_message("Hello")
        result = list(provider.stream_response(history))
        assert result == ["Hello", " world"]

    def test_stream_response_includes_system_param(self):
        mock_stream = self._make_stream(["Hi"])
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="fake")

        history = ConversationHistory(system_prompt="You are helpful")
        history.add_user_message("Hello")
        list(provider.stream_response(history))

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    def test_stream_response_no_system_param_when_none(self):
        mock_stream = self._make_stream(["Hi"])
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="fake")

        history = ConversationHistory()  # no system_prompt
        history.add_user_message("Hello")
        list(provider.stream_response(history))

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert "system" not in call_kwargs

    def test_stream_response_skips_system_role_messages(self):
        mock_stream = self._make_stream(["Response"])
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="fake")

        history = ConversationHistory()
        history.messages.append(Message(role="system", content="System msg"))
        history.add_user_message("Hello")
        list(provider.stream_response(history))

        call_kwargs = mock_client.messages.stream.call_args[1]
        messages = call_kwargs["messages"]
        assert all(m["role"] != "system" for m in messages)

    def test_stream_response_with_code_execution(self):
        mock_stream = self._make_stream(["result"])
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="fake", enable_code_execution=True)

        history = ConversationHistory()
        history.add_user_message("Run code")
        list(provider.stream_response(history))

        call_kwargs = mock_client.messages.stream.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert any(t.get("type") == "code_execution_20250825" for t in tools)
        assert "extra_headers" in call_kwargs
        assert "code-execution-2025-08-25" in call_kwargs["extra_headers"]["anthropic-beta"]

    def test_stream_response_with_web_search(self):
        mock_stream = self._make_stream(["result"])
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="fake", enable_web_search=True)

        history = ConversationHistory()
        history.add_user_message("Search something")
        list(provider.stream_response(history))

        call_kwargs = mock_client.messages.stream.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert any(t.get("type") == "web_search_20250305" for t in tools)

    def test_stream_response_with_web_fetch(self):
        mock_stream = self._make_stream(["result"])
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="fake", enable_web_fetch=True)

        history = ConversationHistory()
        history.add_user_message("Fetch a URL")
        list(provider.stream_response(history))

        call_kwargs = mock_client.messages.stream.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert any(t.get("type") == "web_fetch_20250910" for t in tools)
        assert "extra_headers" in call_kwargs
        assert "web-fetch-2025-09-10" in call_kwargs["extra_headers"]["anthropic-beta"]

    def test_stream_response_code_execution_and_web_fetch_share_beta_header(self):
        mock_stream = self._make_stream(["result"])
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(
                api_key="fake",
                enable_code_execution=True,
                enable_web_fetch=True,
            )

        history = ConversationHistory()
        history.add_user_message("Do stuff")
        list(provider.stream_response(history))

        call_kwargs = mock_client.messages.stream.call_args[1]
        beta = call_kwargs["extra_headers"]["anthropic-beta"]
        assert "code-execution-2025-08-25" in beta
        assert "web-fetch-2025-09-10" in beta

    def test_stream_response_kwargs_override_instance_flags(self):
        """enable_code_execution=True via kwargs should work even if not set in __init__."""
        mock_stream = self._make_stream(["result"])
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Use code")
        list(provider.stream_response(history, enable_code_execution=True))

        call_kwargs = mock_client.messages.stream.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert any(t.get("type") == "code_execution_20250825" for t in tools)


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    def _make_client(self, chunks: list) -> MagicMock:
        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = chunks
        return mock_client

    def _simple_chunk(self, text: str) -> MagicMock:
        chunk = MagicMock()
        chunk.text = text
        chunk.candidates = []
        return chunk

    def test_init_explicit_key(self):
        with patch("google.genai.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider = GeminiProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_init_from_env_var(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        with patch("google.genai.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider = GeminiProvider()
        assert provider.api_key == "env-key"

    def test_init_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with patch("google.genai.Client"):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GeminiProvider()

    def test_stream_response_basic(self):
        chunk = self._simple_chunk("Hello world")
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Hello")
        result = list(provider.stream_response(history))
        assert "Hello world" in result

    def test_role_mapping_assistant_to_model(self):
        chunk = self._simple_chunk("response")
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi there")
        history.add_user_message("Another question")
        list(provider.stream_response(history))

        call_kwargs = mock_client.models.generate_content_stream.call_args[1]
        contents = call_kwargs.get("contents", [])
        roles = [c.role for c in contents]
        assert "model" in roles
        assert "assistant" not in roles

    def test_stream_response_with_system_prompt(self):
        chunk = self._simple_chunk("response")
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory(system_prompt="You are helpful")
        history.add_user_message("Hello")
        list(provider.stream_response(history))

        call_kwargs = mock_client.models.generate_content_stream.call_args[1]
        config = call_kwargs.get("config")
        assert config is not None

    def test_thinking_config_enabled(self):
        chunk = self._simple_chunk("response")
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake", enable_thinking=True)

        history = ConversationHistory()
        history.add_user_message("Think hard")
        list(provider.stream_response(history))
        mock_client.models.generate_content_stream.assert_called_once()

    def test_thinking_via_kwargs(self):
        chunk = self._simple_chunk("response")
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Think")
        list(provider.stream_response(history, thinking="high"))
        mock_client.models.generate_content_stream.assert_called_once()

    def test_thinking_low_level(self):
        chunk = self._simple_chunk("response")
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Think low")
        list(provider.stream_response(history, thinking="low"))
        mock_client.models.generate_content_stream.assert_called_once()

    def test_grounding_tool(self):
        chunk = self._simple_chunk("response")
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake", enable_grounding=True)

        history = ConversationHistory()
        history.add_user_message("Search")
        list(provider.stream_response(history))
        mock_client.models.generate_content_stream.assert_called_once()

    def test_code_execution_tool(self):
        chunk = self._simple_chunk("response")
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake", enable_code_execution=True)

        history = ConversationHistory()
        history.add_user_message("Run code")
        list(provider.stream_response(history))
        mock_client.models.generate_content_stream.assert_called_once()

    def test_grounding_via_kwargs(self):
        chunk = self._simple_chunk("response")
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Search via kwargs")
        list(provider.stream_response(history, enable_grounding=True))
        mock_client.models.generate_content_stream.assert_called_once()

    def test_code_execution_via_kwargs(self):
        chunk = self._simple_chunk("response")
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Code via kwargs")
        list(provider.stream_response(history, enable_code_execution=True))
        mock_client.models.generate_content_stream.assert_called_once()

    def test_chunk_with_no_text_not_yielded(self):
        chunk = MagicMock()
        chunk.text = None
        chunk.candidates = []
        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Hello")
        result = list(provider.stream_response(history))
        assert result == []

    def test_thought_chunk_yielded(self):
        mock_part = MagicMock()
        mock_part.thought = "my deep thought"
        mock_part.executable_code = None
        mock_part.code_execution_result = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        chunk = MagicMock()
        chunk.text = None
        chunk.candidates = [mock_candidate]

        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Think")
        result = list(provider.stream_response(history))
        assert any("THINKING" in r for r in result)
        assert any("my deep thought" in r for r in result)

    def test_executable_code_chunk_yielded(self):
        mock_code_obj = MagicMock()
        mock_code_obj.language = "PYTHON"
        mock_code_obj.code = "print('hello')"

        mock_part = MagicMock()
        mock_part.thought = None
        mock_part.executable_code = mock_code_obj
        mock_part.code_execution_result = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        chunk = MagicMock()
        chunk.text = None
        chunk.candidates = [mock_candidate]

        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Run code")
        result = list(provider.stream_response(history))
        assert any("print('hello')" in r for r in result)

    def test_code_execution_result_chunk_yielded(self):
        mock_result_obj = MagicMock()
        mock_result_obj.outcome = "OK"
        mock_result_obj.output = "42"

        mock_part = MagicMock()
        mock_part.thought = None
        mock_part.executable_code = None
        mock_part.code_execution_result = mock_result_obj

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        chunk = MagicMock()
        chunk.text = None
        chunk.candidates = [mock_candidate]

        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Run code")
        result = list(provider.stream_response(history))
        assert any("42" in r for r in result)

    def test_candidate_with_no_content_parts(self):
        mock_candidate = MagicMock()
        mock_candidate.content = None  # no content

        chunk = MagicMock()
        chunk.text = None
        chunk.candidates = [mock_candidate]

        mock_client = self._make_client([chunk])

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")

        history = ConversationHistory()
        history.add_user_message("Hello")
        # Should not raise
        result = list(provider.stream_response(history))
        assert result == []


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    def test_init_explicit_key(self):
        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_init_from_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider = OpenAIProvider()
        assert provider.api_key == "env-key"

    def test_init_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("openai.OpenAI"):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIProvider()

    def test_stream_response(self):
        mock_chunk1 = MagicMock()
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.choices[0].delta.content = " world"
        mock_chunk3 = MagicMock()
        mock_chunk3.choices[0].delta.content = None  # should be skipped

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]

        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="fake")

        history = ConversationHistory(system_prompt="Be helpful")
        history.add_user_message("Hello")
        result = list(provider.stream_response(history))
        assert result == ["Hello", " world"]

    def test_stream_response_skips_system_role_messages(self):
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Response"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [mock_chunk]

        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="fake")

        history = ConversationHistory()
        history.messages.append(Message(role="system", content="System msg"))
        history.add_user_message("Hello")
        list(provider.stream_response(history))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert all(m.get("role") != "system" for m in messages if m.get("role") != "system")

    def test_stream_response_with_system_prompt(self):
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Hi"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [mock_chunk]

        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="fake")

        history = ConversationHistory(system_prompt="You are helpful")
        history.add_user_message("Hello")
        list(provider.stream_response(history))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"


# ---------------------------------------------------------------------------
# DocumentSummarizer
# ---------------------------------------------------------------------------

class TestDocumentSummarizer:
    def _mock_provider(self, tokens: list[str] | None = None) -> MagicMock:
        if tokens is None:
            tokens = ["Hello", " world"]
        provider = MagicMock(spec=[])  # not GeminiProvider
        provider.stream_response = MagicMock(
            side_effect=lambda *a, **kw: iter(tokens)
        )
        return provider

    def test_init_adds_two_context_messages(self):
        provider = self._mock_provider()
        summarizer = DocumentSummarizer(provider, "Sample document text")
        assert summarizer.get_message_count() == 2

    def test_init_default_system_prompt(self):
        provider = self._mock_provider()
        summarizer = DocumentSummarizer(provider, "text")
        assert summarizer.history.system_prompt is not None
        assert "document" in summarizer.history.system_prompt.lower()

    def test_init_custom_system_prompt(self):
        provider = self._mock_provider()
        summarizer = DocumentSummarizer(provider, "text", system_prompt="Custom")
        assert summarizer.history.system_prompt == "Custom"

    def test_ask_yields_chunks(self):
        provider = self._mock_provider(["chunk1", "chunk2"])
        summarizer = DocumentSummarizer(provider, "doc text")
        result = list(summarizer.ask("What is this?"))
        assert result == ["chunk1", "chunk2"]

    def test_ask_adds_user_and_assistant_to_history(self):
        provider = self._mock_provider(["response"])
        summarizer = DocumentSummarizer(provider, "doc text")
        initial_count = summarizer.get_message_count()
        list(summarizer.ask("Question"))
        assert summarizer.get_message_count() == initial_count + 2

    def test_ask_stores_full_response_in_history(self):
        provider = self._mock_provider(["Hello", " world"])
        summarizer = DocumentSummarizer(provider, "doc text")
        list(summarizer.ask("Question"))
        last_msg = summarizer.history.messages[-1]
        assert last_msg.role == "assistant"
        assert last_msg.content == "Hello world"

    def test_summarize_concise(self):
        provider = self._mock_provider(["concise summary"])
        summarizer = DocumentSummarizer(provider, "doc text")
        result = list(summarizer.summarize(style="concise"))
        assert result == ["concise summary"]
        call_history = provider.stream_response.call_args[0][0]
        last_user_msg = [m for m in call_history.messages if m.role == "user"][-1]
        assert "concise" in last_user_msg.content.lower()

    def test_summarize_detailed(self):
        provider = self._mock_provider(["detailed summary"])
        summarizer = DocumentSummarizer(provider, "doc text")
        result = list(summarizer.summarize(style="detailed"))
        assert result == ["detailed summary"]

    def test_summarize_bullet_points(self):
        provider = self._mock_provider(["• point 1"])
        summarizer = DocumentSummarizer(provider, "doc text")
        result = list(summarizer.summarize(style="bullet-points"))
        assert result == ["• point 1"]

    def test_summarize_unknown_style_falls_back_to_concise(self):
        provider = self._mock_provider(["summary"])
        summarizer = DocumentSummarizer(provider, "doc text")
        result = list(summarizer.summarize(style="unknown-style"))
        assert result == ["summary"]
        call_history = provider.stream_response.call_args[0][0]
        last_user_msg = [m for m in call_history.messages if m.role == "user"][-1]
        assert "concise" in last_user_msg.content.lower()

    def test_clear_history_resets_to_two_messages(self):
        provider = self._mock_provider(["response"])
        summarizer = DocumentSummarizer(provider, "doc text")
        list(summarizer.ask("Question 1"))
        assert summarizer.get_message_count() > 2
        summarizer.clear_history()
        assert summarizer.get_message_count() == 2

    def test_get_message_count(self):
        provider = self._mock_provider()
        summarizer = DocumentSummarizer(provider, "doc text")
        assert summarizer.get_message_count() == 2

    def test_gemini_code_execution_adds_code_instructions(self):
        with patch("google.genai.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            gemini_provider = GeminiProvider(api_key="fake", enable_code_execution=True)

        summarizer = DocumentSummarizer(gemini_provider, "my doc text")
        first_user_msg = summarizer.history.messages[0]
        assert "document_text" in first_user_msg.content

    def test_gemini_no_code_execution_normal_context(self):
        with patch("google.genai.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            gemini_provider = GeminiProvider(api_key="fake", enable_code_execution=False)

        summarizer = DocumentSummarizer(gemini_provider, "my doc text")
        first_user_msg = summarizer.history.messages[0]
        # Without code execution, the code variable instructions should not be present
        assert "IMPORTANT: If you need to use code execution" not in first_user_msg.content


# ---------------------------------------------------------------------------
# create_summarizer factory
# ---------------------------------------------------------------------------

class TestCreateSummarizer:
    def test_anthropic_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            summarizer = create_summarizer("text", provider_name="anthropic")
        assert isinstance(summarizer, DocumentSummarizer)
        from src.summarizer import AnthropicProvider
        assert isinstance(summarizer.provider, AnthropicProvider)

    def test_openai_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake")
        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            summarizer = create_summarizer("text", provider_name="openai")
        assert isinstance(summarizer, DocumentSummarizer)
        from src.summarizer import OpenAIProvider
        assert isinstance(summarizer.provider, OpenAIProvider)

    def test_gemini_provider(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake")
        with patch("google.genai.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            summarizer = create_summarizer("text", provider_name="gemini")
        assert isinstance(summarizer, DocumentSummarizer)
        assert isinstance(summarizer.provider, GeminiProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_summarizer("text", provider_name="unknown")  # type: ignore[arg-type]

    def test_kwargs_forwarded_to_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            summarizer = create_summarizer(
                "text",
                provider_name="anthropic",
                enable_code_execution=True,
            )
        from src.summarizer import AnthropicProvider
        assert isinstance(summarizer.provider, AnthropicProvider)
        assert summarizer.provider.enable_code_execution is True
