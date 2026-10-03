from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                # Live mode via LangGraph / LangChain
                config = {"configurable": {"thread_id": thread_id}}
                result = self.langchain_agent.invoke(
                    {"messages": [("user", message)]},
                    config=config,
                )
                last_msg = result["messages"][-1]
                response_text = getattr(last_msg, "content", str(last_msg))
                session = self.sessions.setdefault(thread_id, SessionState())
                prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
                session.prompt_tokens_processed += prompt_tokens
                resp_tokens = estimate_tokens(response_text)
                session.token_usage += resp_tokens
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": response_text})
                return {
                    "response": response_text,
                    "tokens": resp_tokens,
                    "prompt_tokens": prompt_tokens,
                }
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for one thread."""
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate how much prompt context this baseline kept processing."""
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement deterministic offline behavior with strictly within-session memory."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Prompt context in baseline accumulates all raw messages in this session
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens

        session.messages.append({"role": "user", "content": message})

        # Baseline forgets facts across threads
        if len(session.messages) <= 1:
            response = "Xin chào! Mình chưa có thông tin về bạn trong phiên làm việc này. Bạn có thể chia sẻ thêm thông tin để mình hỗ trợ nhé."
        else:
            response = "Đã nhận thông tin trong phiên làm việc này. Mình sẽ tiếp tục hỗ trợ bạn theo mạch trao đổi hiện tại."

        resp_tokens = estimate_tokens(response)
        session.token_usage += resp_tokens
        session.messages.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "tokens": resp_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire LangGraph create_react_agent + MemorySaver."""
        if self.force_offline or not self.config.model.api_key:
            return None
        try:
            model = build_chat_model(self.config.model)
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            checkpointer = MemorySaver()
            return create_react_agent(model, tools=[], checkpointer=checkpointer)
        except Exception:
            return None
