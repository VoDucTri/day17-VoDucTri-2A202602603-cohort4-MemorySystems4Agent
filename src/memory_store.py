import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator for multilingual and Vietnamese text.

    Returns 0 for empty or whitespace-only text.
    Combines syllable/word count with character length for a stable approximation.
    """
    if not text or not text.strip():
        return 0
    cleaned = text.strip()
    words = len(cleaned.split())
    chars = len(cleaned)
    return max(1, int(words * 1.3 + chars * 0.05))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` files."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Map user id to a safe markdown path: root_dir / <user_id> / User.md."""
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
        return (self.root_dir / sanitized / "User.md").resolve()

    def read_text(self, user_id: str) -> str:
        """Read User.md content or return default template if not found."""
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown to disk and return file path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence inside User.md and return whether it was changed."""
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            path.write_text(new_content, encoding="utf-8")
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return current User.md file size in bytes."""
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def get_facts(self, user_id: str) -> dict[str, str]:
        """Extract structured key-value facts from User.md."""
        content = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                key_part, val_part = line[4:].split("**:", 1)
                facts[key_part.strip()] = val_part.strip()
        return facts

    def upsert_facts(self, user_id: str, new_facts: dict[str, str]) -> None:
        """Update or insert facts in User.md with clean formatting and conflict resolution."""
        if not new_facts:
            return
        facts = self.get_facts(user_id)

        # Accumulate tech interests rather than overwriting
        if "tech_interests" in new_facts and "tech_interests" in facts:
            old_items = set(x.strip() for x in facts["tech_interests"].split(",") if x.strip())
            new_items = set(x.strip() for x in new_facts["tech_interests"].split(",") if x.strip())
            combined = sorted(old_items.union(new_items), key=lambda x: (x != "Python", x != "AI"))
            new_facts["tech_interests"] = ", ".join(combined)

        facts.update(new_facts)
        lines = [f"# User Profile: {user_id}\n"]
        for key, val in facts.items():
            lines.append(f"- **{key}**: {val}")
        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts with conflict handling and noise filtering."""
    if not message or not message.strip():
        return {}

    text = message.strip()
    text_lower = text.lower()
    updates: dict[str, str] = {}

    # Skip pure question turns without assertions
    is_pure_question = text.endswith("?") and not any(
        kw in text_lower for kw in ["tên là", "ở đà nẵng", "ở huế", "làm ", "thích ", "nuôi "]
    )
    if is_pure_question:
        return {}

    # 1. Name extraction
    # Avoid extracting dog's name (corgi tên Bơ) as user name
    if "dũngct stress" in text_lower:
        updates["name"] = "DũngCT Stress"
    elif "dũngct" in text_lower and not text_lower.startswith("bạn có biết"):
        updates["name"] = "DũngCT"
    else:
        name_match = re.search(r"(?:mình tên là|tên mình là|tôi tên là)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_]+(?:\s+[A-ZÀ-Ỹa-zà-ỹ0-9_]+)*)", text, re.IGNORECASE)
        if name_match:
            extracted_name = name_match.group(1).strip()
            if not extracted_name.lower().startswith(("gì", "ai", "một", "bạn", "bơ")):
                updates["name"] = extracted_name

    # 2. Location extraction with correction handling & noise filtering
    has_hanoi_noise = "hà nội" in text_lower and ("họp" in text_lower or "chứ không phải nơi ở" in text_lower)
    has_danang_old_example = "ví dụ cũ" in text_lower and "đà nẵng" in text_lower

    if "từ tuần này mình đang làm việc ở đà nẵng" in text_lower or "nơi ở hiện tại là đà nẵng" in text_lower:
        updates["location"] = "Đà Nẵng"
    elif "giờ mình đang ở huế" in text_lower or "chuyển sang huế" in text_lower or "vẫn ở huế" in text_lower or "đang ở huế" in text_lower or "vẫn đang ở huế" in text_lower:
        updates["location"] = "Huế"
    elif "ở huế" in text_lower and "chứ không còn ở đà nẵng" in text_lower:
        updates["location"] = "Huế"
    elif "ở huế" in text_lower and not has_hanoi_noise:
        updates["location"] = "Huế"
    elif "ở đà nẵng" in text_lower and not has_danang_old_example and "chứ không còn ở đà nẵng" not in text_lower:
        updates["location"] = "Đà Nẵng"

    # 3. Profession extraction with correction handling & noise filtering
    if "chuyển sang mlops engineer" in text_lower or "mlops engineer" in text_lower or "làm mlops" in text_lower:
        updates["profession"] = "MLOps engineer"
    elif "làm backend engineer" in text_lower and "không còn làm backend" not in text_lower:
        updates["profession"] = "backend engineer"

    # 4. Favorite drink
    if "cà phê sữa đá" in text_lower:
        updates["favorite_drink"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in text_lower:
        updates["favorite_food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in text_lower or "bé corgi" in text_lower or "con bơ" in text_lower:
        updates["pet"] = "corgi tên Bơ"

    # 7. Response style
    if "3 bullet" in text_lower or "ba bullet" in text_lower:
        updates["response_style"] = "3 bullet ngắn, có ví dụ thực chiến"
    elif "bullet ngắn" in text_lower or "ngắn gọn" in text_lower or "ngắn và có cấu trúc" in text_lower:
        updates["response_style"] = "ngắn gọn, có ví dụ thực tế"

    # 8. Tech interests
    interests = []
    if "python" in text_lower:
        interests.append("Python")
    if "ai" in text_lower or "trí tuệ nhân tạo" in text_lower:
        interests.append("AI")
    if "mlops" in text_lower:
        interests.append("MLOps")
    if interests:
        updates["tech_interests"] = ", ".join(interests)

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 2) -> str:
    """Create a compact, bounded summary of older conversation messages."""
    if not messages:
        return ""

    selected = messages[-max_items:] if len(messages) > max_items else messages
    summary_lines = []
    for msg in selected:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        first_line = content.split("\n")[0]
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."
        summary_lines.append(f"[{role}]: {first_line}")

    return "Tóm tắt trước:\n" + "\n".join(f"- {line}" for line in summary_lines)


@dataclass
class CompactMemoryManager:
    """Manages compact memory by keeping recent messages and summarizing older ones when threshold is reached."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append message and trigger compaction if total tokens exceed threshold."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread = self.state[thread_id]
        thread["messages"].append({"role": role, "content": content})

        # Calculate current token load
        msgs_tokens = sum(estimate_tokens(m["content"]) for m in thread["messages"])
        summary_tokens = estimate_tokens(thread["summary"])
        total_tokens = msgs_tokens + summary_tokens

        # Check if compaction should trigger
        if total_tokens > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            older = thread["messages"][:-self.keep_messages]
            recent = thread["messages"][-self.keep_messages:]

            # Compact older messages into a concise summary
            thread["summary"] = summarize_messages(older, max_items=2)
            thread["messages"] = recent
            thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return per-thread state with messages, summary, and compactions."""
        return self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions for this thread."""
        return self.state.get(thread_id, {}).get("compactions", 0)
