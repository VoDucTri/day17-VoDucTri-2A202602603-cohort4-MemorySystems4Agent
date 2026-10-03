from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests with lower compact threshold."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dummy_model = ProviderConfig(
        provider="gemini",
        model_name="gemini-1.5-flash",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # Write initial profile
    initial_text = "# User Profile: test_user\n- **Tên**: DũngCT\n- **Nơi ở**: Đà Nẵng\n"
    file_path = store.write_text(user_id, initial_text)

    assert file_path.exists()
    assert store.file_size(user_id) > 0

    # Read back
    content = store.read_text(user_id)
    assert "DũngCT" in content
    assert "Đà Nẵng" in content

    # Edit profile (e.g. location update)
    edited = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert edited is True

    # Verify updated content
    updated_content = store.read_text(user_id)
    assert "Huế" in updated_content
    assert "Đà Nẵng" not in updated_content

    # Editing non-existent string returns False
    assert store.edit_text(user_id, "Hà Nội", "Sài Gòn") is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction when token threshold is exceeded."""
    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    thread_id = "thread-compact-test"

    # Append multiple messages with sufficient tokens
    for i in range(6):
        manager.append(thread_id, "user" if i % 2 == 0 else "assistant", f"Đây là nội dung thử nghiệm số {i} với độ dài đủ lớn để kích hoạt compact memory.")

    assert manager.compaction_count(thread_id) > 0
    ctx = manager.context(thread_id)
    assert ctx["summary"] != ""
    assert len(ctx["messages"]) <= 3


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "dungct_recall"
    # Thread 1: provide facts
    intro = "Chào bạn, mình tên là DũngCT và hiện tại đang ở Huế."
    baseline.reply(user_id, "thread-session-1", intro)
    advanced.reply(user_id, "thread-session-1", intro)

    # Thread 2: ask recall question in a completely fresh session
    query = "Mình tên gì và hiện tại mình đang ở đâu?"
    resp_base = baseline.reply(user_id, "thread-session-2", query)
    resp_adv = advanced.reply(user_id, "thread-session-2", query)

    # Baseline should NOT recall facts in a new thread
    base_text = resp_base["response"]
    assert "DũngCT" not in base_text or "chưa có thông tin" in base_text

    # Advanced should recall facts accurately from User.md
    adv_text = resp_adv["response"]
    assert "DũngCT" in adv_text
    assert "Huế" in adv_text


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "stress_user"
    thread_id = "long_thread_test"

    long_messages = [
        "Đây là đoạn tin tức số một về chương trình thám hiểm vũ trụ với nhiều thông tin chi tiết và lộ trình phát triển kéo dài qua nhiều năm.",
        "Đây là mẩu tin thứ hai về máy bay siêu thanh thế hệ mới với mục tiêu giảm tiếng ồn và tối ưu hiệu suất khí động học khi bay qua khu dân cư.",
        "Đây là bản tin thứ ba về dự báo khí tượng và biến đổi khí hậu toàn cầu kèm theo phân tích định lượng về xu hướng nhiệt độ và rủi ro thời tiết.",
        "Đây là báo cáo thứ tư về chính sách năng lượng sạch dài hạn và kế hoạch cân bằng giữa mở rộng nguồn cung và tiết kiệm tiêu thụ điện.",
        "Chúng ta cùng tổng hợp bốn bài toán trên dưới góc nhìn quản trị hệ thống và tối ưu hóa chi phí vận hành cho các dự án công nghệ lớn.",
        "Tiếp tục thảo luận sâu hơn về cách áp dụng kiến trúc memory compact để nén ngữ cảnh hội thoại mà không làm mất đi các thực thể cốt lõi.",
        "Đánh giá rủi ro khi nén hội thoại quá mức dẫn đến hiện tượng trôi thông tin hoặc mất mát các chi tiết kỹ thuật quan trọng của người dùng.",
        "Kết luận và đề xuất phương án giám sát độ tăng trưởng của memory file cũng như kiểm soát số lượng compaction trong chu kỳ hội thoại dài.",
    ]

    for msg in long_messages:
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)

    # Verify compaction actually triggered in advanced agent
    assert advanced.compaction_count(thread_id) > 0

    # Advanced prompt token load should be significantly lower than Baseline
    base_prompt_tokens = baseline.prompt_token_usage(thread_id)
    adv_prompt_tokens = advanced.prompt_token_usage(thread_id)
    assert adv_prompt_tokens < base_prompt_tokens
