import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

# Ensure UTF-8 output on Windows consoles
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return fraction of expected facts present in answer (0.0 to 1.0)."""
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in answer_lower)
    return round(matches / len(expected), 2)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Calculate a lightweight quality score for offline answers."""
    rec = recall_points(answer, expected)
    # Penalize empty or evasive answers
    if "chưa có thông tin" in answer.lower() or not answer.strip():
        return 0.1

    score = rec * 0.7
    # Reward structured or bulleted responses
    if any(bullet in answer for bullet in ["- ", "• ", "* "]):
        score += 0.2
    elif len(answer) > 20:
        score += 0.15

    # Reward conciseness (between 30 and 400 chars)
    if 30 <= len(answer) <= 400:
        score += 0.1

    return round(min(1.0, max(0.1, score)), 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over multiple conversations and return a BenchmarkRow."""
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    # Get initial memory size if agent supports persistent storage
    init_mem_size = 0
    if hasattr(agent, "profile_store"):
        init_mem_size = sum(f.stat().st_size for f in agent.profile_store.root_dir.glob("**/*") if f.is_file())

    for conv in conversations:
        user_id = conv["user_id"]
        conv_id = conv.get("id", "conv")
        main_thread = f"{conv_id}-main"

        # 1. Run all conversation turns in the main thread
        for turn in conv.get("turns", []):
            agent.reply(user_id=user_id, thread_id=main_thread, message=turn)

        # 2. Ask recall questions in FRESH threads to test cross-session memory
        for q_idx, q in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{conv_id}-recall-{q_idx}"
            res = agent.reply(user_id=user_id, thread_id=recall_thread, message=q["question"])
            ans = res.get("response", "")
            rec = recall_points(ans, q.get("expected_contains", []))
            qual = heuristic_quality(ans, q.get("expected_contains", []))
            recall_scores.append(rec)
            quality_scores.append(qual)

    # Compute aggregate metrics
    if isinstance(agent, BaselineAgent):
        total_agent_tokens = sum(s.token_usage for s in agent.sessions.values())
        total_prompt_tokens = sum(s.prompt_tokens_processed for s in agent.sessions.values())
        compactions = 0
        final_mem_size = 0
    else:
        total_agent_tokens = sum(agent.thread_tokens.values())
        total_prompt_tokens = sum(agent.thread_prompt_tokens.values())
        compactions = sum(agent.compact_memory.compaction_count(tid) for tid in agent.compact_memory.state.keys())
        final_mem_size = sum(f.stat().st_size for f in agent.profile_store.root_dir.glob("**/*") if f.is_file())

    mem_growth = max(0, final_mem_size - init_mem_size)
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 4),
        response_quality=round(avg_quality, 4),
        memory_growth_bytes=mem_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean Markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            f"{r.memory_growth_bytes:,}",
            r.compactions,
        ])
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both standard and long-context stress benchmarks."""
    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir)

    std_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    std_convs = load_conversations(std_path)
    stress_convs = load_conversations(stress_path)

    # 1. Standard Benchmark
    # Clean state dir for clean isolation
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 40)
    print("      STANDARD BENCHMARK (data/conversations.json)")
    print("=" * 40)
    baseline_std = BaselineAgent(config, force_offline=True)
    advanced_std = AdvancedAgent(config, force_offline=True)

    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, std_convs, config),
        run_agent_benchmark("Advanced", advanced_std, std_convs, config),
    ]
    print(format_rows(std_rows))

    # 2. Long-Context Stress Benchmark
    # Clean state dir for stress isolation
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 40)
    print("   LONG-CONTEXT STRESS BENCHMARK (data/advanced_long_context.json)")
    print("=" * 40)
    baseline_stress = BaselineAgent(config, force_offline=True)
    advanced_stress = AdvancedAgent(config, force_offline=True)

    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]
    print(format_rows(stress_rows))
    print("\n")


if __name__ == "__main__":
    main()
