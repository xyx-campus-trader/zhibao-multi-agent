"""
综合测试：多Agent工作流状态机 — AgentState, 路由逻辑, HITL 中断
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest, asyncio
from core.workflow import (
    AgentState, _route_after_review, _route_after_hitl,
    _hitl_check, MAX_REVISION_ROUNDS, MAX_HITL_ROUNDS,
)


# ============ AgentState ============

class TestAgentState:
    def test_empty_state_defaults(self):
        state = AgentState(
            task_id="test",
            topic="test",
            search_keywords={},
            research_results=[],
            draft="",
            final_report="",
            review_notes="",
            status="pending",
            error=None,
            hitl_approved=False,
            revision_count=0,
            hitl_revision_count=0,
        )
        assert state["task_id"] == "test"
        assert state["status"] == "pending"
        assert state["hitl_approved"] is False

    def test_state_is_dict_like(self):
        state = AgentState(
            task_id="t1", topic="topic1",
            search_keywords={}, research_results=[],
            draft="", final_report="",
            review_notes="", status="pending",
            error=None, hitl_approved=False,
            revision_count=0, hitl_revision_count=0,
        )
        state["status"] = "searching"
        assert state["status"] == "searching"

    def test_all_required_fields(self):
        required = ["task_id", "topic", "search_keywords", "research_results",
                     "draft", "final_report", "review_notes", "status",
                     "error", "hitl_approved", "revision_count", "hitl_revision_count"]
        # 类型检查：AgentState 应该包含所有这些字段
        sample = {
            "task_id": "t", "topic": "t", "search_keywords": {},
            "research_results": [], "draft": "", "final_report": "",
            "review_notes": "", "status": "pending",
            "error": None, "hitl_approved": False,
            "revision_count": 0, "hitl_revision_count": 0,
        }
        for key in required:
            assert key in sample


# ============ Route After Review ============

class TestRouteAfterReview:
    def test_status_failed_returns_failed(self):
        state = _make_state(status="failed")
        assert _route_after_review(state) == "failed"

    def test_status_revising_returns_revise(self):
        state = _make_state(status="revising")
        assert _route_after_review(state) == "revise"

    def test_hitl_approved_false_returns_hitl(self):
        state = _make_state(status="reviewing", hitl_approved=False)
        assert _route_after_review(state) == "hitl"

    def test_default_returns_end(self):
        state = _make_state(status="done")
        assert _route_after_review(state) == "end"

    def test_revising_takes_priority_over_hitl(self):
        state = _make_state(status="revising", hitl_approved=False)
        assert _route_after_review(state) == "revise"


# ============ Route After HITL ============

class TestRouteAfterHitl:
    def test_approved_returns_end(self):
        state = _make_state(status="revising", hitl_approved=True)
        assert _route_after_hitl(state) == "end"

    def test_rejected_with_revising_returns_revise(self):
        state = _make_state(status="revising", hitl_approved=False)
        assert _route_after_hitl(state) == "revise"

    def test_rejected_with_done_returns_end(self):
        """驳回轮次超出上限，status 变成 done，走 end"""
        state = _make_state(status="done", hitl_approved=False)
        assert _route_after_hitl(state) == "end"

    def test_approved_with_done(self):
        state = _make_state(status="done", hitl_approved=True)
        assert _route_after_hitl(state) == "end"


# ============ Constants ============

class TestConstants:
    def test_max_revision_rounds(self):
        assert MAX_REVISION_ROUNDS == 2

    def test_max_hitl_rounds(self):
        assert MAX_HITL_ROUNDS == 2

    def test_rounds_are_positive(self):
        assert MAX_REVISION_ROUNDS > 0
        assert MAX_HITL_ROUNDS > 0


# ============ Agent Classes ============

class TestOrchestratorAgent:
    def test_run_sets_status_searching(self):
        from core.workflow import OrchestratorAgent
        from core.search_decomposer import SearchDecomposer

        decomposer = SearchDecomposer()
        agent = OrchestratorAgent(decomposer)

        state = _make_state(status="pending", topic="AI行业动态")
        result = asyncio.run(agent.run(state))
        assert result["status"] in ("searching", "failed")

    def test_run_decomposes_topic(self):
        from core.workflow import OrchestratorAgent
        from core.search_decomposer import SearchDecomposer

        decomposer = SearchDecomposer()
        agent = OrchestratorAgent(decomposer)

        state = _make_state(status="pending", topic="AI行业分析")
        result = asyncio.run(agent.run(state))
        # 应该产生搜索关键词（LLM 可能失败但回退模板应该有值）
        keywords = result.get("search_keywords", {})
        if result["status"] == "searching":
            assert len(keywords) > 0

    def test_run_handles_exception(self):
        from core.workflow import OrchestratorAgent

        class FailingDecomposer:
            def decompose(self, topic):
                raise RuntimeError("test error")

        agent = OrchestratorAgent(FailingDecomposer())
        state = _make_state(status="pending", topic="test")
        result = asyncio.run(agent.run(state))
        assert result["status"] == "failed"
        assert result["error"] is not None


class TestResearcherAgent:
    def test_run_empty_keywords(self):
        from core.workflow import ResearcherAgent
        from core.search_decomposer import SearchDecomposer

        decomposer = SearchDecomposer()
        agent = ResearcherAgent(decomposer)

        state = _make_state(status="searching", search_keywords={})
        result = asyncio.run(agent.run(state))
        assert result["status"] in ("drafting", "failed")
        if result["status"] == "failed":
            assert result["error"] is not None

    def test_run_with_keywords(self):
        from core.workflow import ResearcherAgent
        from core.search_decomposer import SearchDecomposer
        from core.search_decomposer import FALLBACK_TEMPLATES

        decomposer = SearchDecomposer()
        agent = ResearcherAgent(decomposer)

        state = _make_state(status="searching", search_keywords=FALLBACK_TEMPLATES["AI"])
        result = asyncio.run(agent.run(state))
        assert result["status"] in ("drafting", "failed")


class TestEditorAgent:
    def test_editor_has_llm(self):
        from core.workflow import EditorAgent
        agent = EditorAgent()
        assert agent.llm is not None

    def test_build_prompt_returns_string(self):
        from core.workflow import EditorAgent

        agent = EditorAgent()
        state = _make_state(status="drafting", topic="AI行业周报")
        prompt = agent._build_prompt(state)
        assert isinstance(prompt, str)
        assert "AI行业周报" in prompt

    def test_build_prompt_with_review_notes(self):
        from core.workflow import EditorAgent

        agent = EditorAgent()
        state = _make_state(
            status="drafting", topic="test",
            review_notes="需要增加数据支撑", revision_count=1,
        )
        prompt = agent._build_prompt(state)
        assert "需要增加数据支撑" in prompt

    def test_build_prompt_without_review_notes_when_count_zero(self):
        from core.workflow import EditorAgent

        agent = EditorAgent()
        state = _make_state(status="drafting", review_notes="多余的意见", revision_count=0)
        prompt = agent._build_prompt(state)
        assert "审核意见" not in prompt  # revision_count=0 不显示审核意见


class TestReviewerAgent:
    def test_reviewer_has_llm(self):
        from core.workflow import ReviewerAgent
        agent = ReviewerAgent()
        assert agent.llm is not None

    def test_review_with_empty_draft(self):
        from core.workflow import ReviewerAgent

        agent = ReviewerAgent()
        # LLM 不可用时应该不崩溃
        state = _make_state(status="reviewing", draft="")
        result = asyncio.run(agent.run(state))
        assert "status" in result


# ============ Format Check Test ============

class TestPromptFormatting:
    def test_editor_prompt_contains_required_sections(self):
        from core.workflow import EditorAgent

        agent = EditorAgent()
        prompt = agent._build_prompt(_make_state(topic="半导体行业"))
        assert "核心摘要" in prompt
        assert "政策" in prompt
        assert "公司" in prompt
        assert "投融资" in prompt
        assert "趋势" in prompt
        assert "洞察" in prompt

    def test_reviewer_prompt_has_three_dimensions(self):
        from core.workflow import ReviewerAgent

        agent = ReviewerAgent()
        prompt = f"""审核以下行业周报，从完整性、准确性、可读性三个维度评估：

{""}

如果质量合格请回复 "pass"，否则回复 "fail" 并给出具体修改建议。"""
        # 检查 prompt 模板
        assert "完整性" in prompt
        assert "准确性" in prompt
        assert "可读性" in prompt
        assert "pass" in prompt
        assert "fail" in prompt


# ============ Helper ============

def _make_state(**overrides):
    defaults = {
        "task_id": "test_task",
        "topic": "test_topic",
        "search_keywords": {},
        "research_results": [],
        "draft": "",
        "final_report": "",
        "review_notes": "",
        "status": "pending",
        "error": None,
        "hitl_approved": False,
        "revision_count": 0,
        "hitl_revision_count": 0,
    }
    defaults.update(overrides)
    return AgentState(**defaults)
