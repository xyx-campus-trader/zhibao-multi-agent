"""
多Agent协作工作流 — LangGraph Orchestrator → Researcher → Editor → Reviewer
支持 HITL 人工审核中断与恢复、Reviewer 打回重写循环
"""
import logging
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config.settings import settings

logger = logging.getLogger(__name__)

MAX_REVISION_ROUNDS = 2
MAX_HITL_ROUNDS = 2  # HITL 驳回最大轮次，超出则强制通过


class AgentState(TypedDict):
    """LangGraph 共享状态"""
    task_id: str
    topic: str
    search_keywords: dict
    research_results: List[Dict]
    draft: str
    final_report: str
    review_notes: str
    status: str  # pending / searching / drafting / reviewing / done / failed
    error: Optional[str]
    hitl_approved: bool
    revision_count: int          # LLM自动修订轮次
    hitl_revision_count: int     # HITL 人工驳回轮次


# ============ Agent 节点 ============

class OrchestratorAgent:
    """Orchestrator：接收用户意图，拆解搜索任务"""

    def __init__(self, decomposer):
        self.decomposer = decomposer

    async def run(self, state: AgentState) -> AgentState:
        logger.info("[Orchestrator] Decomposing topic: %s", state["topic"])
        state["status"] = "searching"
        try:
            keywords = self.decomposer.decompose(state["topic"])
            state["search_keywords"] = keywords
        except Exception as e:
            state["error"] = str(e)
            state["status"] = "failed"
        return state


class ResearcherAgent:
    """Researcher：执行并行搜索"""

    def __init__(self, decomposer):
        self.decomposer = decomposer

    async def run(self, state: AgentState) -> AgentState:
        logger.info("[Researcher] Searching with keywords: %s",
                     list(state.get("search_keywords", {}).keys()))
        try:
            results = await self.decomposer.search_parallel(
                state.get("search_keywords", {})
            )
            state["research_results"] = results
            # 判空：四路全空时报错，不让 LLM 硬编
            if not results:
                state["error"] = "所有搜索均未返回结果，请检查网络或换个主题"
                state["status"] = "failed"
                logger.warning("[Researcher] All searches returned empty for topic: %s",
                               state.get("topic", ""))
            else:
                state["status"] = "drafting"
        except Exception as e:
            state["error"] = str(e)
            state["status"] = "failed"
        return state


class EditorAgent:
    """Editor：基于搜索结果撰写周报"""

    def __init__(self):
        from core.llm_factory import create_chat_model, invoke_with_retry
        self.llm = create_chat_model(temperature=0.3)
        self._invoke = invoke_with_retry

    async def run(self, state: AgentState) -> AgentState:
        logger.info("[Editor] Drafting report (auto_rev=%d, hitl_rev=%d)",
                     state.get("revision_count", 0),
                     state.get("hitl_revision_count", 0))
        try:
            prompt = self._build_prompt(state)
            response = self._invoke(self.llm, [HumanMessage(content=prompt)])
            state["draft"] = response.content
            state["status"] = "reviewing"
        except Exception as e:
            state["error"] = str(e)
            state["status"] = "failed"
        return state

    def _build_prompt(self, state: AgentState) -> str:
        revision_hint = ""
        if state.get("review_notes") and (state.get("revision_count", 0) > 0 or state.get("hitl_revision_count", 0) > 0):
            revision_hint = f"""

【审核意见，请据此修改】
{state['review_notes']}
"""
        return f"""请根据以下研究资料撰写一份行业周报：
主题：{state['topic']}

研究资料：
{state.get('research_results', [])}
{revision_hint}
要求：
1. 开头200字以内的核心摘要
2. 按政策、公司、投融资、趋势分章节
3. 引用具体数据和案例
4. 结尾给出本周行业洞察总结
"""


class ReviewerAgent:
    """Reviewer：质量审核"""

    def __init__(self):
        from core.llm_factory import create_chat_model, invoke_with_retry
        self.llm = create_chat_model(temperature=0)
        self._invoke = invoke_with_retry

    async def run(self, state: AgentState) -> AgentState:
        logger.info("[Reviewer] Reviewing draft")
        try:
            prompt = f"""审核以下行业周报，从完整性、准确性、可读性三个维度评估：

{state.get('draft', '')}

如果质量合格请回复 "pass"，否则回复 "fail" 并给出具体修改建议。"""
            response = self._invoke(self.llm, [HumanMessage(content=prompt)])
            review = response.content

            if "pass" in review.lower() and "fail" not in review.lower():
                state["final_report"] = state["draft"]
                state["review_notes"] = review
                state["status"] = "done"
            else:
                state["review_notes"] = review
                rev_count = state.get("revision_count", 0)
                if rev_count < MAX_REVISION_ROUNDS:
                    state["revision_count"] = rev_count + 1
                    state["status"] = "revising"
                else:
                    state["final_report"] = state["draft"]
                    state["status"] = "done"
                    logger.info("[Reviewer] Max revision rounds reached, accepting draft")
        except Exception as e:
            state["error"] = str(e)
            state["status"] = "failed"
        return state


# ============ LangGraph 工作流构建 ============

def build_workflow(
    decomposer,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> StateGraph:
    """构建多Agent协作工作流，含 Reviewer→Editor 修订循环 + HITL 中断"""
    orchestrator = OrchestratorAgent(decomposer)
    researcher = ResearcherAgent(decomposer)
    editor = EditorAgent()
    reviewer = ReviewerAgent()

    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", orchestrator.run)
    workflow.add_node("researcher", researcher.run)
    workflow.add_node("editor", editor.run)
    workflow.add_node("reviewer", reviewer.run)
    workflow.add_node("hitl_check", _hitl_check)

    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "researcher")
    workflow.add_edge("researcher", "editor")
    workflow.add_edge("editor", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {
            "revise": "editor",     # 打回重写
            "hitl": "hitl_check",   # 需人工审核
            "end": END,
            "failed": END,
        }
    )
    workflow.add_conditional_edges(
        "hitl_check",
        _route_after_hitl,
        {
            "revise": "editor",  # 驳回 → Editor 修改
            "end": END,          # 通过或强制通过 → 结束
        }
    )

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


async def _hitl_check(state: AgentState) -> AgentState:
    """HITL 人工审核节点 — 中断等待人工确认，驳回时带反馈回到 Editor"""
    decision = interrupt({
        "message": f"请审核周报草稿（主题：{state.get('topic', '')}）",
        "draft_preview": state.get("draft", "")[:500],
        "review_notes": state.get("review_notes", ""),
        "action": "approve_or_reject",
    })
    if isinstance(decision, dict) and decision.get("approved"):
        # 批准 → 输出终稿
        state["hitl_approved"] = True
        state["status"] = "done"
        state["final_report"] = state.get("draft", "")
    else:
        # 驳回 → 记录反馈，增加驳回轮次
        state["hitl_approved"] = False
        feedback = decision.get("feedback", "") if isinstance(decision, dict) else ""
        state["review_notes"] = feedback
        hitl_count = state.get("hitl_revision_count", 0) + 1
        state["hitl_revision_count"] = hitl_count

        if hitl_count < MAX_HITL_ROUNDS:
            # 未超上限 → 回到 Editor 修改
            state["status"] = "revising"
            logger.info(
                "[HITL] Rejected (round %d/%d), returning to editor with feedback: %s",
                hitl_count, MAX_HITL_ROUNDS, feedback[:80],
            )
        else:
            # 超出上限 → 强制通过
            state["status"] = "done"
            state["final_report"] = state.get("draft", "")
            state["hitl_approved"] = True
            logger.info(
                "[HITL] Max reject rounds reached, force accepting draft"
            )
    return state


def _route_after_review(state: AgentState) -> str:
    status = state.get("status", "")
    if status == "failed":
        return "failed"
    if status == "done":
        return "end"
    if status == "revising":
        return "revise"
    if state.get("hitl_approved") is False:
        return "hitl"
    return "end"


def _route_after_hitl(state: AgentState) -> str:
    """HITL 审核后的路由：通过则结束，驳回则回 Editor 修改"""
    if state.get("hitl_approved"):
        return "end"
    # 驳回：status 为 "revising" 才回 editor，否则结束（强制通过的情况）
    if state.get("status") == "revising":
        return "revise"
    return "end"
