"""
多Agent协作工作流 — LangGraph Orchestrator → Researcher → Editor → Reviewer
"""
import logging
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config.settings import settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """LangGraph 共享状态"""
    task_id: str
    topic: str  # 用户原始主题
    search_keywords: dict  # 搜索关键词
    research_results: List[Dict]  # 搜索结果
    draft: str  # 初稿
    final_report: str  # 终稿
    review_notes: str  # 审核意见
    status: str  # pending / searching / drafting / reviewing / done / failed
    error: Optional[str]
    hitl_approved: bool  # 是否需要人工审核


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
            state["status"] = "drafting"
        except Exception as e:
            state["error"] = str(e)
            state["status"] = "failed"
        return state


class EditorAgent:
    """Editor：基于搜索结果流式撰写周报"""

    def __init__(self):
        from core.llm_factory import create_chat_model
        self.llm = create_chat_model(temperature=0.3, streaming=True)

    async def run(self, state: AgentState) -> AgentState:
        logger.info("[Editor] Drafting report")
        try:
            prompt = self._build_prompt(state)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            state["draft"] = response.content
            state["status"] = "reviewing"
        except Exception as e:
            state["error"] = str(e)
            state["status"] = "failed"
        return state

    def _build_prompt(self, state: AgentState) -> str:
        return f"""请根据以下研究资料撰写一份行业周报：

主题：{state['topic']}

研究资料：
{state.get('research_results', [])}

要求：
1. 开头200字以内的核心摘要
2. 按政策、公司、投融资、趋势分章节
3. 引用具体数据和案例
4. 结尾给出本周行业洞察总结
"""


class ReviewerAgent:
    """Reviewer：质量审核"""

    def __init__(self):
        from core.llm_factory import create_chat_model
        self.llm = create_chat_model(temperature=0)

    async def run(self, state: AgentState) -> AgentState:
        logger.info("[Reviewer] Reviewing draft")
        try:
            prompt = f"""审核以下行业周报，从完整性、准确性、可读性三个维度评估：

{state.get('draft', '')}

输出：pass/fail + 修改建议"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            review = response.content

            if "pass" in review.lower():
                state["final_report"] = state["draft"]
                state["review_notes"] = review
                state["status"] = "done"
            else:
                state["review_notes"] = review
                state["status"] = "done"  # 首次版本直接输出，Reviewer意见作为备注
                state["final_report"] = state["draft"]
        except Exception as e:
            state["error"] = str(e)
            state["status"] = "failed"
        return state


# ============ LangGraph 工作流构建 ============

def build_workflow(
    decomposer,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> StateGraph:
    """构建多Agent协作工作流"""
    orchestrator = OrchestratorAgent(decomposer)
    researcher = ResearcherAgent(decomposer)
    editor = EditorAgent()
    reviewer = ReviewerAgent()

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("orchestrator", orchestrator.run)
    workflow.add_node("researcher", researcher.run)
    workflow.add_node("editor", editor.run)
    workflow.add_node("reviewer", reviewer.run)
    workflow.add_node("hitl_check", _hitl_check)

    # 连线：Orchestrator → Researcher → Editor → Reviewer → HITL → END
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "researcher")
    workflow.add_edge("researcher", "editor")
    workflow.add_edge("editor", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {"hitl": "hitl_check", "end": END, "failed": END}
    )
    workflow.add_edge("hitl_check", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


async def _hitl_check(state: AgentState) -> AgentState:
    """HITL人工审核节点"""
    if state.get("hitl_approved", True):
        state["status"] = "done"
    else:
        state["status"] = "pending"
    return state


def _route_after_review(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "failed"
    if state.get("hitl_approved") is False:
        return "hitl"
    return "end"
