# 智报多Agent自动生成系统

基于 Multi-Agent 协作的智能采编平台，面向行业研究场景，实现LLM驱动的自动化周报生成。

## 核心流程

```
用户输入主题 → Orchestrator 拆解搜索任务 → Researcher 并行检索
→ Editor 流式撰写周报 → Reviewer 质量审核 → 输出终稿
```

## 技术栈

- FastAPI + LangGraph + LangChain
- Ollama / OpenAI / DeepSeek 多模型接入
- Redis + PostgreSQL + 内存 三级持久化
- Docker Compose 容器化部署

## 快速启动

```bash
cp .env.example .env
pip install -r requirements.txt
python main.py
```

## API

- `POST /reports/generate` — 创建周报生成任务
- `GET /reports/{task_id}/status` — 查询任务状态
- `GET /reports/{task_id}/download` — 下载周报
