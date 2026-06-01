# 基于LangGraph的内容风险审核Agent

## 项目简介
基于字节跳动风控业务经验设计的内容审核Agent系统，使用LangGraph实现多风险域条件路由和Human-in-the-loop人工复核流程。

## 技术架构
- **框架**：LangGraph + LangChain
- **模型**：Deepseek API
- **核心模式**：StateGraph + 条件路由 + Human-in-the-loop

## 工作流程
用户输入评论
↓
identify_risk（风险类型识别）
↓
route_by_risk（路由1：safe直接结束）
↓
analyze_risk（深度分析+风险等级判断）
↓
route_by_level（路由2：高风险/涉政→人工审核）
↓
human_review（Human-in-the-loop）→ generate_report
↓
generate_report（结构化报告生成）
## 核心功能
- 四类风险识别：涉政/未成年人/违法违规/安全
- 两级条件路由：按风险类型 + 按风险等级
- Human-in-the-loop：高风险内容触发人工介入
- MemorySaver状态持久化：支持会话断点恢复
- 批量处理 + jsonl日志留存 + 风险分布统计
- LLM输出校验 + 自动重试（最多3次）

## 快速开始
```bash
pip install langchain langchain-openai langgraph python-dotenv
# 配置Deepseek API Key
python agent1.py
```

## 设计说明
选择风控审核场景的原因：在字节跳动担任大模型应用工程师期间，
参与过涉政、未成年人等多风险域的模型迭代工作，
对审核业务逻辑有直接认知，用LangGraph复现核心决策链路。
