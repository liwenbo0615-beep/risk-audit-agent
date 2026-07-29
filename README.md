# 基于 LangGraph 的多风险域内容审核 Agent

一个用 LangGraph 编排的 UGC 内容安全审核 Agent。采用「LLM 优先语义审核 + 强规则直判高危 + 失败降级离线」的判定策略，支持 Human-in-the-loop 人工复核，并对外提供 REST API。

> 设计背景：作者曾在字节跳动参与抖音评论风控（涉政、未成年人等多风险域的数据增强与模型迭代），本项目用 Agent 工程化的方式复现内容审核的核心决策链路。这是个人作品集 / Demo 规模项目，测试输入为内置样例。

## 技术栈

- **编排框架**：LangGraph（StateGraph 有状态工作流）、LangChain
- **大模型**：DeepSeek API（`deepseek-chat`，OpenAI 兼容接口）
- **服务化**：FastAPI + Uvicorn
- **配置 / 测试**：python-dotenv、Pydantic、Pytest

## 工作流（6 节点 + 2 处条件路由）

```
normalize_input          文本清洗、截断
   ↓
identify_risk            一级风险识别（离线预判 → 按调度策略做 LLM 验证）
   ↓  route_by_risk      路由1：risk_type==safe 且 confidence>=0.75 → 直接出安全报告
   ├── generate_safe_report → END
   ↓
analyze_risk             二级深度分析，定风险等级 + 建议动作
   ↓  route_by_level     路由2：按 风险类型 / 等级 / 建议动作 判断是否转人工
   ├── human_review      Human-in-the-loop（支持自动决策跳过阻塞）
   ↓
generate_report          结构化 JSON 审核报告
   ↓
  END
```

- 全局状态用 `TypedDict (RiskState)` 维护，贯穿所有节点，带 `trace_id` 全链路追踪。
- 覆盖 8 类风险（涉政 / 未成年人 / 成人色情 / 违法违规 / 暴力 / 营销 / 安全 / 未知）× 5 个风险等级（high/medium/low/none/unknown）。
- `final_action` 取值：有真人/显式决策 → 该决策（approve/reject/skip）；命中复核但无人决策 → `pending_review`（待人工）；未进复核的低风险 → 模型建议动作。

## 核心设计：LLM 优先 + 强规则直判 + 离线降级

`identify_risk` 的判定逻辑（见 `audit/nodes.py` 与 `audit/judge.py`）：

1. **先跑离线确定性分类器**（`audit/classifier.py`，关键词 + 复合规则）拿到一个快速结果，并作为 LLM 不可用时的兜底。
2. **`LLMCallJudge` 采用 LLM 优先策略**：除下面两种"可信直判"情形外，其余内容一律升级到 LLM 做语义审核——以此抓住关键词规则漏掉的**黑话 / 隐晦表达**：
   - **复合强规则命中**（如「未成年语境 × 性相关词 → 未成年涉色」「亲属语境 × 性相关词 → 乱伦涉色」）：本地直接判 high 并 reject，无需 LLM；
   - **命中固定无害短语白名单**（`ABSOLUTE_SAFE_TEXTS`，如"你好/谢谢/今天天气真好"）：直接放行，无需 LLM。注意不是按长度——"约茶吗"虽短也是黑话，仍走 LLM。
3. **在线模式（`OFFLINE_DEMO_MODE=0` 且有 API Key）**：按上述策略调用 DeepSeek；调用失败 / 超时 → **自动降级**回离线分类器结果，保证流程不中断。
4. **离线模式（`OFFLINE_DEMO_MODE=1`）**：完全使用本地确定性规则，无网络、无 Key 也能运行（此时判定为"会升级"的内容回落到离线结果）。
5. 每条结果带 `judge_reason` 记录决策路径：`llm_first` / `absolutely_safe` / `high_confidence_compound_rule` / `escalated:llm_first` / `escalation_failed_offline_fallback`，完全可追溯、可审计。
6. **成本控制**：升级到 LLM 的内容，一级识别与二级分析在**同一次 LLM 调用**里完成，`analyze_risk` 复用该结果，单条内容最多一次付费调用。

## 项目结构

```
audit/
├── classifier.py   # 离线确定性分类器（关键词 + 复合规则）
├── judge.py        # LLM 调用调度策略（规则驱动）
├── nodes.py        # LangGraph 节点函数
├── graph.py        # 工作流编排 + 两处条件路由
├── models.py       # RiskState 状态定义 + Pydantic 请求/响应模型
├── storage.py      # jsonl 审计日志落盘与读取
├── review_store.py # 异步人工复核待审队列（持久化 + 结案）
└── config.py       # 环境变量配置
api/main.py             # FastAPI 服务
scripts/run_batch.py    # 命令行批量审核入口
scripts/run_llm_demo.py # 在线 LLM 路由演示（真实调用 DeepSeek）
scripts/package.sh      # 干净打包交付（排除 .env / 日志 / 缓存 / .git）
tests/                  # Pytest 单测（分类器 / 调度 / 图路由 / 升级降级 / 人审）
```

## 快速开始

### 1. 环境与依赖

```bash
conda create -n agent_project python=3.12
conda activate agent_project
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com

OFFLINE_DEMO_MODE=1               # 0=在线调用 API  1=纯离线规则（不联网/不花钱）
AUDIT_LOG_PATH=audit_log.jsonl
REVIEW_STORE_PATH=review_queue.json  # 异步人工复核待审队列文件
AUTO_REVIEW_DECISION=             # 保留为空；CLI 自动决策请显式传 -d

LLM_TIMEOUT=8
LLM_MAX_RETRIES=0
JUDGE_ENABLED=1
JUDGE_MIN_CONFIDENCE=0.90
```

`.env` 是本地密钥文件，已被 `.gitignore` 忽略，不要提交真实 API Key。

### 3. 三种运行方式

**A. 命令行批量审核**

```bash
# 直接传入评论
python scripts/run_batch.py -c "评论内容1" "评论内容2"

# 从文件读取（每行一条），并把结果写入 JSON
python scripts/run_batch.py -f comments.txt -o result.json

# 指定人工复核的自动决策；不传 -d 时，命中复核队列会停下来等待 input()
python scripts/run_batch.py -f comments.txt -d skip -o result.json
```

CLI 的 Human-in-the-loop 语义：
- 不传 `-d`：命中人工复核队列时进入交互式输入 `approve/reject/skip`；若在非交互环境（管道 / CI / 无 TTY），则自动标记 `pending_review`（不崩溃、不伪造 skip）。
- 传 `-d approve|reject|skip`：用该值模拟人工决策，适合批处理或测试。
- `AUTO_REVIEW_DECISION` 不再驱动 CLI 自动跳过，避免环境变量静默改变审核语义。

**B. REST API 服务**

```bash
uvicorn api.main:app --reload --port 8000
```

启动后访问：
- `http://localhost:8000/docs` — Swagger 交互文档
- `POST /audit` — 单条审核
- `POST /audit/batch` — 批量审核（单次最多 100 条）
- `GET /logs` — 审计日志查询
- `GET /health` — 健康检查
- `GET /reviews/pending` — 拉取待人工复核队列
- `POST /reviews/{trace_id}` — 审核员结案（提交 approve/reject/skip）

**异步 Human-in-the-loop（真待审队列）**：API 不能调用 `input()`（会卡住服务线程），所以采用解耦的两段式：

1. `POST /audit` 命中复核 → 内容持久化进**待审队列**，立即返回 `final_action="pending_review"`（暂不放行，对高风险即 fail-safe），不阻塞请求；
2. 审核员事后异步 `GET /reviews/pending` 拉取 → `POST /reviews/{trace_id}` 提交决策结案 → 记录转 `decided` 并写入最终动作。

> 待审队列持久化在 `REVIEW_STORE_PATH`（默认 `review_queue.json`，单 JSON 文件读-改-写）。⚠️ 仅适用单进程 / 低并发的 Demo 场景；高并发或多副本部署需换带事务的数据库。

调用示例（另开一个终端）：

```bash
# 1) 提交审核：高风险内容 → 自动进待审队列
curl -s -X POST localhost:8000/audit -H 'Content-Type: application/json' \
     -d '{"comment":"教你怎么逃税不被发现"}'
# → {..., "final_action": "pending_review", "trace_id": "<ID>"}

# 2) 审核员拉取待审队列
curl -s localhost:8000/reviews/pending

# 3) 审核员对该条结案（把 <ID> 换成上面的 trace_id）
curl -s -X POST localhost:8000/reviews/<ID> -H 'Content-Type: application/json' \
     -d '{"decision":"reject","reviewer":"张三"}'
# → {..., "status": "decided", "final_action": "reject"}；该条随即从待审队列移除
```

**C. 纯离线演示（不联网、不花钱）**

把 `.env` 里 `OFFLINE_DEMO_MODE` 设为 `1`，再跑上面任意命令即可。

### 4. 运行测试

```bash
python -m pytest -q
```

覆盖离线分类器、LLM 调用调度、图路由分流、识别升级 / 降级等路径。

### 5. 在线 LLM 路由演示

该脚本会真实调用 DeepSeek。确认 `.env` 中 `OFFLINE_DEMO_MODE=0` 且 API Key 有效后再运行：

```bash
python scripts/run_llm_demo.py
```

### 6. 打包交付（生成干净压缩包）

不要直接 zip 整个文件夹——那会把 `.env`（含真实 Key）、日志、缓存、`.git` 一起带出去。用：

```bash
bash scripts/package.sh    # 产物在上级目录，含自检：确保不含 .env / 日志 / 缓存
```

## 诚实性边界

- 本项目为个人作品集 / Demo，测试输入为内置样例，**未对接线上流量**，无 QPS / 准确率 / 日活等运营指标。
- `LLMCallJudge` 是**规则启发式**调度器，不是用模型判断是否调模型。
- DeepSeek 仅用于在线模式；离线模式不依赖任何外部模型。
- 成人色情已拆分为独立的 `adult` 风险域（high / reject），与 `minor`（未成年人，CSAM 立即拒绝并上报）按类型区分。
