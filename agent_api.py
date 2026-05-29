import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import json
from datetime import datetime
# ===== 1. 初始化LLM =====
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="",
    timeout=30,
    max_retries=2
)

# ===== 2. 定义状态 =====
class RiskState(TypedDict):
    user_input: str        # 用户输入的评论
    risk_type: str         # 识别出的风险类型
    risk_level: str        # 风险等级：high/medium/low
    analysis_result: str   # 分析结果
    human_decision: str    # 人工决策：approve/reject/skip
    final_report: str      # 最终报告

# ===== 3. 定义三个Node =====

# Node1：意图识别，判断是哪种风险
def identify_risk(state: RiskState) -> RiskState:
    response = llm.invoke([
        SystemMessage(content="""你是一个内容风险识别专家。
        判断用户输入属于哪种风险类型，只返回以下之一：
        - political（涉政风险）
        - minor（未成年人相关风险）  
        - illegal（违法违规风险）
        - safe（无风险）
        只返回类型词，不要其他内容。"""),
        HumanMessage(content=state["user_input"])
    ])
    state["risk_type"] = response.content.strip()
    return state

# Node2：风险分析，根据不同风险类型做不同处理
def analyze_risk(state: RiskState) -> RiskState:
    risk_prompts = {
        "political": "你是涉政内容审核专家，分析以下内容的涉政风险点，说明风险等级（高/中/低）和原因：",
        "minor": "你是未成年人保护审核专家，分析以下内容对未成年人的潜在危害，说明风险等级（高/中/低）和原因：",
        "illegal": "你是违法违规内容审核专家，分析以下内容的违法违规风险，说明风险等级（高/中/低）和原因：",
    }
    
    prompt = risk_prompts.get(state["risk_type"], risk_prompts["illegal"])
    
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state["user_input"])
    ])
    state["analysis_result"] = response.content
    
    # 提取风险等级
    level_response = llm.invoke([
        SystemMessage(content="根据以下分析结果，只返回风险等级，只能返回'高'或'中'或'低'三个字之一，不要其他内容："),
        HumanMessage(content=response.content)
    ])
    state["risk_level"] = level_response.content.strip()
    return state

# Node3：生成最终报告
def generate_report(state: RiskState) -> RiskState:
    response = llm.invoke([
        SystemMessage(content="""你是审核报告生成专家。
        根据风险类型和分析结果，生成一份结构化的审核报告，包含：
        1. 风险类型
        2. 风险等级
        3. 主要风险点
        4. 处置建议"""),
        HumanMessage(content=f"""
        原始内容：{state["user_input"]}
        风险类型：{state["risk_type"]}
        分析结果：{state["analysis_result"]}
        """)
    ])
    state["final_report"] = response.content
    return state
# Node4：人工审核节点（Human-in-the-loop）
def human_review(state: RiskState) -> RiskState:
    print(f"\n{'!'*50}")
    print(f"⚠️  高风险内容，需要人工审核")
    print(f"原始内容：{state['user_input']}")
    print(f"风险类型：{state['risk_type']}")
    print(f"风险等级：{state['risk_level']}")
    print(f"AI分析：{state['analysis_result'][:200]}...")
    print(f"{'!'*50}")
    
    while True:
        decision = input("请输入审核决策 [approve=通过/reject=拒绝/skip=跳过]: ").strip().lower()
        if decision in ["approve", "reject", "skip"]:
            state["human_decision"] = decision
            print(f"已记录人工决策：{decision}")
            break
        else:
            print("输入无效，请重新输入 approve / reject / skip")
    
    return state
# ===== 4. 定义路由：根据风险类型决定走哪个分析路径 =====
def route_by_risk(state: RiskState) -> Literal["analyze_risk", END]:
    if state["risk_type"] == "safe":
        return END
    return "analyze_risk"

# ===== 5. 构建Graph =====
workflow = StateGraph(RiskState)

# 添加节点
workflow.add_node("identify_risk", identify_risk)
workflow.add_node("analyze_risk", analyze_risk)
workflow.add_node("generate_report", generate_report)
workflow.add_node("human_review", human_review)
# 设置入口
workflow.set_entry_point("identify_risk")

# 添加边
workflow.add_conditional_edges(
    "identify_risk",
    route_by_risk,
    {
        "analyze_risk": "analyze_risk",
        END: END
    }
)
# 分析完之后判断风险等级，高风险走人工审核
def route_by_level(state: RiskState) -> Literal["human_review", "generate_report"]:
    if state["risk_type"]=="political":
        return "human_review"
    if state["risk_level"] == "高" or state["risk_level"] == "中":
        return "human_review"
    return "generate_report"

workflow.add_conditional_edges(
    "analyze_risk",
    route_by_level,
    {
        "human_review": "human_review",
        "generate_report": "generate_report"
    }
)

# 人工审核完之后走报告生成
workflow.add_edge("human_review", "generate_report")
workflow.add_edge("generate_report", END)

# 编译
app = workflow.compile()

# # ===== 6. 运行测试 =====
# if __name__ == "__main__":
#     test_inputs = [
#         "这个政策真的太好了，支持！",
#         "小孩子不应该看这种内容",
#         "教你怎么逃税不被发现"
#     ]
    
#     for text in test_inputs:
#         print(f"\n{'='*50}")
#         print(f"输入：{text}")
        
#         result = app.invoke({
#             "user_input": text,
#             "risk_type": "",
#             "analysis_result": "",
#             "final_report": ""
#         })
        
#         print(f"风险类型：{result['risk_type']}")
#         if result['final_report']:
#             print(f"审核报告：\n{result['final_report']}")
#         else:
#             print("判断结果：无风险，无需进一步审核")
# ===== 7. 日志记录函数 =====
def save_log(result: RiskState):
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": result["user_input"],
        "risk_type": result["risk_type"],
        "final_report": result["final_report"]
    }
    with open("audit_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

# ===== 8. 批量处理函数 =====
def batch_audit(comments: list[str]):
    results = []
    for i, comment in enumerate(comments):
        print(f"\n{'='*50}")
        print(f"处理第 {i+1}/{len(comments)} 条：{comment}")
        
        result = app.invoke({
            "user_input": comment,
            "risk_type": "",
            "analysis_result": "",
            "final_report": ""
        })
        
        save_log(result)
        results.append(result)
        
        print(f"风险类型：{result['risk_type']}")
        if result['final_report']:
            print(f"审核报告：\n{result['final_report']}")
        else:
            print("判断结果：无风险")
    
    # 汇总统计
    print(f"\n{'='*50}")
    print(f"批量审核完成，共处理 {len(results)} 条")
    risk_counts = {}
    for r in results:
        risk_type = r["risk_type"]
        risk_counts[risk_type] = risk_counts.get(risk_type, 0) + 1
    print("风险类型分布：")
    for risk_type, count in risk_counts.items():
        print(f"  {risk_type}: {count}条")
    print(f"日志已保存至 audit_log.jsonl")

# ===== 9. 运行 =====
if __name__ == "__main__":
    test_comments = [
        "这个政策真的太好了，支持！",
        "小孩子不应该看这种内容",
        "教你怎么逃税不被发现",
        "今天天气真好，出去玩吧",
        "我妈妈说，我的鸡巴比你大",
        "未成年人不应该沉迷学习",
        "反对政府，推翻专制统治",
        "这个视频里有暴力血腥画面"
    ]
    
    batch_audit(test_comments)   