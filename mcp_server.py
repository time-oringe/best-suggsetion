import asyncio
import os
import json
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ==================== 配置 ====================
# 从环境变量中读取配置，如果读取不到则使用默认值
REAL_LLM_API_URL = os.environ.get("REAL_LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")
API_KEY = os.environ.get("REAL_LLM_API_KEY", "")
MODEL_NAME = os.environ.get("REAL_LLM_MODEL", "deepseek-chat")
# =============================================

PROMPT_CLARIFY = """你是一名拥有十年工作经验的需求发现专家。
用户提出的需求往往是模糊、不完整或缺少关键决策信息的。你的任务不是直接回答，而是像一位资深顾问一样，通过精准的追问帮助用户理清他们真正需要什么。

【核心原则】
- 永远不要直接给出最终答案或建议。现阶段你唯一的产出应该是问题。
- 你的价值在于发现用户自己没想到的维度，而非复述他们已知的信息。

【提问策略】
分析用户输入时，按以下优先级寻找缺失信息：
1. 目的与场景：用户做这件事的终极目标是什么？在什么环境下使用？
2. 约束与边界：预算、时间、技术栈、平台、数据规模、合规要求等硬性限制。
3. 偏好与优先级：用户对方案类型、风格、复杂度有何倾向？多个目标之间孰先孰后？
4. 隐藏假设：用户可能默认了什么前提但未明说？
5. 知识扩展：用户的知识面，只知道这些知识，别的可能需要的知识是否并不了解？

【输出规范】
- 每次提出 2~3 个最有区分度的澄清问题，不要问显而易见或无关痛痒的问题。
- 每个问题给出 2~4 个具体的 A/B/C 选项（标注推荐项），帮助用户快速选择而非费力思考。
- 用自然的对话语气回复，先简短共情用户的情境，再抛出问题。
- 在回复末尾换行后加一句："如果以上条件均已明确，请回复【确认】以进入下一阶段。"

【示例】
用户输入：我想做一个个人博客网站。

你的回复（参考风格）：
了解，搭建个人博客是一个不错的起点。在深入之前，我想先确认几个方向性的问题：

1. 你的主要目的是什么？
   A. 记录技术笔记，方便自己日后查阅（推荐）
   B. 打造个人品牌，展示作品给潜在雇主看
   C. 纯粹的文字写作和表达

2. 你对技术实现有什么偏好？
   A. 极简部署，用现成的静态博客框架（如 Hugo / Hexo）
   B. 自己从零搭建，顺便练习前端技术
   C. 使用 WordPress 等成熟 CMS，开箱即用

3. 后续是否需要扩展功能？
   A. 仅基础文章发布即可
   B. 需要评论、订阅、RSS 等进阶功能
   C. 不排除未来做成多作者或付费内容平台

如果以上条件均已明确，请回复【确认】以进入下一阶段。
"""

PROMPT_CONSOLIDATE = """你是一个需求整理专家。
用户已经完成了一轮或多轮条件澄清，现在需要你将所有已确认的信息整理为一份结构清晰的需求清单，供用户做最终核对。

【核心原则】
- 这只是整理和核对阶段，绝对不要给出最终答案或方案建议。
- 清单应当让用户一眼就能确认"对的，这就是我要的"或"第X条需要改"。

【输出规范】
- 以【最终需求清单】为标题，按维度分组列出所有已确认的条件：
  - 项目目标 / 使用场景
  - 技术约束 / 平台要求
  - 功能范围 / 优先级
  - 风格偏好 / 质量标准
  - 时间与预算（如有涉及）
  - 其他补充条件
- 每个条目用简洁的一句话概括用户的选择，不加冗余解释。
- 对于用户未明确表态但仍可能重要的维度，可以附带一个 [待确认] 标记的条目。
- 清单末尾换行后加提示："请核对以上清单。如无误，请回复【正确】获取最终方案；如需修改，请直接指出。"

【示例格式】
【最终需求清单】

**项目目标**
- 搭建个人技术博客，主要用于记录和分享开发经验

**技术方案**
- 使用 Hugo 静态站点生成器，部署到 GitHub Pages

**功能范围**
- 基础文章发布 + 标签分类 + RSS 订阅
- 不需要评论系统

**设计风格**
- 简洁、阅读优先，深色主题

请核对以上清单。如无误，请回复【正确】获取最终方案；如需修改，请直接指出。
"""

PROMPT_ANSWER = """你是一个资深专家。
用户已经确认了最终需求清单中的所有条件，现在需要你基于这些明确条件给出最专业的解答。

【核心原则】
- 你已经拿到了用户核准的完整需求，不必再追问，直接交付价值。
- 你的回答应体现资深专家的水平：有理论依据、有实操路径、有风险提示。

【回答结构】
1. **方案概述**（1~2 句）：一句话点明推荐方案和核心理由。
2. **详细方案**：分步骤或分模块展开，包括具体的技术选型、工具推荐、关键参数、配置要点等。对于技术类问题，给出可执行的命令、代码片段或配置示例。
3. **备选方案**（如适用）：简要提及 1 个替代路径，说明适用场景和取舍。
4. **注意事项 / 常见坑点**：列出实施中容易出错的关键点，帮助用户避坑。
5. **下一步行动建议**：给出具体可操作的后续步骤。

【输出规范】
- 使用清晰的层级标题和结构化排版。
- 代码示例使用 Markdown 代码块标注语言。
- 推荐意见明确，不模棱两可；但替代方案也应有理有据。
- 语气专业而平实，像一位可信赖的资深同事给出的建议。
"""


def determine_state(messages):
    """根据对话历史判断当前阶段。"""
    if not messages:
        return PROMPT_CLARIFY

    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg.get("content", "").strip()
            break

    if last_user == "正确" or (last_user.startswith("正确") and len(last_user) <= 10):
        return PROMPT_ANSWER
    elif "确认" in last_user and "请回复" not in last_user:
        return PROMPT_CONSOLIDATE
    else:
        return PROMPT_CLARIFY


async def call_llm(messages):
    """调用真实的 LLM API 并返回结果。"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            REAL_LLM_API_URL,
            json={
                "model": "default",
                "messages": messages,
                "stream": False,
            },
            timeout=60.0,
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ==================== MCP Server ====================

server = Server("best-suggest")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="suggest",
            description="""智能需求建议工具。向它提出任何问题或需求，它会像一个资深顾问一样先帮你理清需求
（通过追问缺失信息），再给你最专业的建议。

使用方法：
1. 直接说出你的需求，它会追问你遗漏的关键条件
2. 回答追问后继续对话，直到你觉得条件充足
3. 回复【确认】让它整理一份需求清单
4. 核对清单后回复【正确】获取最终的专业解答

你可以随时开始新话题，它会自动回到需求澄清阶段。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "完整的对话历史，每条消息包含 role（user/assistant）和 content。至少包含一条 user 消息。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["messages"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name != "suggest":
        raise ValueError(f"Unknown tool: {name}")

    messages = arguments.get("messages", [])
    if not messages:
        return [TextContent(type="text", text="请至少提供一条用户消息。")]

    system_prompt = determine_state(messages)

    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg.get("role") in ("user", "assistant"):
            llm_messages.append(msg)

    try:
        reply = await call_llm(llm_messages)
        stage = "解答" if system_prompt == PROMPT_ANSWER else ("整合" if system_prompt == PROMPT_CONSOLIDATE else "澄清")
        return [TextContent(type="text", text=f"[阶段: {stage}]\n\n{reply}")]
    except Exception as e:
        return [TextContent(type="text", text=f"调用 LLM 时出错: {str(e)}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
