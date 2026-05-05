"""单次 LLM 调用生成表单预测（步骤尽量少）。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import Settings
from app.history import fetch_user_recent_rows
from app.models import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _extract_json(text: str) -> dict[str, Any]:
    m = _JSON_BLOCK.search(text)
    if not m:
        raise ValueError(f"LLM 返回中未找到 JSON: {text[:200]}")
    return json.loads(m.group(0))


def _as_str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    return str(v)


EXAMPLE_RESPONSE = """
{
  "fields": {
    "title": "示例标题",
    "summary": "根据历史记录推断的摘要……",
    "paper_date": "2026-05-02"
  }
}
""".strip()


def _example_for_fields(field_names: list[str]) -> str:
    sample = {n: f"<{n} 的预测字符串>" for n in field_names[:8]}
    return json.dumps({"fields": sample}, ensure_ascii=False, indent=2)


async def predict_form(settings: Settings, req: PredictRequest) -> PredictResponse:
    field_names = [f.name for f in req.fields]
    history, source = await fetch_user_recent_rows(
        settings,
        database=req.database,
        table=req.table,
        user_column=req.user_id_column,
        user_id=req.user_id,
        order_column=req.order_by_column,
        limit=10,
        columns=field_names,
    )
    filled = {k: v for k, v in req.current_values.items() if v not in (None, "")}
    field_prompts = {f.name: f.prompt for f in req.fields if f.prompt}
    logger.info("fields received: %s", [{f.name: f.prompt} for f in req.fields])

    field_prompt_instructions = ""
    if field_prompts:
        lines = [f"  - {name}: {text}" for name, text in field_prompts.items()]
        field_prompt_instructions = (
            "6. 必须严格遵循字段级提示词（field_prompts），其中包含各字段的计算规则和格式要求：\n"
            + "\n".join(lines)
            + "\n"
        )

    sys = SystemMessage(
        content=(
            "你是表单智能填充助手。你必须结合：当前用户 id、用户已填字段、"
            "该用户在数据库中的最近至多 10 条历史记录，推断空字段的合理取值。\n"
            "规则：\n"
            "1. 输出必须是结构化对象，且仅包含字段「fields」：name -> 字符串或 null。\n"
            "2. 用户已在表单中填写且非空的字段：不要改动（不要在 fields 里重复输出它们）。\n"
            "3. 对历史记录中的模式（常用措辞、日期格式、标签风格等）保持一致。\n"
            "4. 无法可靠推断的字段可省略或置为 null。\n"
            "5. 响应要快：不要做冗长推理，直接给结论。\n"
            + field_prompt_instructions
            + f"约定 JSON 示例（形状必须一致）：\n{EXAMPLE_RESPONSE}\n"
            f"本次字段示例（键集合应对齐表单）：\n{_example_for_fields(field_names)}\n"
        )
    )

    human_payload: dict[str, Any] = {
        "login_user_id": req.user_id,
        "database": req.database,
        "table": req.table,
        "form_field_names": field_names,
        "field_prompts": field_prompts or None,
        "current_values_non_empty": filled,
        "recent_rows_for_user": history,
        "custom_prompt": req.custom_prompt or None,
    }
    human = HumanMessage(
        content=(
            "请根据下列 JSON 上下文给出表单缺失字段预测（结构化输出）：\n"
            + json.dumps(human_payload, ensure_ascii=False, default=str)
        )
    )

    if settings.openai_api_key is None:
        logger.error("缺少 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
        raise RuntimeError("服务端未配置 DEEPSEEK_API_KEY（或 OPENAI_API_KEY）")


    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=SecretStr(settings.openai_api_key),
        base_url=settings.openai_base_url or None,
        temperature=0.2,
        timeout=60,
    )
    logger.info("LLM 请求 system: %s", sys.content)
    logger.info("LLM 请求 human: %s", human.content)
    msg = await llm.ainvoke([sys, human])
    logger.info("LLM 返回: %s", msg.content)
    content = msg.content
    if not isinstance(content, str):
        raise RuntimeError(f"LLM 返回非文本内容: {type(content)}")
    parsed = _extract_json(content)
    fields = parsed.get("fields", {})
    if not isinstance(fields, dict):
        raise RuntimeError("LLM 返回的 fields 不是对象")

    merged: dict[str, str | None] = {}
    for k, v in fields.items():
        merged[str(k)] = _as_str_or_none(v)
    for k in filled:
        merged.pop(k, None)

    return PredictResponse(
        fields=merged,
        meta={
            "history_rows": len(history),
            "history_source": source,
        },
    )
