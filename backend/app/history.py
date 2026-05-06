"""一步查询：当前用户在表中的最近 N 条记录，仅通过 MCP。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import Settings

logger = logging.getLogger(__name__)

_SAFE_IDENT = re.compile(r"^[a-zA-Z0-9_-]+$")

# 复用 MCP 会话：每次请求都 new MultiServerMCPClient + get_tools 会重复拉起子进程/握手，常见 2–5s。
_mcp_sql_lock = asyncio.Lock()
_mcp_executor_cache: tuple[str, MultiServerMCPClient, BaseTool, str] | None = None


def _mcp_settings_fingerprint(settings: Settings) -> str:
    path = settings.mcp_connections_json
    assert path is not None and path.is_file()
    st = path.stat()
    return "|".join(
        [
            str(path.resolve()),
            str(st.st_mtime_ns),
            settings.mcp_mysql_server_name,
            str(settings.tool_name_prefix),
            settings.mcp_sql_tool_name or "",
            settings.mcp_sql_query_param,
        ]
    )


async def _get_mcp_sql_executor(settings: Settings) -> tuple[BaseTool, str]:
    global _mcp_executor_cache
    if not settings.mcp_connections_json or not settings.mcp_connections_json.is_file():
        raise RuntimeError("未配置 MCP_CONNECTIONS_JSON 或文件不存在")

    async with _mcp_sql_lock:
        fp = _mcp_settings_fingerprint(settings)
        if _mcp_executor_cache is not None and _mcp_executor_cache[0] == fp:
            return _mcp_executor_cache[2], _mcp_executor_cache[3]

        path = settings.mcp_connections_json
        connections = json.loads(path.read_text(encoding="utf-8"))
        if settings.mcp_mysql_server_name not in connections:
            raise RuntimeError(
                f"MCP 配置中缺少服务器键 {settings.mcp_mysql_server_name!r}，"
                f"当前键: {list(connections.keys())}",
            )

        client = MultiServerMCPClient(
            connections,
            tool_name_prefix=settings.tool_name_prefix,
        )
        tools = await client.get_tools(server_name=settings.mcp_mysql_server_name)
        tool = _pick_sql_tool(tools, settings.mcp_sql_tool_name)
        if tool is None:
            raise RuntimeError("MCP 未暴露任何可用工具")

        string_keys = _tool_string_arg_keys(tool)
        param = settings.mcp_sql_query_param
        if string_keys and param not in string_keys:
            param = string_keys[0] if len(string_keys) == 1 else param

        _mcp_executor_cache = (fp, client, tool, param)
        logger.info(
            "MCP 执行器已缓存（进程内复用 client/get_tools），fingerprint=%s tool=%s",
            fp[:120] + ("…" if len(fp) > 120 else ""),
            tool.name,
        )
        return tool, param


async def prewarm_mcp_sql_executor(settings: Settings) -> None:
    """在应用启动时调用，把 get_tools 的冷启动挪到服务就绪前。"""
    await _get_mcp_sql_executor(settings)


def _safe_ident(name: str, label: str) -> str:
    if not _SAFE_IDENT.match(name):
        raise ValueError(f"非法{label}: {name!r}，仅允许字母数字下划线中横线")
    return name


def _rows_to_jsonable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                item[k] = v.isoformat()
            else:
                item[k] = v
        out.append(item)
    return out


def _pick_sql_tool(tools: list[BaseTool], preferred: str | None) -> BaseTool | None:
    if not tools:
        return None
    if preferred:
        for t in tools:
            if t.name == preferred:
                return t
    for t in tools:
        lower = t.name.lower()
        if any(k in lower for k in ("sql", "query", "execute")):
            return t
    return tools[0]


def _tool_string_arg_keys(tool: BaseTool) -> list[str]:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return []
    try:
        js = schema.model_json_schema()
        props: dict[str, Any] = js.get("properties") or {}
        keys: list[str] = []
        for key, spec in props.items():
            t = spec.get("type")
            if t == "string" or (isinstance(t, list) and "string" in t):
                keys.append(key)
        return keys
    except Exception:
        return []


async def _invoke_mcp_sql(
    settings: Settings,
    sql: str,
) -> str:
    global _mcp_executor_cache
    t_mcp = time.perf_counter()
    tool, param = await _get_mcp_sql_executor(settings)
    t_after_tools = time.perf_counter()

    logger.info("调用 MCP 工具 %s，参数 %s", tool.name, param)
    try:
        result = await tool.ainvoke({param: sql})
    except Exception:
        async with _mcp_sql_lock:
            _mcp_executor_cache = None
        logger.warning("MCP ainvoke 异常，已丢弃缓存以便下次重连", exc_info=True)
        raise
    t_done = time.perf_counter()
    logger.info(
        "MCP 耗时: get_tools(含缓存命中)=%.3fs sql_ainvoke=%.3fs 合计=%.3fs (tool=%s)",
        t_after_tools - t_mcp,
        t_done - t_after_tools,
        t_done - t_mcp,
        tool.name,
    )
    if isinstance(result, str):
        lowered = result.lower()
        for keyword in ("error executing sql", "access denied", "command denied"):
            if keyword in lowered:
                raise RuntimeError(f"MCP 返回数据库错误: {result}")
    return result


def _parse_mcp_query_result(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [raw]
    text = str(raw).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [{"_raw": text}]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if "rows" in data and isinstance(data["rows"], list):
            return [x for x in data["rows"] if isinstance(x, dict)]
        return [data]
    return [{"_raw": text}]



async def fetch_user_recent_rows(
    settings: Settings,
    database: str,
    table: str,
    user_column: str,
    user_id: str,
    order_column: str,
    limit: int = 5,
    columns: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """返回 (rows, source_tag)。"""
    db = _safe_ident(database, "库名")
    tbl = _safe_ident(table, "表名")
    uc = _safe_ident(user_column, "用户列")
    oc = _safe_ident(order_column, "排序列")

    if settings.mcp_connections_json and settings.mcp_connections_json.is_file():
        uid = user_id.replace("\\", "\\\\").replace("'", "''")
        if columns:
            col_names = [_safe_ident(c, "列名") for c in columns]
            select_cols = "`, `".join(col_names)
            select_clause = f"`{select_cols}`"
        else:
            select_clause = "*"
        sql_literal = (
            f"SELECT {select_clause} FROM `{db}`.`{tbl}` WHERE `{uc}` = '{uid}' "
            f"ORDER BY `{oc}` DESC LIMIT {int(limit)}"
        )
        raw = await _invoke_mcp_sql(settings, sql_literal)
        rows = _parse_mcp_query_result(raw)
        return _rows_to_jsonable(rows), "mcp"

    return [], "none"
