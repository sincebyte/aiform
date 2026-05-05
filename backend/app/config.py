from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# DeepSeek 使用 OpenAI 兼容协议：https://api.deepseek.com/
_DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "DEEPSEEK_API_KEY"),
        description="DeepSeek/OpenAI 兼容密钥",
    )
    openai_base_url: str = Field(
        default=_DEFAULT_DEEPSEEK_BASE,
        validation_alias=AliasChoices("OPENAI_BASE_URL", "DEEPSEEK_BASE_URL"),
        description="默认同 DeepSeek；使用官方 OpenAI 时请设为 https://api.openai.com/v1",
    )
    openai_model: str = Field(
        default="deepseek-v4-pro",
        validation_alias=AliasChoices("OPENAI_MODEL", "DEEPSEEK_MODEL"),
    )

    mcp_connections_json: Path | None = Field(
        default=None,
        alias="MCP_CONNECTIONS_JSON",
        description="JSON 文件路径，结构与 langchain-mcp-adapters MultiServerMCPClient 的 connections 一致",
    )
    mcp_mysql_server_name: str = Field(
        default="work-mysql-prod",
        alias="MCP_MYSQL_SERVER_NAME",
    )
    mcp_sql_tool_name: str | None = Field(
        default=None,
        alias="MCP_SQL_TOOL_NAME",
        description="若为空则按名称启发式选择 SQL 工具",
    )
    mcp_sql_query_param: str = Field(
        default="query",
        alias="MCP_SQL_QUERY_PARAM",
        description="调用 MCP 工具时传入 SQL 的参数名（常见：query / sql）",
    )
    tool_name_prefix: bool = Field(
        default=False,
        alias="MCP_TOOL_NAME_PREFIX",
    )

    mysql_host: str | None = Field(default=None, alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str | None = Field(default=None, alias="MYSQL_USER")
    mysql_password: str | None = Field(default="", alias="MYSQL_PASSWORD")

    cors_origins: str = Field(
        default="*",
        alias="CORS_ORIGINS",
        description="逗号分隔，例如 http://127.0.0.1:8080,*",
    )


def get_settings() -> Settings:
    return Settings()
