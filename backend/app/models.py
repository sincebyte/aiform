from typing import Any

from pydantic import BaseModel, Field


class FormFieldSpec(BaseModel):
    name: str
    tag: str = "input"
    input_type: str | None = Field(default=None, alias="type")
    label: str | None = None
    prompt: str | None = None

    model_config = {"populate_by_name": True}


class PredictRequest(BaseModel):
    database: str = Field(description="MySQL 库名")
    table: str = Field(description="表名")
    user_id: str = Field(description="当前登录用户 id")
    user_id_column: str = Field(default="user_id", description="表中标识用户的列名")
    order_by_column: str = Field(
        default="id",
        description="用于取最近记录的排序列（DESC），需为可排序列",
    )
    limit: int = Field(default=10, description="查询最近历史记录条数", ge=1, le=100)
    custom_prompt: str = Field(default="", description="业务侧自定义提示")
    fields: list[FormFieldSpec] = Field(description="表单字段元数据")
    current_values: dict[str, Any] = Field(
        default_factory=dict,
        description="表单当前已填写值（name -> value）",
    )


class PredictResponse(BaseModel):
    fields: dict[str, str | None] = Field(
        description="待写入表单的字段值；仅包含需要填充或建议的字段",
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="调试信息：历史条数、数据来源等",
    )
