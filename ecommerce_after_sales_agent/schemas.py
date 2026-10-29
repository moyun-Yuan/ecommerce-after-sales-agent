"""模型输出契约：分流、答复审查与评测分别定义，工作流状态仍使用 TypedDict。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class StructuredResult(BaseModel):
    """拒绝多余字段和隐式类型转换，避免字符串 false 被当作真值。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class TriageDecision(StructuredResult):
    """售后问题分流结果。两个字段都必须提供。"""

    route: Literal["answer_directly", "use_tools", "escalate_to_human"]
    reason: str = Field(min_length=1, description="简体中文分类理由")


class ReviewDecision(StructuredResult):
    """对政策依据、引用和敏感信息的审查结果。"""

    grounded: StrictBool = Field(description="政策或事实声明是否有检索依据")
    citations_ok: StrictBool = Field(description="政策来源编号是否有效")
    pii_found: StrictBool = Field(description="是否存在尚未脱敏的敏感信息")
    approved: StrictBool = Field(description="是否允许通过本次审查")
    reason: str = Field(min_length=1, description="简体中文审查理由")


class CorrectnessDecision(StructuredResult):
    """评测正确性结果；保留原报告使用的 pass 字段。"""

    passed: StrictBool = Field(alias="pass")
    reason: str = Field(min_length=1)


class GroundednessDecision(StructuredResult):
    """评测政策依据结果。"""

    grounded: StrictBool
    reason: str = Field(min_length=1)
