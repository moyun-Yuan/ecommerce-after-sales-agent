"""工作流与评测共用的 DeepSeek 对话模型工厂。"""
from __future__ import annotations

import json
from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, ValidationError

from . import config

ResultT = TypeVar("ResultT", bound=BaseModel)


class StructuredOutputError(ValueError):
    """模型响应无法满足指定输出契约；不包含可能敏感的原始回复。"""


class ThinkingChatDeepSeek(ChatDeepSeek):
    """补齐专用适配器的思考字段回传；不把思考正文展示给客户或写进统计报告。

    适配当前固定版本的请求构造接口，由真实 HTTP 序列化回归测试保护。
    专用适配器负责把响应 reasoning_content 保存到 AIMessage.additional_kwargs。
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        thinking = (payload.get("extra_body") or {}).get("thinking", {}).get("type") == "enabled"
        if thinking and payload.get("tools"):
            originals = [m for m in self._convert_input(input_).to_messages() if isinstance(m, AIMessage)]
            assistants = [m for m in payload["messages"] if m["role"] == "assistant"]
            if len(originals) != len(assistants):
                raise ValueError("思考消息与请求消息数量不一致，停止发送。")
            for original, outgoing in zip(originals, assistants):
                reasoning = original.additional_kwargs.get("reasoning_content")
                if not isinstance(reasoning, str):
                    raise ValueError("思考模式工具历史缺少 reasoning_content，不能安全恢复该轮调用。")
                outgoing["reasoning_content"] = reasoning
        return payload


def invoke_structured(model, schema: type[ResultT], messages: list[BaseMessage]) -> ResultT:
    """仅为本次调用绑定 JSON 模式和 Pydantic 解析，不修改基础模型配置。

    DeepSeek JSON 模式约束 JSON 格式，字段与类型由本地 Pydantic 校验。
    json_mode 不会自动把 schema 发给模型，因此显式加入输出格式说明。
    网络错误、鉴权错误仍向调用者抛出，不伪装成校验失败。
    """
    instruction = SystemMessage(content=(
        "请只输出一个符合以下 JSON Schema 的 JSON 对象，不要输出 Markdown 或额外解释。"
        "所有理由使用简体中文；布尔字段必须是 JSON 的 true/false，不得使用字符串。\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    ))
    structured_model = model.with_structured_output(schema, method="json_mode")
    try:
        return structured_model.invoke([instruction, *messages])
    except (OutputParserException, ValidationError) as exc:
        raise StructuredOutputError(f"{schema.__name__} 输出校验失败") from exc


def create_chat_model(model: str | None = None, temperature: float = 0.0, *, role: str = "general",
                      http_client=None) -> ThinkingChatDeepSeek:
    if not config.DEEPSEEK_API_KEY:
        raise ValueError("请在项目 .env 文件中填写 DEEPSEEK_API_KEY。")
    options = config.model_options(role)
    mode = "enabled" if options["thinking"] else "disabled"
    thinking_kwargs = ({"reasoning_effort": options["effort"], "max_tokens": options["max_tokens"]}
                       if options["thinking"] else {"temperature": temperature})
    return ThinkingChatDeepSeek(
        model=model or config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        timeout=120 if options["thinking"] else 45,
        max_retries=1,
        use_responses_api=False,
        http_socket_options=(),
        extra_body={"thinking": {"type": mode}},
        tags=[f"stage:{role}", f"thinking:{mode}"],
        http_client=http_client,
        **thinking_kwargs,
    )
