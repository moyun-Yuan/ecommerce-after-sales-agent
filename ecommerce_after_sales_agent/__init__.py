"""售后智控：基于 LangGraph、DeepSeek 与 Chroma 的电商售后 Agent。"""

# Embedding 与重排模型是公开模型，默认使用匿名方式下载。
# 本地失效的 Hugging Face 令牌可能导致公开模型也返回 401，因此用户没有主动
# 配置令牌时，把令牌路径指向不存在的文件，强制使用匿名访问。这里必须在其他
# 模块导入 huggingface_hub 之前执行。
import os as _os
from pathlib import Path as _Path

if not (_os.getenv("HF_TOKEN") or _os.getenv("HUGGING_FACE_HUB_TOKEN")):
    _os.environ.setdefault(
        "HF_TOKEN_PATH", str(_Path(__file__).resolve().parent.parent / "data" / ".hf_no_token")
    )
    _os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    # 匿名下载是预期行为，隐藏要求设置 HF_TOKEN 的提示。
    import logging as _logging

    _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

__version__ = "0.2.0"
