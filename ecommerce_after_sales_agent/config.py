"""集中管理模型、数据路径、接口令牌和链路追踪配置。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Hugging Face 令牌兼容处理位于包的 __init__.py，保证先于模型库导入执行。

# 尽早加载项目根目录中的 .env，其他模块不直接读取环境变量。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- DeepSeek 模型配置 -------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()

# 默认三个角色共用一个模型，也可以在 .env 中分别覆盖。
TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", DEEPSEEK_MODEL)
SOLVER_MODEL = os.getenv("SOLVER_MODEL", DEEPSEEK_MODEL)
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", DEEPSEEK_MODEL)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", REVIEWER_MODEL)


def _env_bool(name: str, default: bool) -> bool:
    """严格读取开关，避免把字符串 false 当成 True。"""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须填写 true 或 false。")


def model_options(role: str) -> dict:
    """默认全部关闭思考；全局或节点开关用于按需启用及对照测试。"""
    if role not in {"general", "triage", "solver", "reviewer", "judge"}:
        raise ValueError("未知模型角色。")
    default = _env_bool("DEEPSEEK_THINKING", False)
    thinking = _env_bool(f"{role.upper()}_THINKING", default)
    effort = os.getenv(f"{role.upper()}_REASONING_EFFORT", os.getenv("DEEPSEEK_REASONING_EFFORT", "low")).strip()
    if effort not in {"low", "high", "max"}:
        raise ValueError("思考强度只支持 low、high、max。")
    max_tokens = int(os.getenv("THINKING_MAX_TOKENS", "8192"))
    if max_tokens <= 0:
        raise ValueError("THINKING_MAX_TOKENS 必须大于零。")
    return {"thinking": thinking, "effort": effort, "max_tokens": max_tokens}

# --- 检索与重排模型 ----------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 政策检索使用的双编码向量模型
EMBEDDING_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
RERANK_METHOD = "bge_cosine_plus_zh_bm25"

# --- 数据与文件路径 ----------------------------------------------------------
DATA_PROFILE = "cn-v2"
CURRENCY = "CNY"
CURRENCY_NAME = "人民币"
# 数据目录集中保存业务库、向量库和工作流检查点。
DATA_DIR = PROJECT_ROOT / "data" / "cn"
# 持久化文件名不随代码包改名，保证已有业务数据仍可读取。
DB_PATH = DATA_DIR / "after_sales_agent.db"           # 客户、订单、审批等业务数据
CHROMA_DIR = DATA_DIR / "chroma"                      # 持久化向量库
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.sqlite"  # LangGraph 持久化运行状态

# Chroma 中的政策知识库集合名称；这是持久化标识，不随代码包改名。
CHROMA_COLLECTION = "after_sales_agent_zh_v2"

# 员工审批接口令牌。接口不会提供默认值，避免误把未鉴权的审批服务暴露出去。
STAFF_API_TOKEN = os.getenv("STAFF_API_TOKEN", "").strip()

DATA_DIR.mkdir(parents=True, exist_ok=True)


def setup_tracing() -> bool:
    """规范 LangSmith 配置；启用追踪且存在密钥时返回 True。"""
    tracing_on = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    has_key = bool(os.getenv("LANGSMITH_API_KEY"))
    if tracing_on and has_key:
        # LangChain 读取这些变量，同时保证子进程能够继承。
        os.environ.setdefault("LANGSMITH_PROJECT", "ecommerce_after_sales_agent")
        os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        return True
    return False
