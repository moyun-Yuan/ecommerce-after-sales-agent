# DeepSeek 与本地配置

## 创建配置文件

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写自己的密钥：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
STAFF_API_TOKEN=
```

`.env` 已被 Git 忽略，不要提交、截图或发送其中的真实密钥。

## 检查模型与工具调用

```powershell
python check_model.py
```

脚本只检查普通回复、工具调用请求和工具结果回传，不读取订单，也不执行退款。

## 配置员工 API

生成随机令牌：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

把结果填入 `STAFF_API_TOKEN`。如果留空，员工审批 API 会返回 503，避免未鉴权的
审批服务被意外启用。调用员工接口时，把令牌放入 `X-Staff-Token` 请求头。

## 初始化和启动

```powershell
python -m ecommerce_after_sales_agent.seed
python -m streamlit run app.py
```

或启动 FastAPI：

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

当前项目使用 DeepSeek 兼容的 Chat Completions 工具调用。三个模型角色默认共用
`DEEPSEEK_MODEL`，也可以通过 `TRIAGE_MODEL`、`SOLVER_MODEL`、`REVIEWER_MODEL`
分别配置。
