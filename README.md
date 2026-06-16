# NTN-MEM

**零第三方依赖的轻量记忆管理系统**——冷热分层存储、场景感知搜索、决策偏置召回、画像蒸馏。

只用 Python 标准库：`wsgiref`（HTTP 服务）+ `urllib`（调 Qdrant/LLM）+ `sqlite3`（持久化）。无 FastAPI、无 Redis、无外部消息队列。

---

## 快速启动

### Docker（推荐）

```bash
# 1. 克隆
git clone https://github.com/wufayuzhi/mem.git && cd mem

# 2. 配置
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY、NTN_QDRANT_URL

# 3. 启动
docker compose up -d

# 4. 验证
curl http://localhost:8081/health
```

### 直接 Python

```bash
# 1. Python >= 3.10
python3 --version

# 2. 启动
export PYTHONPATH="$PWD/src"
export NTN_MEM_DB=/tmp/mem.db
export NTN_MEM_PROVIDER=openai
export OPENAI_API_KEY=sk-xxx
export NTN_QDRANT_URL=http://localhost:6333

python3 -m ntn_agents.ntn_mem.server --host 0.0.0.0 --port 8081
```

### LXD

```bash
chmod +x scripts/setup-lxd.sh
sudo ./scripts/setup-lxd.sh MEM
```

或手动步骤见 [`scripts/start.sh --lxd`](scripts/start.sh)。

---

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `NTN_MEM_HOST` | | `0.0.0.0` | 监听地址 |
| `NTN_MEM_PORT` | | `8081` | 监听端口 |
| `NTN_MEM_DB` | | `/data/mem.db` | SQLite 数据库路径 |
| `NTN_MEM_PROVIDER` | ✓ | | 嵌入提供方：`openai` / `siliconflow` / `ollama` |
| `NTN_QDRANT_URL` | ✓ | | Qdrant 向量存储地址 |
| `OPENAI_BASE_URL` | | `https://api.openai.com/v1` | LLM API 地址 |
| `OPENAI_API_KEY` | ✓ | | API 密钥 |
| `NTN_MEM_EMBED_MODEL` | | `BAAI/bge-m3` | 嵌入模型名 |
| `NTN_MEM_LLM_MODEL` | | | LLM 模型（recollect 分析用） |

---

## 项目结构

```
mem/
├── src/ntn_agents/ntn_mem/    ← 24 个 Python 源码文件
│   ├── server.py               ← WSGI 服务入口
│   ├── app.py                  ← HTTP 路由 + API 实现
│   ├── embedding.py            ← 嵌入向量生成（兼容多 provider）
│   ├── qdrant.py               ← Qdrant HTTP 客户端
│   ├── recollect.py            ← 两级渐进式记忆召回
│   ├── tiering.py              ← 冷热分层存储引擎
│   ├── profile_distill.py      ← 画像蒸馏
│   ├── manager_private.py      ← 私有记忆 CRUD
│   ├── manager_knowledge.py    ← 知识库管理 + 跨库搜索
│   ├── lifecycle_daemon.py     ← 生命周期守护
│   ├── procedural_to_skill.py  ← 过程→技能转化
│   └── shared_kb_ingestion.py  ← 共享知识库接入
├── ops/
│   ├── systemd/ntn-mem.service ← LXD/systemd 部署
│   └── config/                 ← 配置示例
├── Dockerfile                  ← Docker 镜像
├── docker-compose.yml          ← Docker Compose（含内嵌 Qdrant）
├── scripts/
│   ├── start.sh                ← 一键启动（自动检测 Docker/Python）
│   └── setup-lxd.sh            ← LXD 容器一键部署
├── .env.example                ← 环境变量模板
└── pyproject.toml
```

---

## API 概览

所有接口以 JSON 返回。详细用法见源码注释。

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/memory/profile/update` | 画像更新 |
| `GET` | `/v1/memory/recollect` | 两级渐进式召回 |
| `GET` | `/v1/memory/recollect/detail` | 纯检索原文召回 |
| `GET` | `/v1/agents/list` | 已注册 Agent |
| `GET` | `/v1/agents/{id}/memories` | 某 Agent 的记忆 |
| `POST` | `/v1/agents/register` | 注册 Agent |
| `GET` | `/v1/knowledge/list` | 知识库列表 |
| `GET` | `/v1/knowledge/search` | 知识库搜索 |
| `POST` | `/v1/knowledge/ingest` | 知识库导入 |

---

## 特性

- **冷热分层**：自动归档冷数据，搜索时 fallback 合并
- **场景感知**：6 类场景 + 3 类决策方向搜索加权
- **画像蒸馏**：每轮对话自动更新 Agent 画像
- **知识库**：多知识源管理 + 跨库搜索
- **记忆召回**：gist（LLM 概要）+ detail（原文检索）两级渐进
- **生命周期**：TTL 过期清理 + 冷热数据守护

## 许可证

MIT
