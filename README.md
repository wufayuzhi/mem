# NTN-Agents 代码仓

> 本目录 = NTN-Agents 项目可执行代码区。文档在仓库根 `1-需求书/`~`9-参考资料/` 。
> 死线参考:`/mnt/shared/agents/ntn-agents/0-决策与避雷/路径与命名钉死表-v1.md`。

## 目录结构

```
4-代码/
├── src/ntn_agents/         # 业务源码(标准 src/ layout)
│   ├── ntn_common/         # 跨模块公共:配置加载/时间/日志/db 连接
│   ├── ntn_db/             # state.db schema + 迁移 + 初始化(I-02)
│   ├── ntn_long_link/      # 企微 OpenClaw 长链接 2 路客户端(I-01/I-07)
│   ├── ntn_dispatcher/     # 派单 + 抢单 + 子进程驱动(I-02/I-03/I-08/I-09)
│   ├── ntn_approval_bot/   # 灰色案审批服务端 + 推送(I-05/I-06)
│   └── ntn_hooks/          # ccb PreToolUse hook 决策核(I-04)
├── vendor/multi-agent/     # 旧 multi-agent 代码(原样存档,只读参考)
├── tests/{unit,integration}/
├── ops/
│   ├── systemd/            # 5 个 unit 文件
│   └── config/             # *.yaml.example + secrets.env.example
├── scripts/                # 一次性初始化 / 校验脚本
├── migrations/             # state.db 迁移 SQL(0001_init.sql 起)
└── pyproject.toml
```

## 快速校验

```bash
cd /mnt/shared/agents/ntn-agents/4-代码

# 装包(开发模式)
pip install -e '.[dev]'

# 初始化 db(写到 /data/state.db,需 root 或 ntn:ntn)
ntn-state-init --db /data/state.db

# 跑测试
pytest -q

# 风格检查
ruff check src tests
```

## 凭证

不用 `.env` 直接放代码区。按 D-17 规定,生产环境凭证走:

```
/etc/ntn-agents/secrets.env  (0600, ntn:ntn, EnvironmentFile= 注入)
```

模板见 `ops/config/secrets.env.example`。
