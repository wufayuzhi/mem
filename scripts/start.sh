#!/usr/bin/env bash
# =============================================================================
# NTN-MEM 一键启动脚本
# 支持 Docker 和直接 Python 两种方式
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# 加载 .env（如果存在）
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# 默认值
HOST="${NTN_MEM_HOST:-0.0.0.0}"
PORT="${NTN_MEM_PORT:-8081}"
DB="${NTN_MEM_DB:-/data/mem.db}"
PROVIDER="${NTN_MEM_PROVIDER:-openai}"
QDRANT_URL="${NTN_QDRANT_URL:-http://10.69.68.15:6333}"

usage() {
    echo "用法: $0 [OPTIONS]"
    echo ""
    echo "启动方式（自动检测）："
    echo "  $0                    优先检测 docker，回退到直接 Python"
    echo ""
    echo "选项："
    echo "  --docker         强制使用 Docker Compose"
    echo "  --direct         强制直接 Python 运行"
    echo "  --lxd            输出 LXD 安装说明"
    echo "  -h, --help       显示此帮助"
    exit 0
}

MODE="auto"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker) MODE="docker" ;;&
        --direct) MODE="direct" ;;&
        --lxd)    MODE="lxd" ;;&
        -h|--help) usage ;;
        *) echo "未知选项: $1"; usage ;;
    esac
    shift
done

# --------------- Docker 模式 ---------------
docker_mode() {
    echo "▶ 启动 Docker Compose..."
    docker compose up -d mem
    echo "✅ MEM 服务已启动 → http://localhost:${PORT}"
    echo "   日志: docker compose logs -f mem"
}

# --------------- 直接 Python 模式 ---------------
direct_mode() {
    echo "▶ 直接 Python 启动..."

    # 检查 Python >= 3.10
    py_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$(echo "$py_ver" | cut -d. -f1)" -lt 3 ] || { [ "$(echo "$py_ver" | cut -d. -f1)" -eq 3 ] && [ "$(echo "$py_ver" | cut -d. -f2)" -lt 10 ]; }; then
        echo "❌ 需要 Python >= 3.10，当前: $py_ver"
        exit 1
    fi

    export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
    export NTN_MEM_DB="$DB"
    export NTN_MEM_PROVIDER="$PROVIDER"
    export NTN_QDRANT_URL="$QDRANT_URL"

    echo "  数据库: $DB"
    echo "  Qdrant: $QDRANT_URL"
    echo "  Provider: $PROVIDER"
    echo ""

    exec python3 -m ntn_agents.ntn_mem.server --host "$HOST" --port "$PORT"
}

# --------------- LXD 安装说明 ---------------
lxd_mode() {
    echo "=== LXD 容器部署 ==="
    echo ""
    echo "1. 创建容器（如果未创建）："
    echo "   lxc launch ubuntu:22.04 MEM"
    echo ""
    echo "2. 代码挂载："
    echo "   lxc config device add MEM code-share disk"
    echo "       source=\${PWD} path=/opt/ntn-mem"
    echo ""
    echo "3. 数据目录："
    echo "   lxc exec MEM -- mkdir -p /data"
    echo ""
    echo "4. 复制 secrets："
    echo "   lxc exec MEM -- mkdir -p /etc/ntn-mem"
    echo "   lxc file push .env /etc/ntn-mem/secrets.env"
    echo ""
    echo "5. 安装服务："
    echo "   lxc file push ops/systemd/ntn-mem.service \\"
    echo "       /etc/systemd/system/"
    echo "   lxc exec MEM -- systemctl daemon-reload"
    echo "   lxc exec MEM -- systemctl enable --now ntn-mem"
    echo ""
    echo "6. 查看日志："
    echo "   lxc exec MEM -- journalctl -u ntn-mem -f"
}

# --------------- 自动检测 ---------------
case "$MODE" in
    docker)
        docker_mode
        ;;
    direct)
        direct_mode
        ;;
    lxd)
        lxd_mode
        ;;
    auto)
        if command -v docker &>/dev/null && docker compose version &>/dev/null; then
            docker_mode
        else
            direct_mode
        fi
        ;;
esac
