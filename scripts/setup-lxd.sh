#!/usr/bin/env bash
# =============================================================================
# NTN-MEM LXD 容器一键部署脚本
# 在宿主机上运行，自动创建/配置 LXD 容器
# =============================================================================
set -euo pipefail

CONTAINER_NAME="${1:-MEM}"
IMAGE="${2:-ubuntu:22.04}"
PORT="${NTN_MEM_PORT:-8081}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== NTN-MEM LXD 部署 ==="
echo "容器名: $CONTAINER_NAME"
echo "源码:   $REPO_DIR"
echo ""

# 1. 检查 LXD
if ! command -v lxc &>/dev/null; then
    echo "❌ 未检测到 lxc 命令，请先安装 LXD"
    exit 1
fi

# 2. 创建容器（如果不存在）
if lxc info "$CONTAINER_NAME" &>/dev/null 2>&1; then
    echo "✓ 容器 $CONTAINER_NAME 已存在"
else
    echo "▶ 创建容器 $CONTAINER_NAME ..."
    lxc launch "$IMAGE" "$CONTAINER_NAME"
    echo "   等待容器就绪..."
    sleep 5
fi

# 3. 安装 Python（容器内）
echo "▶ 确保 Python3 已安装..."
lxc exec "$CONTAINER_NAME" -- bash -c "
    command -v python3 &>/dev/null || apt-get update -qq && apt-get install -y -qq python3
"

# 4. 挂载代码目录
echo "▶ 挂载代码目录..."
if lxc config device get "$CONTAINER_NAME" code-share source &>/dev/null 2>&1; then
    echo "   code-share 设备已存在，更新..."
    lxc config device remove "$CONTAINER_NAME" code-share
fi
lxc config device add "$CONTAINER_NAME" code-share disk \
    source="$REPO_DIR" \
    path=/opt/ntn-mem

# 5. 创建数据目录
echo "▶ 创建数据目录..."
lxc exec "$CONTAINER_NAME" -- mkdir -p /data /etc/ntn-mem

# 6. 推送 secrets（如果存在）
if [ -f "$REPO_DIR/.env" ]; then
    echo "▶ 推送 .env 到容器..."
    lxc file push "$REPO_DIR/.env" "$CONTAINER_NAME/etc/ntn-mem/secrets.env"
fi

# 7. 安装 systemd 服务
echo "▶ 安装 systemd 服务..."
lxc file push "$REPO_DIR/ops/systemd/ntn-mem.service" \
    "$CONTAINER_NAME/etc/systemd/system/"
lxc exec "$CONTAINER_NAME" -- systemctl daemon-reload
lxc exec "$CONTAINER_NAME" -- systemctl enable ntn-mem

# 8. 启动
echo "▶ 启动服务..."
lxc exec "$CONTAINER_NAME" -- systemctl restart ntn-mem

# 9. 等待就绪
echo "▶ 等待服务就绪..."
sleep 3
status=$(lxc exec "$CONTAINER_NAME" -- systemctl is-active ntn-mem 2>/dev/null || echo "unknown")
if [ "$status" = "active" ]; then
    container_ip=$(lxc list "$CONTAINER_NAME" --format=json | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['state']['network']['eth0']['addresses'][0]['address'])")
    echo "✅ MEM 服务已启动！"
    echo "   容器内: http://localhost:${PORT}"
    echo "   外网:   http://${container_ip}:${PORT}"
    echo ""
    echo "查看日志: lxc exec $CONTAINER_NAME -- journalctl -u ntn-mem -f"
else
    echo "❌ 服务启动失败（状态: $status）"
    echo "   日志: lxc exec $CONTAINER_NAME -- journalctl -u ntn-mem -n 50 --no-pager"
fi
