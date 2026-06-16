# NTN-MEM Docker 镜像
# 零第三方依赖——纯标准库
# 兼容 arm64 / amd64

FROM python:3.11-slim

LABEL org.opencontainers.image.title="NTN-MEM"
LABEL org.opencontainers.image.description="轻量记忆管理系统"
LABEL org.opencontainers.image.version="0.2.0"

WORKDIR /opt/ntn-mem

# 拷贝源码
COPY src/ src/
COPY pyproject.toml .

# 数据目录
VOLUME ["/data"]

# 默认端口
EXPOSE 8081

# 启动命令
CMD ["python3", "-m", "ntn_agents.ntn_mem.server", "--host", "0.0.0.0", "--port", "8081"]
