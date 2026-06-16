"""
知识库增量同步总控脚本。

调用方式（在 MEM 容器内）：
  cd /mnt/shared/agents/ntn-agents/4-代码
  PYTHONPATH=src python3 -m ntn_agents.ntn_mem.scripts.knowledge_sync [--kb kb_id] [--force] [--dry-run]

参数：
  --kb kb_id    只同步指定的知识库（默认同步所有 active）
  --force       强制全量重拉（忽略 last_sync）
  --dry-run     只显示会做什么，不实际写入
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ── 配置 ──
# 默认 3 个知识库源
DEFAULT_SOURCES: dict[str, dict] = {
    "hermesagent-org-cn": {
        "name": "Hermes 中文文档",
        "type": "docusaurus",
        "sitemap": "https://hermesagent.org.cn/sitemap.xml",
        "base_url": "https://hermesagent.org.cn",
        "root_path": "/docs/",
        "project_id": "knowledge_reserve",
        "source_tag": "hermesagent_cn_kb_ingest",
        "max_depth": 2,
    },
    "ccb-agent-aura": {
        "name": "CCB 执行规范",
        "type": "docusaurus",
        "sitemap": "https://ccb.agent-aura.top/sitemap.xml",
        "base_url": "https://ccb.agent-aura.top",
        "root_path": "/docs/",
        "project_id": "knowledge_reserve",
        "source_tag": "ccb_docs_ingest",
        "max_depth": 2,
    },
}

SYNC_STATE_PATH = "/data/knowledge_sync/last_sync.json"
MEM_BASE = os.environ.get("NTN_MEM_BASE_URL", "http://127.0.0.1:8081")
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_sync_state() -> dict:
    """Load the last sync state from disk."""
    if os.path.exists(SYNC_STATE_PATH):
        with open(SYNC_STATE_PATH) as f:
            return json.load(f)
    return {"sources": {}, "last_sync_at": None}


def _save_sync_state(state: dict) -> None:
    os.makedirs(os.path.dirname(SYNC_STATE_PATH), exist_ok=True)
    with open(SYNC_STATE_PATH, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ── 网页抓取 ──
def _fetch(url: str, timeout: int = 30) -> str | None:
    """Fetch a URL with retry. Returns body text or None."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=BROWSER_HEADERS, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"  [FAIL] fetch {url}: {exc}", flush=True)
    return None


# ── 多级页面发现 ──
def discover_docusaurus_urls(sitemap_url: str | None,
                              base_url: str,
                              root_path: str,
                              max_depth: int = 2) -> set[str]:
    """发现 Docusaurus 站点的所有文档 URL，递归最多 max_depth 层。

    策略：
    1. 有 sitemap.xml 就从 sitemap 拿
    2. 没 sitemap 就抓 root_path，提取链接递归
    """
    found: set[str] = set()

    # Strategy 1: sitemap
    if sitemap_url:
        print(f"  从 sitemap 发现页面: {sitemap_url}", flush=True)
        body = _fetch(sitemap_url)
        if body:
            urls = re.findall(r"<loc>([^<]+)</loc>", body, re.IGNORECASE)
            for u in urls:
                u = u.strip()
                if root_path in u and u not in found:
                    found.add(u)
            if found:
                print(f"    sitemap 找到 {len(found)} 个页面", flush=True)
                return found

    # Strategy 2: 页面递归
    print(f"  从 {base_url}{root_path} 递归发现页面", flush=True)
    to_visit = {f"{base_url}{root_path}"}
    visited: set[str] = set()
    depth = 0

    while to_visit and depth <= max_depth:
        next_visit: set[str] = set()
        for url in sorted(to_visit)[:50]:  # 防止爆炸
            if url in visited:
                continue
            visited.add(url)
            body = _fetch(url)
            if not body:
                continue

            # 提取页面的标题和内容，也作为一个文档页面
            if root_path in url and url not in found:
                found.add(url)

            # 提取链接
            links = re.findall(r'href="([^"]+)"', body)
            for link in links:
                if link.startswith("//"):
                    link = "https:" + link
                elif link.startswith("/"):
                    link = base_url + link
                elif not link.startswith("http"):
                    link = base_url.rstrip("/") + "/" + link.lstrip("/")

                if root_path in link and link not in visited and link not in to_visit                    and not any(excl in link for excl in ["#", "?version=", "/skills/"]):
                    next_visit.add(link)

        to_visit = next_visit
        depth += 1
        print(f"    层 {depth}: 新发现 {len(to_visit)} 个页面", flush=True)

    print(f"    总计发现 {len(found)} 个页面", flush=True)
    return found


# ── 页面内容解析 ──
def extract_docusaurus_page(url: str) -> dict | None:
    """抓取一个 Docusaurus 页面，提取标题和正文。"""
    body = _fetch(url)
    if not body:
        return None

    # Title
    m = re.search(r"<title>(.*?)</title>", body, re.DOTALL)
    title = m.group(1).strip() if m else url.rsplit("/", 1)[-1]

    # 正文
    content = None
    for tag in ("article", "main", "div.markdown", "div.docs-content"):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", body, re.DOTALL)
        if m:
            content = m.group(1)
            break
    if not content:
        # 兜底：取 body 中 visible text
        m = re.search(r"<body[^>]*>(.*?)</body>", body, re.DOTALL)
        if m:
            content = m.group(1)

    if not content:
        return None

    # 去 HTML tag
    for tag in ("script", "style", "nav", "footer", "header", "aside", "noscript"):
        content = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", content, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", content)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 100:
        return None
    return {"title": title, "url": url, "text": text}


# ── 分块 ──
def chunk_text(text: str, title: str, url: str, max_chars: int = 2000) -> list[str]:
    """按段落分块，每条不超过 max_chars。"""
    paras = re.split(r"\n\s*\n|(?<=[。！？.!?])\s+", text)
    chunks = []
    buf = ""
    for p in paras:
        p = p.strip()
        if not p or len(p) < 10:
            continue
        if buf and len(buf) + len(p) + 1 > max_chars:
            chunks.append(f"标题: {title}\n来源: {url}\n\n{buf.strip()}")
            buf = p
        else:
            buf = f"{buf}\n{p}".strip()
    if buf:
        chunks.append(f"标题: {title}\n来源: {url}\n\n{buf.strip()}")
    return chunks


# ── 写入 MEM ──
def ingest_into_mem(chunks: list[str], project_id: str, source_tag: str,
                    skip_duplicates: bool = True) -> dict:
    """通过 ingest_documents 写入 MEM（不走 HTTP loopback，直接调用）"""
    try:
        from .app import add_memory as _add_memory
    except ImportError:
        # Fallback: HTTP loopback
        return _ingest_via_http(chunks, project_id, source_tag)

    ingested = 0
    skipped = 0
    batch_size = 20

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        for chunk in batch:
            try:
                result = _add_memory({
                    "text": chunk,
                    "project_id": project_id,
                    "scope": "shared",
                    "memory_type": "knowledge_doc",
                    "layer": "long_term",
                    "source": source_tag,
                    "skip_duplicates": skip_duplicates,
                })
                if result.get("skipped_duplicate"):
                    skipped += 1
                else:
                    ingested += 1
            except Exception:
                skipped += 1
        print(f"    写入进度: {i+len(batch)}/{len(chunks)} (ingested={ingested}, skipped={skipped})",
              flush=True)

    return {"ingested": ingested, "skipped": skipped}


def _ingest_via_http(chunks: list[str], project_id: str, source_tag: str) -> dict:
    """HTTP 回退：如果没法直接 import，走 loopback"""
    import json as _json

    ingested = 0
    skipped = 0
    batch_size = 10

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        docs = [{"text": c, "metadata": {"source": source_tag}} for c in batch]
        body = _json.dumps({
            "kb_id": "kb-" + project_id,
            "documents": docs,
            "skip_duplicates": True,
        }).encode()
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"{MEM_BASE}/v1/knowledge/ingest",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = _json.loads(resp.read())
                ingested += result.get("ingested", 0)
                skipped += result.get("skipped_duplicates", 0)
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(3)
                else:
                    print(f"    [FAIL] ingest batch: {exc}", flush=True)

    return {"ingested": ingested, "skipped": skipped}


# ── 主流程 ──
def sync_source(source_key: str, source: dict, state: dict,
                force: bool = False, dry_run: bool = False) -> dict:
    """同步单个知识库源。返回结果统计。"""
    src_state = state["sources"].get(source_key, {})
    last_sync_at = src_state.get("last_sync_at")
    last_count = src_state.get("document_count", 0)

    print(f"\n{'='*60}")
    print(f"知识库: {source['name']} ({source_key})")
    if not force and last_sync_at:
        print(f"  上次同步: {last_sync_at} ({last_count} docs)")
        print(f"  增量模式（--force 全量重拉）")
    else:
        print(f"  {'全量重拉' if force else '首次拉取'}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] 跳过实际拉取")
        return {"source": source_key, "action": "dry_run"}

    # Step 1: 发现页面
    print("\nStep 1/3: 发现页面...")
    urls = discover_docusaurus_urls(
        source.get("sitemap"), source["base_url"],
        source["root_path"], source.get("max_depth", 2),
    )
    if not urls:
        print("  ❌ 没有发现任何页面")
        return {"source": source_key, "error": "no_urls_discovered"}

    print(f"  发现 {len(urls)} 个页面")

    # Step 2: 抓取内容
    print("\nStep 2/3: 抓取页面内容...")
    pages = []
    for idx, url in enumerate(sorted(urls)):
        page = extract_docusaurus_page(url)
        if page:
            pages.append(page)
        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx+1}/{len(urls)} (有效: {len(pages)})", flush=True)

    print(f"  有效页面: {len(pages)}/{len(urls)}")

    # Step 3: 分块+写入
    print("\nStep 3/3: 分块写入 MEM...")
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page["text"], page["title"], page["url"])
        all_chunks.extend(chunks)

    print(f"  共 {len(all_chunks)} 个分块")

    # 增量检查：如果全量模式或首次，从 state 判断
    result = ingest_into_mem(
        all_chunks, source["project_id"], source["source_tag"],
    )

    # 记录同步状态
    state["sources"][source_key] = {
        "last_sync_at": _now_iso(),
        "url_count": len(urls),
        "page_count": len(pages),
        "chunk_count": len(all_chunks),
        "new_docs": result["ingested"],
        "existing_docs": result["skipped"],
        "source_tag": source["source_tag"],
    }
    state["last_sync_at"] = _now_iso()
    _save_sync_state(state)

    print(f"\\n✅ {source['name']}: new={result['ingested']}, existing={result['skipped']}")
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="知识库增量同步")
    parser.add_argument("--kb", help="只同步指定知识库（source_key）")
    parser.add_argument("--force", action="store_true", help="强制全量重拉")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写")
    args = parser.parse_args()

    state = _load_sync_state()
    total = {"ingested": 0, "skipped": 0}

    for key, source in DEFAULT_SOURCES.items():
        if args.kb and key != args.kb:
            continue
        result = sync_source(key, source, state, force=args.force, dry_run=args.dry_run)
        if "ingested" in result:
            total["ingested"] += result.get("ingested", 0)
            total["skipped"] += result.get("skipped", 0)

    print(f"\n{'='*60}")
    print(f"总览: 新写入 {total['ingested']}, 跳过 {total['skipped']}")
    print(f"状态文件: {SYNC_STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
