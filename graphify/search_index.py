from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_FILE = ".graphify_embeddings.npy"
EMBED_IDS_FILE = ".graphify_embed_ids.json"
ALIAS_INDEX_FILE = ".graphify_alias_index.json"
METADATA_FILE = ".graphify_search_index.json"


# Project/domain-specific path aliases belong in graphify.aliases.json next to
# graph.json. The built-in set intentionally stays empty so public graphify does
# not bake POPO, chat, or product-specific vocabulary into every project.
PATH_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = ()

CLASS_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Builder", ("构建器",)),
    ("Handler", ("处理器",)),
    ("Manager", ("管理器",)),
    ("Helper", ("辅助类",)),
    ("ViewController", ("视图控制器",)),
    ("Controller", ("控制器",)),
    ("Component", ("组件",)),
    ("Factory", ("工厂",)),
    ("DataSource", ("数据源",)),
    ("Delegate", ("代理",)),
    ("Model", ("模型",)),
    ("View", ("视图",)),
    ("Cell", ("单元格",)),
    ("Menu", ("菜单",)),
    ("Translate", ("翻译",)),
    ("Search", ("搜索",)),
    ("Notification", ("通知",)),
    ("Login", ("登录",)),
    ("Theme", ("主题",)),
    ("Audio", ("音频",)),
    ("Video", ("视频",)),
    ("File", ("文件",)),
    ("Image", ("图片",)),
    ("WebView", ("网页视图",)),
    ("Tabbar", ("底部导航",)),
    ("AI", ("AI人工智能",)),
    ("OCR", ("文字识别",)),
    ("Emoji", ("表情",)),
    ("Reaction", ("表情反应",)),
    ("Todo", ("待办",)),
)


@dataclass(frozen=True)
class IndexBuildResult:
    node_count: int
    embedding_dim: int
    model_name: str
    out_dir: Path
    alias_count: int


def _load_graph_nodes(graph_path: Path) -> list[dict[str, Any]]:
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    return list(data.get("nodes", []))


def _split_identifier(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_:/.\-+]+", " ", text)
    return text


def _custom_alias_rules(graph_dir: Path) -> list[tuple[str, tuple[str, ...]]]:
    path = graph_dir / "graphify.aliases.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rules: list[tuple[str, tuple[str, ...]]] = []
    if isinstance(raw, dict):
        for pattern, aliases in raw.items():
            if isinstance(aliases, str):
                rules.append((str(pattern), tuple(a for a in aliases.split() if a)))
            elif isinstance(aliases, list):
                rules.append((str(pattern), tuple(str(a) for a in aliases if str(a).strip())))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            pattern = item.get("pattern")
            aliases = item.get("aliases", [])
            if pattern and isinstance(aliases, list):
                rules.append((str(pattern), tuple(str(a) for a in aliases if str(a).strip())))
    return rules


def generate_aliases(
    label: str,
    source_file: str,
    *,
    graph_dir: Path | None = None,
    custom_rules: list[tuple[str, tuple[str, ...]]] | None = None,
) -> list[str]:
    aliases: set[str] = set()
    source_lower = source_file.lower()
    label_lower = label.lower()
    rules = PATH_ALIASES + tuple(custom_rules if custom_rules is not None else (_custom_alias_rules(graph_dir) if graph_dir else ()))
    for pattern, values in rules:
        pattern_lower = pattern.lower()
        if pattern_lower in source_lower or pattern_lower in label_lower:
            aliases.update(values)
    for suffix, values in CLASS_ALIASES:
        if suffix.lower() in label_lower:
            aliases.update(values)
    for chinese in re.findall(r"[\u4e00-\u9fff]+", label):
        if len(chinese) >= 2:
            aliases.add(chinese)
    return sorted(a for a in aliases if a)


def _node_text(node: dict[str, Any], aliases: list[str]) -> str:
    label = str(node.get("label") or "")
    source_file = str(node.get("source_file") or "")
    node_id = str(node.get("id") or "")
    pieces = [
        label,
        _split_identifier(label),
        source_file,
        _split_identifier(Path(source_file).stem),
        _split_identifier(node_id),
        " ".join(aliases),
    ]
    return " ".join(p for p in pieces if p).strip()


def build_index(
    graph_path: str | Path,
    *,
    model_name: str | None = None,
    out_dir: str | Path | None = None,
    batch_size: int = 128,
) -> IndexBuildResult:
    graph_path = Path(graph_path).resolve()
    graph_dir = Path(out_dir).resolve() if out_dir else graph_path.parent
    graph_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name or os.environ.get("GRAPHIFY_EMBEDDING_MODEL") or DEFAULT_MODEL_NAME

    nodes = _load_graph_nodes(graph_path)
    node_ids: list[str] = []
    texts: list[str] = []
    alias_index: dict[str, list[str]] = defaultdict(list)
    custom_rules = _custom_alias_rules(graph_dir)
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        label = str(node.get("label") or "")
        source_file = str(node.get("source_file") or "")
        aliases = generate_aliases(label, source_file, graph_dir=graph_dir, custom_rules=custom_rules)
        for alias in aliases:
            alias_index[alias].append(str(node_id))
        node_ids.append(str(node_id))
        texts.append(_node_text(node, aliases))

    try:
        import numpy as np
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ImportError("vector search requires numpy and fastembed") from exc

    model = TextEmbedding(model_name=model_name)
    embeddings = np.array(list(model.embed(texts, batch_size=batch_size, show_progress_bar=False)), dtype=np.float32)
    if embeddings.size:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = (embeddings / norms).astype(np.float32)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)

    np.save(graph_dir / EMBEDDINGS_FILE, embeddings)
    (graph_dir / EMBED_IDS_FILE).write_text(json.dumps(node_ids, ensure_ascii=False), encoding="utf-8")
    (graph_dir / ALIAS_INDEX_FILE).write_text(json.dumps(dict(alias_index), ensure_ascii=False), encoding="utf-8")
    try:
        stat = graph_path.stat()
        graph_stat = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    except OSError:
        graph_stat = {}
    metadata = {
        "model": model_name,
        "node_count": len(node_ids),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "dtype": str(embeddings.dtype),
        "graph": str(graph_path),
        "graph_stat": graph_stat,
    }
    (graph_dir / METADATA_FILE).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return IndexBuildResult(
        node_count=len(node_ids),
        embedding_dim=metadata["embedding_dim"],
        model_name=model_name,
        out_dir=graph_dir,
        alias_count=len(alias_index),
    )


def load_index(graph_path: str | Path, cache: dict | None = None) -> tuple[list[str], Any, dict[str, list[str]], dict[str, Any]] | None:
    graph_path = Path(graph_path).resolve()
    graph_dir = graph_path.parent
    embed_npy = graph_dir / EMBEDDINGS_FILE
    embed_ids = graph_dir / EMBED_IDS_FILE
    alias_index_path = graph_dir / ALIAS_INDEX_FILE
    metadata_path = graph_dir / METADATA_FILE
    if not embed_npy.exists() or not embed_ids.exists():
        return None
    try:
        stat = embed_npy.stat()
        key = (str(graph_dir), stat.st_mtime_ns, stat.st_size)
        if cache is not None and cache.get("key") == key:
            return cache["value"]
        import numpy as np
        embeddings = np.load(embed_npy)
        node_ids = json.loads(embed_ids.read_text(encoding="utf-8"))
        aliases = json.loads(alias_index_path.read_text(encoding="utf-8")) if alias_index_path.exists() else {}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        value = (node_ids, embeddings, aliases, metadata)
        if cache is not None:
            cache["key"] = key
            cache["value"] = value
        return value
    except Exception:
        return None


def rank_indexed_nodes(
    graph_path: str | Path,
    question: str,
    *,
    top_k: int = 5,
    cache: dict | None = None,
) -> list[tuple[float, str]]:
    loaded = load_index(graph_path, cache=cache)
    if loaded is None:
        return []
    node_ids, embeddings, aliases, metadata = loaded
    if len(node_ids) == 0:
        return []
    try:
        import numpy as np
        from fastembed import TextEmbedding
        model_name = metadata.get("model") or os.environ.get("GRAPHIFY_EMBEDDING_MODEL") or DEFAULT_MODEL_NAME
        model = TextEmbedding(model_name=model_name)
        q_embed = np.array(list(model.embed([question])))[0].astype(np.float32)
        norm = float(np.linalg.norm(q_embed))
        if norm:
            q_embed = q_embed / norm
        scores = np.dot(q_embed, embeddings.T)
    except Exception:
        scores = None

    nid_to_score: dict[str, float] = {}
    if scores is not None:
        for i, nid in enumerate(node_ids):
            nid_to_score[str(nid)] = float(scores[i])
    for alias, ids in aliases.items():
        if alias and alias in question:
            for nid in ids:
                # An explicit alias hit is stronger evidence than embedding
                # proximity. This keeps domain-neutral translations like
                # project-specific vocabulary precise when the embedding model
                # drifts toward a broader neighboring concept.
                nid_to_score[str(nid)] = nid_to_score.get(str(nid), 0.0) + 10.0

    ranked = sorted(nid_to_score.items(), key=lambda item: item[1], reverse=True)
    return [(score, nid) for nid, score in ranked[:top_k]]
