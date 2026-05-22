from __future__ import annotations

import json
import sys
import types

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph


def _install_fake_fastembed(monkeypatch, vectors):
    fake = types.ModuleType("fastembed")

    class TextEmbedding:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def embed(self, texts, **_kwargs):
            for i, _text in enumerate(texts):
                yield np.array(vectors[min(i, len(vectors) - 1)], dtype=np.float32)

    fake.TextEmbedding = TextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake)


def _write_graph(path, nodes):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    try:
        data = json_graph.node_link_data(G, edges="links")
    except TypeError:
        data = json_graph.node_link_data(G)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_index_writes_float32_embeddings_and_aliases(tmp_path, monkeypatch):
    from graphify.search_index import build_index

    graph_path = tmp_path / "graph.json"
    _write_graph(
        graph_path,
        [
            {
                "id": "session_manager",
                "label": "POPOSessionManager",
                "source_file": "Session/Manager/POPOSessionManager.mm",
                "source_location": None,
            }
        ],
    )
    (tmp_path / "graphify.aliases.json").write_text(
        json.dumps({"Session/Manager": ["会话", "会话管理"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    _install_fake_fastembed(monkeypatch, [[1.0, 0.0, 0.0]])

    result = build_index(graph_path, model_name="fake/model")

    assert result.node_count == 1
    embeddings = np.load(tmp_path / ".graphify_embeddings.npy")
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (1, 3)
    aliases = json.loads((tmp_path / ".graphify_alias_index.json").read_text(encoding="utf-8"))
    assert "会话" in aliases
    assert aliases["会话"] == ["session_manager"]


def test_custom_alias_rules_match_identifier_labels(tmp_path, monkeypatch):
    from graphify.search_index import build_index

    graph_path = tmp_path / "graph.json"
    _write_graph(
        graph_path,
        [
            {
                "id": "transpond",
                "label": "PopoSessionTranspondMessageHandler",
                "source_file": "Handler/PopoSessionTranspondMessageHandler.mm",
                "source_location": None,
            }
        ],
    )
    (tmp_path / "graphify.aliases.json").write_text(
        json.dumps({"Transpond": ["转发"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    _install_fake_fastembed(monkeypatch, [[1.0, 0.0]])

    build_index(graph_path, model_name="fake/model")

    aliases = json.loads((tmp_path / ".graphify_alias_index.json").read_text(encoding="utf-8"))
    assert aliases["转发"] == ["transpond"]


def test_query_prefers_exact_identifier_over_embedding_hit(tmp_path, monkeypatch):
    import graphify.serve as serve
    from graphify.serve import _load_graph, _query_graph_text

    graph_path = tmp_path / "graph.json"
    _write_graph(
        graph_path,
        [
            {"id": "exact", "label": "ExactThing", "source_file": "exact.py"},
            {"id": "semantic", "label": "SemanticOnly", "source_file": "semantic.py"},
        ],
    )
    np.save(tmp_path / ".graphify_embeddings.npy", np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    (tmp_path / ".graphify_embed_ids.json").write_text(json.dumps(["exact", "semantic"]), encoding="utf-8")
    (tmp_path / ".graphify_alias_index.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".graphify_search_index.json").write_text(json.dumps({"model": "fake/model"}), encoding="utf-8")
    _install_fake_fastembed(monkeypatch, [[1.0, 0.0]])
    serve._EMBEDDING_CACHE.clear()

    text = _query_graph_text(_load_graph(str(graph_path)), "ExactThing", depth=0)

    assert "Start: ['ExactThing']" in text


def test_chinese_alias_beats_vague_embedding_match(tmp_path, monkeypatch):
    from graphify.search_index import rank_indexed_nodes

    graph_path = tmp_path / "graph.json"
    _write_graph(
        graph_path,
        [
            {
                "id": "transpond",
                "label": "PopoSessionTranspondMessageHandler",
                "source_file": "Session/Transpond/PopoSessionTranspondMessageHandler.mm",
            },
            {
                "id": "send",
                "label": "PopoNewSendMessageManager",
                "source_file": "NewSendMsg/PopoNewSendMessageManager.mm",
            },
        ],
    )
    np.save(tmp_path / ".graphify_embeddings.npy", np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    (tmp_path / ".graphify_embed_ids.json").write_text(json.dumps(["transpond", "send"]), encoding="utf-8")
    (tmp_path / ".graphify_alias_index.json").write_text(
        json.dumps({"转发": ["transpond"], "发送": ["send"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / ".graphify_search_index.json").write_text(json.dumps({"model": "fake/model"}), encoding="utf-8")
    _install_fake_fastembed(monkeypatch, [[1.0, 0.0]])

    ranked = rank_indexed_nodes(graph_path, "消息转发入口", top_k=2)

    assert ranked[0][1] == "transpond"
