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


def test_build_index_reuses_unchanged_node_embeddings(tmp_path, monkeypatch):
    from graphify.search_index import build_index

    graph_path = tmp_path / "graph.json"
    _write_graph(
        graph_path,
        [
            {"id": "alpha", "label": "AlphaManager", "source_file": "alpha.py"},
            {"id": "beta", "label": "BetaManager", "source_file": "beta.py"},
        ],
    )
    calls: list[list[str]] = []

    def install_counting_fastembed():
        fake = types.ModuleType("fastembed")

        class TextEmbedding:
            def __init__(self, model_name: str):
                self.model_name = model_name

            def embed(self, texts, **_kwargs):
                batch = list(texts)
                calls.append(batch)
                for text in batch:
                    if "AlphaManager" in text:
                        yield np.array([1.0, 0.0], dtype=np.float32)
                    elif "BetaManagerV2" in text:
                        yield np.array([0.0, 0.5], dtype=np.float32)
                    else:
                        yield np.array([0.0, 1.0], dtype=np.float32)

        fake.TextEmbedding = TextEmbedding
        monkeypatch.setitem(sys.modules, "fastembed", fake)

    install_counting_fastembed()
    first = build_index(graph_path, model_name="fake/model")
    assert first.reused_count == 0
    assert first.embedded_count == 2

    _write_graph(
        graph_path,
        [
            {"id": "alpha", "label": "AlphaManager", "source_file": "alpha.py"},
            {"id": "beta", "label": "BetaManagerV2", "source_file": "beta.py"},
        ],
    )
    second = build_index(graph_path, model_name="fake/model")

    assert second.reused_count == 1
    assert second.embedded_count == 1
    assert calls[-1] and len(calls[-1]) == 1
    assert "BetaManagerV2" in calls[-1][0]
    ids = json.loads((tmp_path / ".graphify_embed_ids.json").read_text(encoding="utf-8"))
    embeddings = np.load(tmp_path / ".graphify_embeddings.npy")
    assert ids == ["alpha", "beta"]
    np.testing.assert_allclose(embeddings[0], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(embeddings[1], np.array([0.0, 1.0], dtype=np.float32))


def test_build_index_drops_deleted_nodes_incrementally(tmp_path, monkeypatch):
    from graphify.search_index import build_index

    graph_path = tmp_path / "graph.json"
    _write_graph(
        graph_path,
        [
            {"id": "alpha", "label": "AlphaManager", "source_file": "alpha.py"},
            {"id": "beta", "label": "BetaManager", "source_file": "beta.py"},
        ],
    )
    _install_fake_fastembed(monkeypatch, [[1.0, 0.0], [0.0, 1.0]])
    build_index(graph_path, model_name="fake/model")

    _write_graph(
        graph_path,
        [
            {"id": "alpha", "label": "AlphaManager", "source_file": "alpha.py"},
        ],
    )
    result = build_index(graph_path, model_name="fake/model")

    assert result.reused_count == 1
    assert result.embedded_count == 0
    ids = json.loads((tmp_path / ".graphify_embed_ids.json").read_text(encoding="utf-8"))
    embeddings = np.load(tmp_path / ".graphify_embeddings.npy")
    assert ids == ["alpha"]
    assert embeddings.shape == (1, 2)


def test_alias_rule_change_reembeds_affected_nodes(tmp_path, monkeypatch):
    from graphify.search_index import build_index

    graph_path = tmp_path / "graph.json"
    _write_graph(
        graph_path,
        [
            {"id": "transpond", "label": "TranspondHandler", "source_file": "Session/Transpond/Foo.mm"},
        ],
    )
    calls: list[list[str]] = []
    fake = types.ModuleType("fastembed")

    class TextEmbedding:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def embed(self, texts, **_kwargs):
            batch = list(texts)
            calls.append(batch)
            for text in batch:
                if "转发" in text:
                    yield np.array([0.0, 1.0], dtype=np.float32)
                else:
                    yield np.array([1.0, 0.0], dtype=np.float32)

    fake.TextEmbedding = TextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake)

    first = build_index(graph_path, model_name="fake/model")
    assert first.embedded_count == 1

    (tmp_path / "graphify.aliases.json").write_text(
        json.dumps({"Transpond": ["转发"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    second = build_index(graph_path, model_name="fake/model")

    assert second.reused_count == 0
    assert second.embedded_count == 1
    assert "转发" in calls[-1][0]
    aliases = json.loads((tmp_path / ".graphify_alias_index.json").read_text(encoding="utf-8"))
    assert aliases["转发"] == ["transpond"]


def test_model_change_forces_full_reembed(tmp_path, monkeypatch):
    from graphify.search_index import build_index

    graph_path = tmp_path / "graph.json"
    _write_graph(
        graph_path,
        [
            {"id": "alpha", "label": "AlphaManager", "source_file": "alpha.py"},
            {"id": "beta", "label": "BetaManager", "source_file": "beta.py"},
        ],
    )
    _install_fake_fastembed(monkeypatch, [[1.0, 0.0], [0.0, 1.0]])
    build_index(graph_path, model_name="fake/model")

    result = build_index(graph_path, model_name="fake/model-v2")

    assert result.reused_count == 0
    assert result.embedded_count == 2


def test_load_index_rejects_stale_graph_stat(tmp_path, monkeypatch):
    from graphify.search_index import build_index, load_index

    graph_path = tmp_path / "graph.json"
    _write_graph(
        graph_path,
        [
            {"id": "alpha", "label": "AlphaManager", "source_file": "alpha.py"},
        ],
    )
    _install_fake_fastembed(monkeypatch, [[1.0, 0.0]])
    build_index(graph_path, model_name="fake/model")

    _write_graph(
        graph_path,
        [
            {"id": "alpha", "label": "AlphaManager", "source_file": "alpha.py"},
            {"id": "beta", "label": "BetaManager", "source_file": "beta.py"},
        ],
    )

    assert load_index(graph_path) is None


def test_load_index_rejects_stale_alias_rules(tmp_path, monkeypatch):
    from graphify.search_index import build_index, load_index

    graph_path = tmp_path / "graph.json"
    _write_graph(
        graph_path,
        [
            {"id": "transpond", "label": "TranspondHandler", "source_file": "Session/Transpond/Foo.mm"},
        ],
    )
    _install_fake_fastembed(monkeypatch, [[1.0, 0.0]])
    build_index(graph_path, model_name="fake/model")

    (tmp_path / "graphify.aliases.json").write_text(
        json.dumps({"Transpond": ["转发"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    assert load_index(graph_path) is None
