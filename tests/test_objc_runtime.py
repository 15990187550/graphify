from __future__ import annotations


def test_objc_runtime_extracts_notifications_and_header_pairs(tmp_path):
    from graphify.objc_runtime import extract_objc_runtime

    (tmp_path / "Poster.h").write_text("@interface Poster : NSObject\n@end\n", encoding="utf-8")
    (tmp_path / "Poster.m").write_text(
        """
@implementation Poster
- (void)send {
    [[NSNotificationCenter defaultCenter] postNotificationName:MyNotice object:nil];
}
@end
""",
        encoding="utf-8",
    )
    (tmp_path / "Observer.m").write_text(
        """
@implementation Observer
- (void)start {
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(handle:) name:MyNotice object:nil];
}
@end
""",
        encoding="utf-8",
    )

    result = extract_objc_runtime(tmp_path)

    relations = {e["relation"] for e in result["edges"]}
    assert "notifies" in relations
    assert "paired_with" in relations
    notify = next(e for e in result["edges"] if e["relation"] == "notifies")
    assert notify["context"] == "notification"
    assert notify["confidence"] == "INFERRED"


def test_objc_runtime_header_pairs_use_distinct_file_nodes(tmp_path):
    from graphify.objc_runtime import extract_objc_runtime

    (tmp_path / "Poster.h").write_text("@interface Poster : NSObject\n@end\n", encoding="utf-8")
    (tmp_path / "Poster.m").write_text("@implementation Poster\n@end\n", encoding="utf-8")

    result = extract_objc_runtime(tmp_path)
    nodes = {n["id"]: n for n in result["nodes"]}
    pair = next(e for e in result["edges"] if e["relation"] == "paired_with")

    assert pair["source"] != pair["target"]
    assert nodes[pair["source"]]["label"] == "Poster.h"
    assert nodes[pair["target"]]["label"] == "Poster.m"


def test_objc_runtime_extracts_property_handler_calls(tmp_path):
    from graphify.objc_runtime import extract_objc_runtime

    (tmp_path / "Helper.m").write_text("@implementation MyHandler\n@end\n", encoding="utf-8")
    (tmp_path / "Owner.h").write_text(
        "@interface Owner : NSObject\n@property (nonatomic, strong) MyHandler *messageHandler;\n@end\n",
        encoding="utf-8",
    )
    (tmp_path / "Owner.m").write_text(
        """
@implementation Owner
- (void)go {
    [self.messageHandler run];
}
@end
""",
        encoding="utf-8",
    )

    result = extract_objc_runtime(tmp_path)

    assert any(
        e["relation"] == "calls"
        and e["context"] == "call"
        and "messageHandler" in (e.get("source_location") or "")
        for e in result["edges"]
    )


def test_objc_runtime_property_handler_call_targets_class_node(tmp_path):
    from graphify.objc_runtime import extract_objc_runtime

    (tmp_path / "Helper.m").write_text("@implementation MyHandler\n@end\n", encoding="utf-8")
    (tmp_path / "Owner.h").write_text(
        "@interface Owner : NSObject\n@property (nonatomic, strong) MyHandler *messageHandler;\n@end\n",
        encoding="utf-8",
    )
    (tmp_path / "Owner.m").write_text(
        """
@implementation Owner
- (void)go {
    [self.messageHandler run];
}
@end
""",
        encoding="utf-8",
    )

    result = extract_objc_runtime(tmp_path)
    nodes = {n["id"]: n for n in result["nodes"]}
    edge = next(
        e for e in result["edges"]
        if e["relation"] == "calls" and "messageHandler" in (e.get("source_location") or "")
    )

    assert nodes[edge["source"]]["label"] == "Owner"
    assert nodes[edge["target"]]["label"] == "MyHandler"


def test_objc_runtime_does_not_infer_small_protocol_from_method_overlap(tmp_path):
    from graphify.objc_runtime import extract_objc_runtime

    (tmp_path / "TinyProtocol.h").write_text(
        """
@protocol TinyProtocol
- (void)show;
- (void)dismiss;
@end
""",
        encoding="utf-8",
    )
    (tmp_path / "UnrelatedViewController.m").write_text(
        """
@implementation UnrelatedViewController
- (void)show {}
- (void)dismiss {}
@end
""",
        encoding="utf-8",
    )

    result = extract_objc_runtime(tmp_path)

    assert not any(
        e["relation"] == "conforms_to"
        and str(e.get("source_location") or "").startswith("implicit:")
        for e in result["edges"]
    )


def test_objc_runtime_extracts_registered_class_arguments(tmp_path):
    from graphify.objc_runtime import extract_objc_runtime

    (tmp_path / "MessageRouter.m").write_text(
        """
@implementation MessageRouter
+ (void)load {
    [MessageRegistry registerHandlerClass:[ForwardHandler class] forType:1];
}
@end
""",
        encoding="utf-8",
    )
    (tmp_path / "ForwardHandler.m").write_text(
        "@implementation ForwardHandler\n@end\n",
        encoding="utf-8",
    )

    result = extract_objc_runtime(tmp_path)
    nodes = {n["id"]: n for n in result["nodes"]}
    edge = next(
        e for e in result["edges"]
        if e["relation"] == "calls"
        and "registered class: ForwardHandler" in (e.get("source_location") or "")
    )

    assert nodes[edge["source"]]["label"] == "MessageRouter"
    assert nodes[edge["target"]]["label"] == "ForwardHandler"
