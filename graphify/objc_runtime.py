from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from graphify.extract import _file_stem, _make_id


_SYSTEM_PREFIXES = (
    "NS", "UI", "CA", "CG", "MK", "CF", "WK", "PK", "SK", "CL", "HK", "EK",
    "CN", "IN", "MP", "QL", "SC", "SF", "UN", "AV",
)

_BUILTIN_TYPES = {
    "id", "BOOL", "NSInteger", "NSUInteger", "CGFloat", "int", "float", "double",
    "char", "long", "short", "void", "SEL", "Class", "NSString", "NSArray",
    "NSDictionary", "NSSet", "NSNumber", "NSData", "NSDate", "NSURL", "NSError",
    "UIImage", "UIColor", "UIFont", "UIView", "UIViewController", "UITableView",
    "UICollectionView", "UIScrollView", "UILabel", "UIButton", "UIImageView",
    "UITextField", "UITextView",
}


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _node_id(rel_path: str) -> str:
    p = Path(rel_path)
    return _make_id(p.parent.name, p.stem)


def _file_node_id(rel_path: str) -> str:
    return _make_id(rel_path)


def _symbol_node_id(rel_path: str, symbol_name: str) -> str:
    return _make_id(_file_stem(Path(rel_path)), symbol_name)


def _class_or_protocol_is_system(name: str) -> bool:
    return name.startswith(_SYSTEM_PREFIXES)


def _iter_objc_files(root: Path, extra_excludes: list[str] | None = None) -> list[Path]:
    excludes = set(extra_excludes or [])
    files: list[Path] = []
    for ext in ("*.m", "*.mm", "*.h"):
        for path in root.glob(f"**/{ext}"):
            text = path.as_posix()
            if "/graphify-out/" in text or "/Pods/" in text or "/.git/" in text:
                continue
            if any(ex and ex in text for ex in excludes):
                continue
            files.append(path)
    return sorted(files)


def extract_objc_runtime(root: str | Path, *, extra_excludes: list[str] | None = None) -> dict:
    """Infer cross-file Objective-C runtime relationships from source text.

    This complements the tree-sitter extractor. It intentionally emits
    INFERRED edges only: notifications, delegate/protocol links, categories,
    property-handler calls, factory registrations, and header/implementation
    pairing are conventions rather than direct AST calls.
    """
    root = Path(root).resolve()
    notification_posts: dict[str, list[str]] = defaultdict(list)
    notification_observes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    singleton_defs: dict[str, str] = {}
    singleton_usages: dict[str, list[str]] = defaultdict(list)
    delegate_cons: dict[str, list[str]] = defaultdict(list)
    delegate_defs: dict[str, list[str]] = defaultdict(list)
    callback_sets: dict[str, list[str]] = defaultdict(list)
    callback_regs: dict[str, list[str]] = defaultdict(list)
    file_property_types: dict[str, dict[str, str]] = defaultdict(dict)
    property_usages: dict[str, list[str]] = defaultdict(list)
    file_class_name: dict[str, str] = {}
    protocol_defs: dict[str, str] = {}
    protocol_cons: dict[str, list[tuple[str, str]]] = defaultdict(list)
    category_classes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    factory_regs: dict[str, list[str]] = defaultdict(list)
    protocol_methods: dict[str, set[str]] = defaultdict(set)
    impl_methods_by_class: dict[tuple[str, str], set[str]] = defaultdict(set)
    file_primary_symbol: dict[str, str] = {}

    for path in _iter_objc_files(root, extra_excludes=extra_excludes):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel_path = _rel(path, root)
        is_header = path.suffix == ".h"
        is_impl = path.suffix in (".m", ".mm")

        for match in re.finditer(r"@property\s*\([^)]*\)\s*(\w+(?:\s*<[^>]+>)?)\s*\*?\s*(\w+)\s*;", content):
            type_name = match.group(1).strip()
            prop_name = match.group(2)
            clean_type = type_name.split("<", 1)[0].strip()
            if clean_type and clean_type not in _BUILTIN_TYPES and clean_type[0].isupper():
                file_property_types[rel_path][prop_name] = clean_type

        for match in re.finditer(r"@implementation\s+(\w+)", content):
            cls = match.group(1)
            file_class_name[rel_path] = cls
            file_primary_symbol[rel_path] = cls

        for match in re.finditer(r"@interface\s+(\w+)\b(?!\s*\()", content):
            cls = match.group(1)
            if not _class_or_protocol_is_system(cls):
                file_primary_symbol.setdefault(rel_path, cls)

        if is_impl:
            for match in re.finditer(r"(?:self\.|_)(\w+(?:Handler|Manager|Center|Helper|Delegate|Builder|Provider|DataSource))\b", content):
                property_usages[rel_path].append(match.group(1))

        for match in re.finditer(r"(?:postNotificationName|postNotification\w*):\s*([A-Za-z_]\w*)", content):
            notification_posts[match.group(1)].append(rel_path)
        for match in re.finditer(r"addObserver\w*:.*?name:\s*([A-Za-z_]\w*)", content, re.S):
            selector = re.search(r"selector:\s*@selector\(([^)]+)\)", match.group(0))
            notification_observes[match.group(1)].append((rel_path, selector.group(1) if selector else ""))

        if re.search(r"\+\s*\(instancetype\)\s*shared\w+", content):
            for match in re.finditer(r"@implementation\s+(\w+)", content):
                singleton_defs[match.group(1)] = rel_path
        for match in re.finditer(r"\[(\w+(?:Manager|Center|Service|Helper|Handler))\s+shared\w*\]", content):
            singleton_usages[match.group(1)].append(rel_path)

        for match in re.finditer(r"<([^>]+)>", content):
            for proto in (p.strip() for p in match.group(1).split(",")):
                if proto and not _class_or_protocol_is_system(proto):
                    delegate_cons[proto].append(rel_path)
        for match in re.finditer(r"@property\s*\([^)]*\)\s*(?:id<(\w+)>\s*|(\w+)\s*\*)\s*(\w*delegate\w*)", content):
            proto = match.group(1) or match.group(2) or ""
            if proto:
                delegate_defs[proto].append(rel_path)

        for match in re.finditer(r"@protocol\s+(\w+)", content):
            proto = match.group(1)
            if not _class_or_protocol_is_system(proto):
                protocol_defs[proto] = rel_path
                file_primary_symbol.setdefault(rel_path, proto)
        for match in re.finditer(r"@interface\s+(\w+)\s*(?::\s*\w+\s*)?<([^>]+)>", content):
            cls = match.group(1)
            for proto in (p.strip() for p in match.group(2).split(",")):
                if proto and not _class_or_protocol_is_system(proto):
                    protocol_cons[proto].append((rel_path, cls))

        for match in re.finditer(r"@(?:interface|implementation)\s+(\w+)\s*\((\w+)\)", content):
            category_classes[match.group(1)].append((rel_path, match.group(2)))

        for match in re.finditer(r"\+\s*\(void\)\s*load\s*\{", content):
            body = content[match.end(): match.end() + 2000]
            for reg in re.finditer(r"\[(\w+)\s+(?:regist|register)\w*:", body):
                factory_regs[reg.group(1)].append(rel_path)
            for reg in re.finditer(r"\[\[?(\w+)\s+(?:shared\w*|default\w*|sharedInstance|standard\w*|getInstance)\]*\]?\s+(\w*[Rr]egist\w*)", body):
                factory_regs[reg.group(1)].append(rel_path)

        for match in re.finditer(r"Set(\w+Callback)\s*\(", content):
            callback_sets[match.group(1)].append(rel_path)
        for match in re.finditer(r"\.(\w+Callback)\s*=", content):
            callback_sets[match.group(1)].append(rel_path)
        for match in re.finditer(r"nbase::Bind\s*\(\s*&(\w+)", content):
            callback_regs[match.group(1)].append(rel_path)

        if is_header:
            current_proto: str | None = None
            for line in content.splitlines():
                start = re.match(r"\s*@protocol\s+(\w+)", line)
                if start:
                    current_proto = start.group(1)
                    continue
                if re.match(r"\s*@end\b", line):
                    current_proto = None
                    continue
                if current_proto:
                    for method in re.finditer(r"[-+]\s*\([^)]*\)\s*(\w+)", line):
                        protocol_methods[current_proto].add(method.group(1))
                    for prop in re.finditer(r"@property\s*\([^)]*\)\s*\w+(?:\s*<[^>]+>)?\s*\*?\s*(\w+)", line):
                        protocol_methods[current_proto].add(prop.group(1))

        if is_impl:
            current_class = Path(rel_path).stem
            for line in content.splitlines():
                impl = re.match(r"\s*@implementation\s+(\w+)", line)
                if impl:
                    current_class = impl.group(1)
                for method in re.finditer(r"[-+]\s*\([^)]*\)\s*(\w+)", line):
                    impl_methods_by_class[(rel_path, current_class)].add(method.group(1))

    class_to_file = {cls: rel_path for rel_path, cls in file_class_name.items()}
    class_to_file.update({cls: rel_path for cls, rel_path in singleton_defs.items() if cls not in class_to_file})

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str, str | None]] = set()

    def add_node(rel_path: str) -> str:
        symbol = file_primary_symbol.get(rel_path)
        node_id = _symbol_node_id(rel_path, symbol) if symbol else _node_id(rel_path)
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": symbol or Path(rel_path).name,
                "file_type": "code",
                "source_file": rel_path,
                "source_location": None,
            }
        return node_id

    def add_file_node(rel_path: str) -> str:
        node_id = _file_node_id(rel_path)
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": Path(rel_path).name,
                "file_type": "code",
                "source_file": rel_path,
                "source_location": None,
            }
        return node_id

    def add_edge(src_file: str, tgt_file: str, relation: str, score: float, location: str | None,
                 context: str, *, file_nodes: bool = False) -> None:
        if not src_file or not tgt_file or src_file == tgt_file:
            return
        src_id = add_file_node(src_file) if file_nodes else add_node(src_file)
        tgt_id = add_file_node(tgt_file) if file_nodes else add_node(tgt_file)
        if src_id == tgt_id and relation != "paired_with":
            return
        key = (src_id, tgt_id, relation, location)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({
            "source": src_id,
            "target": tgt_id,
            "relation": relation,
            "confidence": "INFERRED",
            "confidence_score": score,
            "source_file": src_file,
            "source_location": location,
            "weight": 1.0,
            "context": context,
        })

    for name, posters in notification_posts.items():
        for observer_file, selector in notification_observes.get(name, []):
            for post_file in posters:
                loc = f"NSNotification: {name}" + (f" -> {selector}" if selector else "")
                add_edge(post_file, observer_file, "notifies", 0.85, loc, "notification")

    for cls, users in singleton_usages.items():
        definition = singleton_defs.get(cls)
        if definition:
            for user in users:
                add_edge(user, definition, "calls", 0.85, f"sharedInstance: {cls}", "call")

    for proto, conformers in delegate_cons.items():
        base = proto.replace("Delegate", "").replace("Protocol", "")
        for conformer in conformers:
            for definition in delegate_defs.get(proto, []) + delegate_defs.get(base, []):
                add_edge(conformer, definition, "implements", 0.75, f"conforms to {proto}", "protocol")

    for cb_name, cb_files in callback_sets.items():
        reg_name = cb_name.replace("Set", "")
        for reg_file in callback_regs.get(reg_name, []) + callback_regs.get(cb_name, []):
            for cb_file in cb_files:
                add_edge(reg_file, cb_file, "calls", 0.85, f"callback: {cb_name}", "call")

    for rel_path, usages in property_usages.items():
        base = re.sub(r"\.(mm|m)$", "", rel_path)
        for prop_name in usages:
            prop_type = None
            for candidate in (rel_path, base + ".h", base + ".mm", base + ".m"):
                prop_type = file_property_types.get(candidate, {}).get(prop_name)
                if prop_type:
                    break
            target = class_to_file.get(prop_type or "")
            if target:
                add_edge(rel_path, target, "calls", 0.85, f"property: self.{prop_name} ({prop_type})", "call")

    for proto, conformers in protocol_cons.items():
        proto_file = protocol_defs.get(proto)
        if proto_file:
            for conformer, _class_name in conformers:
                add_edge(conformer, proto_file, "conforms_to", 0.8, f"@protocol {proto}", "protocol")

    for cls, category_files in category_classes.items():
        target = class_to_file.get(cls)
        if target:
            for category_file, category_name in category_files:
                add_edge(category_file, target, "extends", 0.7, f"category: {cls}({category_name})", "extension")

    for registry, registrants in factory_regs.items():
        target = class_to_file.get(registry) or singleton_defs.get(registry)
        if target:
            for registrant in registrants:
                add_edge(registrant, target, "registers_in", 0.75, f"+load -> [{registry} register]", "registration")

    explicit = {(e["source"], e["target"]) for e in edges if e["relation"] == "conforms_to"}
    for proto, required in protocol_methods.items():
        if len(required) < 3:
            continue
        proto_file = protocol_defs.get(proto)
        if not proto_file:
            continue
        threshold = max(3, int(len(required) * 0.6))
        for (rel_path, _cls), methods in impl_methods_by_class.items():
            overlap = methods & required
            src_id = add_node(rel_path)
            tgt_id = add_node(proto_file)
            if len(overlap) >= threshold and (src_id, tgt_id) not in explicit:
                add_edge(rel_path, proto_file, "conforms_to", 0.6, f"implicit: {len(overlap)}/{len(required)} methods match @protocol {proto}", "protocol")
                explicit.add((src_id, tgt_id))

    by_base: dict[str, list[str]] = defaultdict(list)
    for path in _iter_objc_files(root, extra_excludes=extra_excludes):
        rel_path = _rel(path, root)
        by_base[re.sub(r"\.(h|mm|m|cc|cpp)$", "", rel_path)].append(rel_path)
    for variants in by_base.values():
        headers = [p for p in variants if p.endswith(".h")]
        impls = [p for p in variants if p.endswith((".m", ".mm", ".cc", ".cpp"))]
        for header in headers:
            for impl in impls:
                add_edge(header, impl, "paired_with", 0.95, None, "pairing", file_nodes=True)

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }


def write_runtime_edges(root: str | Path, out_path: str | Path, *, extra_excludes: list[str] | None = None) -> dict:
    result = extract_objc_runtime(root, extra_excludes=extra_excludes)
    Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
