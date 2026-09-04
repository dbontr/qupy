from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path


def _public(name: str) -> bool:
    return not name.startswith("_")


def _declared_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and _public(
            node.name
        ):
            names.add(node.name)
    return names


def _class_map(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and _public(node.name)
    }


def _member_names(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for member in node.body:
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _public(member.name) or member.name == "__init__":
                names.add(member.name)
        elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
            if _public(member.target.id):
                names.add(member.target.id)
        elif isinstance(member, ast.Assign):
            for target in member.targets:
                if isinstance(target, ast.Name) and _public(target.id):
                    names.add(target.id)
    return names


def _callables(
    nodes: list[ast.stmt],
) -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    result: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = defaultdict(list)
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            _public(node.name) or node.name == "__init__"
        ):
            result[node.name].append(node)
    return dict(result)


def _shape(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[object, ...]:
    positional = [*node.args.posonlyargs, *node.args.args]
    default_start = len(positional) - len(node.args.defaults)
    positional_defaults = tuple(index >= default_start for index in range(len(positional)))
    return (
        tuple(arg.arg for arg in node.args.posonlyargs),
        tuple(arg.arg for arg in node.args.args),
        positional_defaults,
        node.args.vararg.arg if node.args.vararg is not None else None,
        tuple(arg.arg for arg in node.args.kwonlyargs),
        tuple(default is not None for default in node.args.kw_defaults),
        node.args.kwarg.arg if node.args.kwarg is not None else None,
    )


def _missing_shapes(
    expected: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]],
    actual: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]],
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    for name, expected_nodes in sorted(expected.items()):
        if name not in actual:
            continue
        actual_shapes = {_shape(node) for node in actual[name]}
        for node in expected_nodes:
            shape = _shape(node)
            if shape not in actual_shapes:
                errors.append(f"{prefix}{name} has no checked signature matching {shape!r}")
    return errors


def verify(checked_path: Path, generated_path: Path) -> None:
    checked = ast.parse(checked_path.read_text(encoding="utf-8"), filename=str(checked_path))
    generated = ast.parse(
        generated_path.read_text(encoding="utf-8"),
        filename=str(generated_path),
    )

    errors: list[str] = []
    missing = sorted(_declared_names(generated) - _declared_names(checked))
    errors.extend(f"missing public native declaration: {name}" for name in missing)

    checked_classes = _class_map(checked)
    generated_classes = _class_map(generated)
    for name, expected_class in sorted(generated_classes.items()):
        actual_class = checked_classes.get(name)
        if actual_class is None:
            continue
        missing_members = sorted(_member_names(expected_class) - _member_names(actual_class))
        errors.extend(
            f"missing native member: {name}.{member}" for member in missing_members
        )
        errors.extend(
            _missing_shapes(
                _callables(expected_class.body),
                _callables(actual_class.body),
                f"{name}.",
            )
        )

    errors.extend(_missing_shapes(_callables(generated.body), _callables(checked.body), ""))
    if errors:
        raise SystemExit("native stub conformance failed:\n- " + "\n- ".join(errors))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the checked QuPy native stub with nanobind-generated runtime metadata"
    )
    parser.add_argument("checked", type=Path)
    parser.add_argument("generated", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    verify(args.checked, args.generated)
    print("native stub conformance passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
