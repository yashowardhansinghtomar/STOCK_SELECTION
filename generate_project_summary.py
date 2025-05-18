#!/usr/bin/env python3
from core.logger import logger

import os
import ast
import argparse
import textwrap

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".cache", "cache/fundamentals"}


def parse_python_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            file_contents = f.read()
        except UnicodeDecodeError:
            return {}
    try:
        tree = ast.parse(file_contents)
    except SyntaxError:
        return {}

    info = {"imports": [], "classes": [], "functions": []}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                info["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            for alias in node.names:
                info["imports"].append(f"{module}.{alias.name}".strip("."))

        if isinstance(node, ast.ClassDef):
            class_docstring = ast.get_docstring(node) or ""
            methods = []
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef):
                    method_doc = ast.get_docstring(body_item) or ""
                    methods.append({"method_name": body_item.name, "docstring": method_doc})
            info["classes"].append({"class_name": node.name, "docstring": class_docstring, "methods": methods})

        if isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef):
            func_docstring = ast.get_docstring(node) or ""
            info["functions"].append({"function_name": node.name, "docstring": func_docstring})

    return info


def attach_parents(tree):
    for node in ast.iter_child_nodes(tree):
        node.parent = tree
        attach_parents(node)


def extract_data_access_summary(path):
    data_access = []
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                    except Exception:
                        continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in {"load_data", "save_data"}:
                            args = node.args
                            keywords = {kw.arg: getattr(kw.value, 's', None) for kw in node.keywords}
                            data_item = args[0].s if args and isinstance(args[0], ast.Str) else None
                            source_dest = keywords.get("source") or keywords.get("destination") or "default"
                            access = f"{func_name}('{data_item}', {source_dest})"
                            data_access.append((filepath.replace("\\", "/"), access))
    return data_access


def build_tree_and_extract(path, prefix="", is_last=True):
    lines = []
    basename = os.path.basename(path)
    connector = "└── " if is_last else "├── "

    # Skip full content of excluded folders
    if os.path.isdir(path):
        if basename in EXCLUDED_DIRS or path.replace("\\", "/") in EXCLUDED_DIRS:
            lines.append(prefix + connector + basename + "/  [directory excluded]")
            return lines
    else:
        # Skip specific file types completely
        if path.endswith((".pyc", ".csv", ".json")):
            return []  # completely ignore them, not even their name

    # Show current folder/file name
    lines.append(prefix + connector + basename)

    # Recurse into directories
    if os.path.isdir(path):
        entries = sorted(os.listdir(path))
        entries = [e for e in entries if not e.startswith('.')]

        skip_count = 0
        filtered_entries = []

        for e in entries:
            full = os.path.join(path, e)
            if os.path.isdir(full):
                filtered_entries.append(e)
            elif not e.endswith((".pyc", ".csv", ".json")):
                filtered_entries.append(e)
            else:
                skip_count += 1

        if skip_count:
            lines.append(prefix + ("    " if is_last else "│   ") +
                         f"📄 Skipped {skip_count} data files (.csv, .json, .pyc)")

        for i, entry in enumerate(filtered_entries):
            fullpath = os.path.join(path, entry)
            is_last_entry = (i == len(filtered_entries) - 1)
            sub_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(build_tree_and_extract(fullpath, sub_prefix, is_last_entry))

    # Parse only Python files
    elif path.endswith(".py"):
        file_info = parse_python_file(path)
        indent = prefix + ("    " if is_last else "│   ")

        if file_info.get("imports"):
            lines.append(indent + "Imports:")
            for imp in sorted(set(file_info["imports"])):
                lines.append(indent + f"  - {imp}")

        for c in file_info.get("classes", []):
            lines.append(indent + f"Class: {c['class_name']}")
            if c['docstring']:
                wrapped_doc = textwrap.indent(textwrap.fill(c['docstring'], width=60), indent + "  ")
                lines.append(indent + "  Docstring:")
                lines.append(wrapped_doc)
            if c['methods']:
                lines.append(indent + "  Methods:")
                for m in c['methods']:
                    lines.append(indent + f"    - {m['method_name']}")
                    if m['docstring']:
                        wrapped_mdoc = textwrap.indent(textwrap.fill(m['docstring'], width=60), indent + "      ")
                        lines.append(indent + "      Docstring:")
                        lines.append(wrapped_mdoc)

        for f in file_info.get("functions", []):
            lines.append(indent + f"Function: {f['function_name']}")
            if f['docstring']:
                wrapped_fdoc = textwrap.indent(textwrap.fill(f['docstring'], width=60), indent + "  ")
                lines.append(indent + "  Docstring:")
                lines.append(wrapped_fdoc)

    return lines



def main():
    parser = argparse.ArgumentParser(description="Generate a tree-like project summary with Python file details.")
    parser.add_argument("--path", default=".", help="Path to the root folder of the project.")
    parser.add_argument("--output", default="project_summary.txt", help="Name of the output file.")
    args = parser.parse_args()

    root_path = os.path.abspath(args.path)

    final_lines = [f"Project Structure & Python File Details for: {root_path}"]

    final_lines.append("\n🧠 Detected Data Access Points:")
    data_access = extract_data_access_summary(root_path)
    for filepath, call in sorted(data_access):
        rel_path = os.path.relpath(filepath, root_path).replace("\\", "/")
        final_lines.append(f"• {rel_path} → {call}")

    final_lines.append("\n📁 Full Directory Summary with Code Info:")
    final_lines.extend(build_tree_and_extract(root_path))

    output_file = os.path.join(root_path, args.output)
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(line + "\n")

    logger.info(f"✅ Summary created: {output_file}")


if __name__ == "__main__":
    real_ast_parse = ast.parse

    def parse_with_parent(*args, **kwargs):
        tree = real_ast_parse(*args, **kwargs)
        attach_parents(tree)
        return tree

    ast.parse = parse_with_parent
    main()
