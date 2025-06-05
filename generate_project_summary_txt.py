import os
import ast
import argparse
import textwrap

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".cache", "cache", "venv", ".venv"}

class SimpleLogger:
    def info(self, msg): print(f"ℹ️  {msg}")

logger = SimpleLogger()

def attach_parents(node, parent=None):
    for child in ast.iter_child_nodes(node):
        child.parent = parent
        attach_parents(child, child)

def parse_python_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except UnicodeDecodeError:
        logger.info(f"Skipping file due to encoding error: {filepath}")
        return {}

    try:
        tree = ast.parse(source)
        attach_parents(tree)
    except SyntaxError as e:
        logger.info(f"Skipping file due to syntax error: {filepath} — {e}")
        return {}

    imports, classes, functions = [], [], []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.extend(f"{module}.{alias.name}" for alias in node.names)

        if isinstance(node, ast.ClassDef):
            classes.append({
                "class_name": node.name,
                "docstring": ast.get_docstring(node) or "",
                "methods": [
                    {"method_name": m.name, "docstring": ast.get_docstring(m) or ""}
                    for m in node.body if isinstance(m, ast.FunctionDef)
                ]
            })
        elif isinstance(node, ast.FunctionDef) and not any(isinstance(p, ast.ClassDef) for p in iter_parents(node)):
            functions.append({
                "function_name": node.name,
                "docstring": ast.get_docstring(node) or ""
            })

    return {"imports": imports, "classes": classes, "functions": functions}

def iter_parents(node):
    while hasattr(node, 'parent'):
        node = node.parent
        yield node

def extract_data_access_summary(path):
    def resolve_arg(arg):
        if isinstance(arg, ast.Constant):  # Python 3.8+
            return repr(arg.value)
        elif isinstance(arg, ast.Str):  # Python < 3.8
            return repr(arg.s)
        elif isinstance(arg, ast.Attribute):
            parts = []
            while isinstance(arg, ast.Attribute):
                parts.append(arg.attr)
                arg = arg.value
            if isinstance(arg, ast.Name):
                parts.append(arg.id)
            return '.'.join(reversed(parts))
        elif isinstance(arg, ast.Name):
            return arg.id
        return "UNKNOWN_EXPR"

    results = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source)
            except Exception as e:
                logger.info(f"Could not parse {filepath}: {e}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    func = node.func.id
                    if func in {"load_data", "save_data"}:
                        arg = node.args[0] if node.args else None
                        arg_value = resolve_arg(arg) if arg else "MISSING_ARG"
                        context = "default"
                        for kw in node.keywords:
                            if kw.arg in {"source", "destination"}:
                                context = resolve_arg(kw.value)
                        results.append((filepath.replace("\\", "/"), f"{func}({arg_value}, source_dest={context})"))
    return results


def build_tree(path, prefix="", is_last=True):
    lines = []
    name = os.path.basename(path)
    connector = "└── " if is_last else "├── "
    lines.append(prefix + connector + name)

    if os.path.isdir(path):
        try:
            entries = sorted(e for e in os.listdir(path) if not e.startswith('.') and e not in EXCLUDED_DIRS)
        except Exception:
            return lines

        subprefix = prefix + ("    " if is_last else "│   ")
        for i, entry in enumerate(entries):
            full = os.path.join(path, entry)
            lines.extend(build_tree(full, subprefix, i == len(entries) - 1))
    elif path.endswith(".py"):
        info = parse_python_file(path)
        subprefix = prefix + ("    " if is_last else "│   ")

        if info.get("imports"):
            lines.append(subprefix + "Imports:")
            lines.extend(subprefix + f"  - {imp}" for imp in sorted(set(info["imports"])))

        for cls in info.get("classes", []):
            lines.append(subprefix + f"Class: {cls['class_name']}")
            if cls["docstring"]:
                lines.append(subprefix + "  Docstring:")
                lines.extend(textwrap.wrap(cls["docstring"], width=70, initial_indent=subprefix + "    ", subsequent_indent=subprefix + "    "))
            if cls["methods"]:
                lines.append(subprefix + "  Methods:")
                for m in cls["methods"]:
                    lines.append(subprefix + f"    - {m['method_name']}")
                    if m["docstring"]:
                        lines.append(subprefix + "      Docstring:")
                        lines.extend(textwrap.wrap(m["docstring"], width=65, initial_indent=subprefix + "        ", subsequent_indent=subprefix + "        "))

        for func in info.get("functions", []):
            lines.append(subprefix + f"Function: {func['function_name']}")
            if func["docstring"]:
                lines.append(subprefix + "  Docstring:")
                lines.extend(textwrap.wrap(func["docstring"], width=70, initial_indent=subprefix + "    ", subsequent_indent=subprefix + "    "))

    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".", help="Project root directory")
    parser.add_argument("--output", default="project_summary.txt", help="Output file")
    args = parser.parse_args()

    abs_path = os.path.abspath(args.path)

    logger.info("🔍 Scanning for data access points...")
    data_points = extract_data_access_summary(abs_path)

    logger.info("📂 Building file tree...")
    lines = [f"Project Summary for: {abs_path}", "", "🧠 Data Access Points:"]
    if data_points:
        lines.extend([f"• {os.path.relpath(p, abs_path)} → {call}" for p, call in data_points])
    else:
        lines.append("  No 'load_data' or 'save_data' calls found.")

    lines.append("\n📁 Code Structure Overview:")
    lines.extend(build_tree(abs_path))

    with open(os.path.join(abs_path, args.output), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"✅ Summary written to {args.output}")

if __name__ == "__main__":
    main()
