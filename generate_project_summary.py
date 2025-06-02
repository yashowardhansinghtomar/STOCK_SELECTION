import os
import ast
import argparse
import json
import time
import hashlib
import re
from collections import defaultdict, Counter

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".cache", "cache/fundamentals"}

# Global counters to accumulate vocab
FEATURE_VOCAB = Counter()
TABLE_VOCAB = Counter()

def attach_parents(tree):
    for node in ast.iter_child_nodes(tree):
        node.parent = tree
        attach_parents(node)


def compute_cyclomatic_complexity(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
    except Exception:
        return 0

    complexity = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.ExceptHandler)):
            complexity += 1
    return complexity + 1


def get_docstring_coverage(tree):
    total_defs = sum(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in ast.walk(tree))
    documented = sum(ast.get_docstring(n) is not None for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
    return round((documented / total_defs) * 100, 2) if total_defs > 0 else 0


def get_missing_docstrings(tree):
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node) is None:
            if isinstance(node.parent, ast.ClassDef):
                missing.append(f"{node.parent.name}.{node.name}")
            else:
                missing.append(node.name)
        elif isinstance(node, ast.ClassDef) and ast.get_docstring(node) is None:
            missing.append(node.name)
    return missing


def hash_file_contents(filepath):
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()
    except Exception:
        return None


def get_priority(info):
    if "model_file" in info:
        return "high"
    if info.get("execution_tag") in {"planner", "agent", "model"}:
        return "medium"
    return "low"


def infer_category_and_subsystem(filepath, imports):
    lower_fp = filepath.lower()
    filename = os.path.basename(lower_fp)

    if "agents" in lower_fp or filename.endswith("_agent.py"):
        return "agent", "planning" if "planner" in lower_fp else "execution" if "execution" in lower_fp else "feedback"
    elif "models" in lower_fp or filename.endswith("_model.py"):
        return "model", "ml"
    elif "rl" in lower_fp:
        return "model", "rl"
    elif "core" in lower_fp:
        return "core", "infrastructure"
    elif "scripts" in lower_fp:
        return "script", "utility"
    elif "tests" in lower_fp:
        return "test", "qa"
    elif any("train" in imp for imp in imports):
        return "model", "training"
    return "misc", "misc"


def extract_model_names_from_calls(tree):
    models_used = []
    models_saved = []
    models_used_paths = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"load_model", "save_model"}:
                if node.args and isinstance(node.args[0], ast.Str):
                    model_name = node.args[0].s
                    if node.func.id == "load_model":
                        models_used.append(model_name)
                        if "/" in model_name or model_name.endswith(".pkl"):
                            models_used_paths.append(model_name)
                    else:
                        models_saved.append(model_name)
    return models_used, models_saved, models_used_paths


def extract_intervals_and_db_tables(tree):
    intervals = set()
    db_tables = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Str):
            val = node.s.lower()
            if val in {"day", "15minute", "60minute", "1h", "1d", "15minute"}:
                intervals.add(val)
            if re.match(r"^(training_data|model_store|stock_features|open_positions)$", val):
                db_tables.add(val)
    return sorted(intervals), sorted(db_tables)


def detect_metrics_used(content):
    return sorted(set(re.findall(r"\\b(sharpe|drawdown|win_rate|max_drawdown|sortino|cagr)\\b", content)))



def scan_project(path):
    project_data = []
    folder_summary = defaultdict(lambda: {"total_files": 0, "total_loc": 0, "total_complexity": 0})

    for dirpath, _, filenames in os.walk(path):
        if any(excl in dirpath for excl in EXCLUDED_DIRS):
            continue
        for filename in filenames:
            if filename.endswith(".py") or filename.endswith((".keras", ".zip", ".pkl")):
                full_path = os.path.join(dirpath, filename)
                parsed = parse_python_file(full_path)
                if parsed:
                    project_data.append(parsed)

                    # Folder summary update
                    folder = os.path.relpath(dirpath, path).replace("\\", "/")
                    folder_summary[folder]["total_files"] += 1
                    folder_summary[folder]["total_loc"] += parsed.get("loc", 0)
                    folder_summary[folder]["total_complexity"] += parsed.get("cyclomatic_complexity", 0)

                    # Risk flagging
                    parsed["risk_flags"] = []
                    if parsed.get("loc", 0) > 300:
                        parsed["risk_flags"].append("high_LOC")
                    if parsed.get("cyclomatic_complexity", 0) > 10:
                        parsed["risk_flags"].append("high_complexity")
                    if not parsed.get("tags"):
                        parsed["risk_flags"].append("missing_tags")
                    if not parsed.get("models_used") and not parsed.get("models_saved"):
                        parsed["risk_flags"].append("no_model_reference")

    return {"files": project_data, "folder_summary": dict(folder_summary)}


def main():
    parser = argparse.ArgumentParser(description="Generate a JSON project summary for AI consumption.")
    parser.add_argument("--path", default=".", help="Root path of the project")
    parser.add_argument("--output", default="project_summary.json", help="Output JSON file")
    args = parser.parse_args()

    result = scan_project(args.path)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    with open("project_vocab.json", "w") as vf:
        json.dump({
            "feature_vocab": FEATURE_VOCAB.most_common(),
            "table_vocab": TABLE_VOCAB.most_common()
        }, vf, indent=2)
    print("✅ Vocabulary saved to project_vocab.json")

    print(f"✅ JSON summary saved to {args.output}")

def extract_data_flow_info(tree):
    reads_from, writes_to, calls_into = set(), set(), set()
    modifies_stream = False
    transforms = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == 'load_data':
                if node.args and isinstance(node.args[0], ast.Str):
                    reads_from.add(node.args[0].s)
            elif node.func.id == 'save_data':
                if node.args and isinstance(node.args[0], ast.Str):
                    writes_to.add(node.args[0].s)
            elif node.func.id in {'merge', 'concat', 'groupby'}:
                modifies_stream = True
                transforms.add(node.func.id)
            else:
                calls_into.add(node.func.id)

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if func_name in {'merge', 'concat', 'groupby', 'transform', 'apply', 'filter'}:
                modifies_stream = True
                transforms.add(func_name)
            calls_into.add(func_name)

    return {
        "reads_from": sorted(reads_from),
        "writes_to": sorted(writes_to),
        "calls_into": sorted(calls_into),
        "modifies_stream": modifies_stream,
        "transforms": sorted(transforms)
    }


def parse_python_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            file_contents = f.read()
        except UnicodeDecodeError:
            return {}
    try:
        tree = ast.parse(file_contents)
        attach_parents(tree)
    except SyntaxError:
        return {}

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.extend(f"{module}.{alias.name}".strip(".") for alias in node.names)

    depends_on = [imp for imp in imports if imp.startswith(("core", "agents", "models", "db"))]
    file_category, subsystem = infer_category_and_subsystem(filepath, imports)

    models_used, models_saved, models_used_paths = extract_model_names_from_calls(tree)
    intervals, db_tables = extract_intervals_and_db_tables(tree)
    flow_info = extract_data_flow_info(tree)

    loc = len(file_contents.splitlines())
    complexity = compute_cyclomatic_complexity(filepath)
    doc_cov = get_docstring_coverage(tree)
    risk_score = round(loc * (complexity / 10.0) * (1 - doc_cov / 100.0), 2)
    metrics_used = detect_metrics_used(file_contents)

    info = {
        "file": filepath.replace("\\", "/"),
        "imports": imports,
        "depends_on": depends_on,
        "classes": [],
        "functions": [],
        "function_count": 0,
        "class_count": 0,
        "data_access": [],
        "loc": loc,
        "cyclomatic_complexity": complexity,
        "last_modified": time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(os.path.getmtime(filepath))),
        "sha1": hash_file_contents(filepath),
        "docstring_coverage": doc_cov,
        "risk_score": risk_score,
        "models_used": models_used,
        "models_saved": models_saved,
        "models_used_paths": models_used_paths,
        "file_category": file_category,
        "subsystem": subsystem,
        "missing_docstrings": get_missing_docstrings(tree),
        "intervals_used": intervals,
        "db_tables": db_tables,
        "metrics_used": metrics_used,
        "tags": sorted(set([file_category, subsystem] + metrics_used + intervals)),
        **flow_info
    }

    if filepath.endswith(('.keras', '.pkl', '.zip')):
        info["model_file"] = True

    class_count = 0
    function_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_count += 1
            methods = [
                {"method_name": m.name, "docstring": ast.get_docstring(m) or ""}
                for m in node.body if isinstance(m, ast.FunctionDef)
            ]
            info["classes"].append({
                "class_name": node.name,
                "docstring": ast.get_docstring(node) or "",
                "methods": methods
            })

        if isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef):
            function_count += 1
            info["functions"].append({
                "function_name": node.name,
                "docstring": ast.get_docstring(node) or ""
            })

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"load_data", "save_data"}:
                args = node.args
                keywords = {kw.arg: getattr(kw.value, 's', None) for kw in node.keywords}
                data_item = args[0].s if args and isinstance(args[0], ast.Str) else None
                access = {
                    "type": node.func.id,
                    "target": data_item,
                    "source": keywords.get("source") or keywords.get("destination") or "default"
                }
                info["data_access"].append(access)

    info["function_count"] = function_count
    info["class_count"] = class_count

    fname = filepath.lower()
    if "agent" in fname:
        info["execution_tag"] = "agent"
    elif "model" in fname or "train" in fname:
        info["execution_tag"] = "model"
    elif "test" in fname or "pytest" in imports:
        info["execution_tag"] = "test"

    info["file_role_priority"] = get_priority(info)

    return info

def extract_additional_data_signals(file_contents):
    input_tables, output_tables, used_features = set(), set(), set()

    lines = file_contents.lower().splitlines()

    for line in lines:
        # Table detection by pattern
        tables = re.findall(r"[\'\"]([a-zA-Z0-9_]+_table|training_data|model_store|stock_features|open_positions)[\'\"]", line)
        for tbl in tables:
            if 'from' in line or 'read' in line:
                input_tables.add(tbl)
                TABLE_VOCAB[tbl] += 1
            elif 'into' in line or 'write' in line or 'save' in line:
                output_tables.add(tbl)
                TABLE_VOCAB[tbl] += 1

        # Feature usage detection from variable names or strings
        feature_mentions = re.findall(r'feature\[\"(\w+)\"\]|\b(rsi|sma|macd|ema|vwap|adx|obv|bollinger|ichimoku)\b', line)
        for match in feature_mentions:
            feature = match[0] or match[1]
            if feature:
                used_features.add(feature)
                FEATURE_VOCAB[feature] += 1

    return sorted(input_tables), sorted(output_tables), sorted(used_features)


if __name__ == "__main__":
    main()
