import os
import ast
import argparse
import json
import time
import hashlib
import re
from collections import defaultdict, Counter

# Global counters for the separate vocabulary file
FEATURE_VOCAB = Counter()
TABLE_VOCAB = Counter()

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # Provide defaults for flags if not present
        config.setdefault('flags', {})
        config['flags'].setdefault('include_docstrings', True)
        config['flags'].setdefault('include_code_metrics', True)
        config['flags'].setdefault('include_full_code_structure', False)
        config['flags'].setdefault('generate_vocab_file', True)
        config.setdefault('risk_thresholds', {'loc': 300, 'complexity': 15})
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found. Please create it.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{config_path}' contains invalid JSON.")
        exit(1)

def attach_parents(tree):
    for node in ast.iter_child_nodes(tree):
        node.parent = tree
        attach_parents(node)

def compute_cyclomatic_complexity(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
    except Exception:
        return 0

    complexity = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.ExceptHandler, ast.AsyncFor)):
            complexity += 1
        elif isinstance(node, ast.BoolOp) and node.op in (ast.And, ast.Or):
            complexity += len(node.values) -1
        elif isinstance(node, ast.comprehension): # list/dict/set comprehensions, generator expressions
            complexity +=1

    return complexity + 1


def get_docstring_info(tree):
    total_defs = 0
    documented_defs = 0
    missing_docstrings_list = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            total_defs += 1
            docstring = ast.get_docstring(node)
            if docstring:
                documented_defs += 1
            else:
                parent_name = ""
                if hasattr(node, 'parent') and isinstance(node.parent, ast.ClassDef):
                    parent_name = node.parent.name + "."
                missing_docstrings_list.append(f"{parent_name}{node.name}")
        elif isinstance(node, ast.Module): # Module-level docstring
            total_defs +=1
            if ast.get_docstring(node):
                documented_defs +=1
            else:
                missing_docstrings_list.append("module-level")


    coverage = round((documented_defs / total_defs) * 100, 2) if total_defs > 0 else 0
    return coverage, missing_docstrings_list


def hash_file_contents(filepath):
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()
    except Exception:
        return None

def get_qualified_call_name(node_func):
    if isinstance(node_func, ast.Name):
        return node_func.id
    elif isinstance(node_func, ast.Attribute):
        # Recursively build the qualified name
        # e.g., for 'module.submodule.function()', value is 'module.submodule', attr is 'function'
        value_str = get_qualified_call_name(node_func.value)
        if value_str:
            return f"{value_str}.{node_func.attr}"
        return node_func.attr # Fallback if value is complex
    elif isinstance(node_func, ast.Call): # e.g. a().b()
        return get_qualified_call_name(node_func.func)
    # Add more types if needed e.g. ast.Subscript
    return None


def extract_general_data_flow(tree):
    calls_into = set()
    transforms_applied = set()
    modifies_stream = False # Heuristic: if common data transformation functions are called

    common_transforms = {'merge', 'concat', 'groupby', 'transform', 'apply', 'filter', 'pivot_table', 'resample', 'rolling', 'sort_values', 'drop_duplicates'}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            qualified_name = get_qualified_call_name(node.func)
            if qualified_name:
                calls_into.add(qualified_name)
                # Check if the call is a common transform method
                # This is a heuristic and might need refinement based on common libraries (pandas, spark)
                method_name = qualified_name.split('.')[-1]
                if method_name in common_transforms:
                    transforms_applied.add(method_name)
                    modifies_stream = True

    return sorted(list(calls_into)), sorted(list(transforms_applied)), modifies_stream


def extract_configured_io_calls(tree, config):
    io_calls = defaultdict(list)
    configured_funcs = config.get('io_functions', {})

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            qualified_name = get_qualified_call_name(node.func)
            if not qualified_name:
                continue

            for io_type, func_list in configured_funcs.items():
                if qualified_name in func_list:
                    # Try to get the first string argument as the source/target
                    source_arg = None
                    if node.args and isinstance(node.args[0], ast.Str):
                        source_arg = node.args[0].s
                    elif node.args and isinstance(node.args[0], ast.Name): # e.g. load_data(filepath_variable)
                         source_arg = f"var:{node.args[0].id}"

                    io_calls[io_type].append({
                        "function_called": qualified_name,
                        "target_argument": source_arg,
                        "line_number": node.lineno
                    })
                    break # Found its type, move to next call
    return dict(io_calls)


def extract_db_tables_and_features(file_contents, config):
    db_tables = set()
    features = set()

    db_patterns = config.get('db_table_patterns', [])
    feature_patterns = config.get('feature_patterns', [])

    for line in file_contents.splitlines():
        for pattern in db_patterns:
            try:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    # If pattern has capturing groups, re.findall returns tuples or strings
                    tbl = match if isinstance(match, str) else match[0] if match else None
                    if tbl:
                        db_tables.add(tbl)
                        if config.get('flags', {}).get('generate_vocab_file', True):
                             TABLE_VOCAB[tbl] +=1
            except re.error as e:
                print(f"Warning: Regex error in db_table_pattern '{pattern}': {e}")


        for pattern in feature_patterns:
            try:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    feat = match if isinstance(match, str) else match[0] if match else None # Adapt if patterns use groups
                    if feat:
                        features.add(feat)
                        if config.get('flags', {}).get('generate_vocab_file', True):
                            FEATURE_VOCAB[feat] += 1
            except re.error as e:
                print(f"Warning: Regex error in feature_pattern '{pattern}': {e}")


    return sorted(list(db_tables)), sorted(list(features))


def parse_python_file(filepath, config):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_contents = f.read()
    except UnicodeDecodeError:
        print(f"Warning: Could not decode file {filepath}, skipping.")
        return None
    except Exception as e:
        print(f"Warning: Could not read file {filepath}: {e}, skipping.")
        return None

    try:
        tree = ast.parse(file_contents)
        attach_parents(tree) # For parent navigation if needed
    except SyntaxError:
        # For non-python files or syntax errors, return minimal info
        if not filepath.endswith(".py"): # Handle non-python files gracefully
             return {
                "file": filepath.replace("\\", "/"),
                "error": "Not a Python file or syntax error, basic info only.",
                "loc": len(file_contents.splitlines()),
                "sha1": hash_file_contents(filepath),
                "last_modified": time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(os.path.getmtime(filepath)))
             }
        print(f"Warning: Syntax error in {filepath}, skipping AST analysis.")
        return None


    info = {"file": filepath.replace("\\", "/")}

    # Core connection fields
    raw_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raw_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if node.level > 0: # Relative import
                # Attempt to resolve relative import based on file path
                # This is a simplified resolution
                base_path_parts = os.path.dirname(filepath).split(os.sep)
                level = node.level
                if level > len(base_path_parts): level = len(base_path_parts) # cap level
                
                # For `from .. import X`, module_name is empty. For `from ..moduleY import X`, module_name is `moduleY`
                # We want to construct the absolute path from project root perspective.
                # Assuming project_root_packages are direct children of args.path or similar.
                # This needs a robust way to map file path to module path relative to project roots.
                # For now, just prefixing with dots for AI to potentially interpret.
                prefix = "." * node.level
                raw_imports.extend(f"{prefix}{module_name}.{alias.name}".strip(".") for alias in node.names)
            else: # Absolute import
                 raw_imports.extend(f"{module_name}.{alias.name}".strip(".") for alias in node.names)
    info["imports"] = sorted(list(set(raw_imports))) # Deduplicate

    project_dependencies = []
    project_roots = tuple(config.get('project_root_packages', []))
    if project_roots: # only if project_root_packages are defined
        for imp in info["imports"]:
            if imp.startswith(project_roots):
                project_dependencies.append(imp)
    info["project_dependencies"] = sorted(list(set(project_dependencies)))


    calls, transforms, modifies_stream = extract_general_data_flow(tree)
    info["generic_function_calls"] = calls
    info["transforms_applied"] = transforms
    info["modifies_stream"] = modifies_stream

    configured_io = extract_configured_io_calls(tree, config)
    for io_type, calls_list in configured_io.items():
        info[f"{io_type}_operations"] = calls_list


    db_tables, features = extract_db_tables_and_features(file_contents, config)
    info["db_tables_identified"] = db_tables
    info["features_identified"] = features


    # Optional detailed code structure
    if config['flags'].get('include_full_code_structure', False):
        info["classes"] = []
        info["functions"] = []
        function_count = 0
        class_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_count +=1
                methods = []
                if config['flags'].get('include_docstrings', True): # Only get method docstrings if main docstrings are on
                    methods = [{"name": m.name, "docstring": ast.get_docstring(m) or "", "line_number": m.lineno}
                               for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                else:
                     methods = [{"name": m.name, "line_number": m.lineno}
                               for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                class_info = {"name": node.name, "methods": methods, "line_number": node.lineno}
                if config['flags'].get('include_docstrings', True):
                    class_info["docstring"] = ast.get_docstring(node) or ""
                info["classes"].append(class_info)
            elif isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef): # Top-level functions
                function_count +=1
                func_info = {"name": node.name, "line_number": node.lineno}
                if config['flags'].get('include_docstrings', True):
                    func_info["docstring"] = ast.get_docstring(node) or ""
                info["functions"].append(func_info)
        info["function_count"] = function_count
        info["class_count"] = class_count
    else: # Still provide counts if not full structure
        info["function_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef))
        info["class_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))


    # Optional metrics
    if config['flags'].get('include_code_metrics', True):
        info["loc"] = len(file_contents.splitlines())
        info["cyclomatic_complexity"] = compute_cyclomatic_complexity(filepath)
        # Risk score calculation needs docstring coverage if docstrings are enabled
        doc_cov_for_risk = 0
        if config['flags'].get('include_docstrings', True):
            doc_cov, _ = get_docstring_info(tree)
            doc_cov_for_risk = doc_cov
            info["docstring_coverage"] = doc_cov # Store it if calculated
            info["missing_docstrings"] = _
        else: # if docstrings are off, can't use it for risk, assume 0 coverage for risk calculation
            info["docstring_coverage"] = -1 # Indicate not calculated
            info["missing_docstrings"] = []


        # Simplified risk score: loc * complexity * (1 - doc_coverage_fraction)
        # Avoid division by zero if complexity is low, make risk proportional
        risk_numerator = info["loc"] * info["cyclomatic_complexity"] * max(0.05, (1 - doc_cov_for_risk / 100.0))
        info["risk_score"] = round(risk_numerator / 10.0, 2) # Keep similar scale to original

    # Optional docstrings (if not already handled by metrics block)
    elif config['flags'].get('include_docstrings', True): # if metrics are off but docstrings are on
        info["docstring_coverage"], info["missing_docstrings"] = get_docstring_info(tree)
    else: # if both metrics and docstrings are off
        info["docstring_coverage"] = -1
        info["missing_docstrings"] = []


    # Other general info
    info["sha1"] = hash_file_contents(filepath)
    try:
        info["last_modified"] = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(os.path.getmtime(filepath)))
    except Exception:
        info["last_modified"] = None


    # Custom tags from path
    custom_tags = []
    path_tags_config = config.get('custom_tags_from_path', [])
    normalized_filepath = filepath.replace("\\", "/")
    for item in path_tags_config:
        if item.get("pattern") and item.get("tag"):
            try:
                if re.search(item["pattern"], normalized_filepath, re.IGNORECASE):
                    custom_tags.append(item["tag"])
            except re.error as e:
                 print(f"Warning: Regex error in custom_tags_from_path pattern '{item['pattern']}': {e}")
    info["custom_tags"] = sorted(list(set(custom_tags)))

    return info


def scan_project(path, config):
    project_data = []
    folder_summary = defaultdict(lambda: {"total_files": 0, "total_loc": 0, "total_complexity": 0, "total_risk": 0.0, "files_in_folder": []})
    excluded_dirs = set(config.get('excluded_dirs', []))

    for dirpath, dirnames, filenames in os.walk(path):
        # Modify dirnames in-place to exclude directories
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs and not os.path.join(dirpath, d) in excluded_dirs]

        # Check if current dirpath itself should be excluded (e.g., project_root/.git/logs)
        if any(excl_dir in dirpath.replace("\\", "/") for excl_dir in excluded_dirs if excl_dir.startswith(tuple(d + "/" for d in dirnames)) or excl_dir == os.path.basename(dirpath)):
            continue


        for filename in filenames:
            # Allow analysis of various file types if they might contain relevant info,
            # but full AST parsing only for .py
            # For now, let's stick to .py for primary analysis, but config could extend this.
            if filename.endswith(".py"): # Or other configurable extensions
                full_path = os.path.join(dirpath, filename)
                parsed_info = parse_python_file(full_path, config)
                if parsed_info:
                    project_data.append(parsed_info)

                    # Folder summary update (only if metrics are enabled)
                    # And if the file wasn't just a basic info dict for non-py/error files
                    if config['flags'].get('include_code_metrics', True) and "loc" in parsed_info:
                        rel_folder = os.path.relpath(dirpath, path).replace("\\", "/")
                        if rel_folder == ".": rel_folder = "root"

                        folder_summary[rel_folder]["total_files"] += 1
                        folder_summary[rel_folder]["total_loc"] += parsed_info.get("loc", 0)
                        folder_summary[rel_folder]["total_complexity"] += parsed_info.get("cyclomatic_complexity", 0)
                        folder_summary[rel_folder]["total_risk"] += parsed_info.get("risk_score", 0)
                        folder_summary[rel_folder]["files_in_folder"].append(parsed_info["file"])


    # Add risk flags based on configured thresholds (if metrics are enabled)
    if config['flags'].get('include_code_metrics', True):
        loc_threshold = config.get('risk_thresholds', {}).get('loc', 300)
        complexity_threshold = config.get('risk_thresholds', {}).get('complexity', 15)

        for file_info in project_data:
            if "loc" not in file_info: continue # Skip if basic info only
            file_info["risk_flags"] = []
            if file_info.get("loc", 0) > loc_threshold:
                file_info["risk_flags"].append(f"high_LOC (>{loc_threshold})")
            if file_info.get("cyclomatic_complexity", 0) > complexity_threshold:
                file_info["risk_flags"].append(f"high_complexity (>{complexity_threshold})")
            if not file_info.get("custom_tags") and config.get("custom_tags_from_path"): # If tagging is configured but none found
                file_info["risk_flags"].append("missing_custom_tags")

            # Check for absence of configured I/O operations
            has_io = False
            for io_type in config.get('io_functions', {}).keys():
                if file_info.get(f"{io_type}_operations"):
                    has_io = True
                    break
            if not has_io and config.get('io_functions'): # If IO functions are configured but none used
                file_info["risk_flags"].append("no_configured_io_calls")


    return {"files": project_data, "folder_summary": dict(folder_summary)}


def main():
    parser = argparse.ArgumentParser(description="Generate a JSON project summary for AI consumption, focusing on connections.")
    parser.add_argument("--path", default=".", help="Root path of the project (default: current directory)")
    parser.add_argument("--output", default="project_summary.json", help="Output JSON file for project data (default: project_summary.json)")
    parser.add_argument("--config", default="analyzer_config.json", help="Path to the JSON configuration file (default: analyzer_config.json)")
    parser.add_argument("--vocab_output", default="project_vocab.json", help="Output JSON file for vocabularies (default: project_vocab.json)")

    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Starting project analysis for: {os.path.abspath(args.path)}")
    print(f"Using configuration: {os.path.abspath(args.config)}")

    project_analysis_result = scan_project(args.path, config)

    # Create the main project summary file
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(project_analysis_result, f, indent=2)
        print(f"✅ Project summary saved to {args.output}")
    except Exception as e:
        print(f"Error: Could not write project summary to {args.output}: {e}")


    # Create the vocabulary file if enabled
    if config.get('flags', {}).get('generate_vocab_file', True):
        if FEATURE_VOCAB or TABLE_VOCAB:
            vocab_data = {
                "feature_vocab": FEATURE_VOCAB.most_common(),
                "table_vocab": TABLE_VOCAB.most_common()
            }
            try:
                with open(args.vocab_output, "w", encoding='utf-8') as vf:
                    json.dump(vocab_data, vf, indent=2)
                print(f"✅ Vocabulary saved to {args.vocab_output}")
            except Exception as e:
                print(f"Error: Could not write vocabulary to {args.vocab_output}: {e}")

        else:
            print("ℹ️ No features or tables found to generate vocabulary file.")

    print("Analysis complete.")

if __name__ == "__main__":
    main()