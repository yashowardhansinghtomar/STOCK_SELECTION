#!/usr/bin/env python3

import os
import ast
import argparse
import textwrap

# --- START: Simplified Logger for Demonstration ---
# In a real scenario, you would fix core/logger.py to avoid recursion.
# This simple logger prevents the RecursionError for this script.
class SimpleLogger:
    def info(self, msg, prefix=""):
        print(f"{prefix}{msg}")

logger = SimpleLogger()
# --- END: Simplified Logger ---


EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".cache", "cache/fundamentals"}


def parse_python_file(filepath):
    """
    Parses a Python file to extract imports, classes (with methods and docstrings),
    and functions (with docstrings).

    Args:
        filepath (str): The path to the Python file.

    Returns:
        dict: A dictionary containing parsed information (imports, classes, functions).
              Returns an empty dictionary if the file cannot be read or parsed.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            file_contents = f.read()
        except UnicodeDecodeError:
            logger.info(f"Skipping file due to UnicodeDecodeError: {filepath}")
            return {}
    try:
        tree = ast.parse(file_contents)
    except SyntaxError as e:
        logger.info(f"Skipping file due to SyntaxError: {filepath} - {e}")
        return {}

    info = {"imports": [], "classes": [], "functions": []}

    # Attach parent nodes for easier traversal (used by attach_parents later)
    # Note: ast.walk already handles traversal, but parent links are useful for context.
    # The attach_parents function is called globally for the main AST tree.

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # Handle 'import module'
            for alias in node.names:
                info["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Handle 'from module import name'
            module = node.module if node.module else ""
            for alias in node.names:
                info["imports"].append(f"{module}.{alias.name}".strip("."))

        if isinstance(node, ast.ClassDef):
            class_docstring = ast.get_docstring(node) or ""
            methods = []
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef):
                    # Check if it's a method (a FunctionDef within a ClassDef)
                    method_doc = ast.get_docstring(body_item) or ""
                    methods.append({"method_name": body_item.name, "docstring": method_doc})
            info["classes"].append({"class_name": node.name, "docstring": class_docstring, "methods": methods})

        # Check for functions that are NOT methods (i.e., top-level functions)
        # The 'node.parent' check assumes attach_parents has been run.
        if isinstance(node, ast.FunctionDef):
            # Ensure it's not a method of a class
            is_method = False
            current = node.parent
            while current:
                if isinstance(current, ast.ClassDef):
                    is_method = True
                    break
                # Only traverse up if parent is not the module itself
                if not hasattr(current, 'parent'): # Reached the top of the tree
                    break
                current = current.parent

            if not is_method:
                func_docstring = ast.get_docstring(node) or ""
                info["functions"].append({"function_name": node.name, "docstring": func_docstring})

    return info


def attach_parents(tree):
    """
    Recursively attaches a 'parent' attribute to each node in the AST,
    pointing to its parent node. This is useful for contextual analysis.
    """
    for node in ast.iter_child_nodes(tree):
        node.parent = tree
        attach_parents(node)


def extract_data_access_summary(path):
    """
    Extracts summary of data access points (e.g., calls to 'load_data', 'save_data')
    from Python files within the given path.

    Args:
        path (str): The root directory to search for Python files.

    Returns:
        list: A list of tuples, where each tuple contains (filepath, data_access_call_string).
    """
    data_access = []
    for dirpath, dirnames, filenames in os.walk(path):
        # Modify dirnames in-place to skip excluded directories for os.walk
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                    except Exception as e:
                        logger.info(f"Could not parse {filepath} for data access: {e}")
                        continue
                for node in ast.walk(tree):
                    # Look for function calls
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in {"load_data", "save_data"}:
                            args = node.args
                            keywords = {kw.arg: getattr(kw.value, 's', None) for kw in node.keywords}
                            
                            data_item = None
                            if args and isinstance(args[0], (ast.Str, ast.Constant)):
                                # For Python 3.8+, ast.Str is deprecated, ast.Constant is used for literals
                                data_item = args[0].s if isinstance(args[0], ast.Str) else args[0].value

                            source_dest = keywords.get("source") or keywords.get("destination")
                            if source_dest is None:
                                source_dest = "default" # Fallback if source/destination not found

                            access = f"{func_name}('{data_item}', source_dest='{source_dest}')"
                            data_access.append((filepath.replace("\\", "/"), access))
    return data_access


def build_tree_and_extract(path, prefix="", is_last=True):
    """
    Recursively builds a tree-like representation of the project structure,
    extracting and displaying details for Python files (imports, classes, functions, docstrings).

    Args:
        path (str): The current path (file or directory).
        prefix (str): The prefix string for indentation in the tree.
        is_last (bool): True if the current item is the last sibling in its directory.

    Returns:
        list: A list of strings, each representing a line in the project summary.
    """
    lines = []
    basename = os.path.basename(path)
    connector = "└── " if is_last else "├── "

    # Skip full content of excluded folders
    if os.path.isdir(path):
        if basename in EXCLUDED_DIRS:
            lines.append(prefix + connector + basename + "/ 📄 [directory excluded]")
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
        # Filter out hidden files/directories and explicitly excluded ones
        entries = [e for e in entries if not e.startswith('.') and e not in EXCLUDED_DIRS]

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
            # Indicate skipped data files
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
                # Wrap docstrings for better readability
                wrapped_doc = textwrap.indent(textwrap.fill(c['docstring'], width=70), indent + "    ")
                lines.append(indent + "  Docstring:")
                lines.append(wrapped_doc)
            if c['methods']:
                lines.append(indent + "  Methods:")
                for m in c['methods']:
                    lines.append(indent + f"    - {m['method_name']}")
                    if m['docstring']:
                        wrapped_mdoc = textwrap.indent(textwrap.fill(m['docstring'], width=70), indent + "      ")
                        lines.append(indent + "      Docstring:")
                        lines.append(wrapped_mdoc)

        for f in file_info.get("functions", []):
            lines.append(indent + f"Function: {f['function_name']}")
            if f['docstring']:
                wrapped_fdoc = textwrap.indent(textwrap.fill(f['docstring'], width=70), indent + "    ")
                lines.append(indent + "  Docstring:")
                lines.append(wrapped_fdoc)

    return lines


def main():
    """
    Main function to parse command-line arguments, build the project summary,
    and write it to an output file.
    """
    parser = argparse.ArgumentParser(description="Generate a tree-like project summary with Python file details.")
    parser.add_argument("--path", default=".", help="Path to the root folder of the project.")
    parser.add_argument("--output", default="project_summary.txt", help="Name of the output file.")
    args = parser.parse_args()

    root_path = os.path.abspath(args.path)

    final_lines = [f"Project Structure & Python File Details for: {root_path}\n"]

    logger.info("🧠 Detecting Data Access Points...")
    final_lines.append("🧠 Detected Data Access Points:")
    data_access = extract_data_access_summary(root_path)
    if data_access:
        for filepath, call in sorted(data_access):
            rel_path = os.path.relpath(filepath, root_path).replace("\\", "/")
            final_lines.append(f"• {rel_path} → {call}")
    else:
        final_lines.append("  No explicit 'load_data' or 'save_data' calls found.")

    final_lines.append("\n📁 Full Directory Summary with Code Info:")
    logger.info("📁 Building Directory Summary...")
    final_lines.extend(build_tree_and_extract(root_path))

    output_file = os.path.join(root_path, args.output)
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(line + "\n")

    logger.info(f"✅ Summary created: {output_file}")


if __name__ == "__main__":
    # Temporarily override ast.parse to attach parent nodes for all parsed trees.
    # This is crucial for correctly identifying top-level functions vs. methods.
    real_ast_parse = ast.parse

    def parse_with_parent(*args, **kwargs):
        tree = real_ast_parse(*args, **kwargs)
        attach_parents(tree)
        return tree

    ast.parse = parse_with_parent
    main()
