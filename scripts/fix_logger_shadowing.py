import os
import re

PROJECT_ROOT = "."  # Or set your absolute project path here
LOG_METHODS = ["info", "warning", "error", "debug", "critical"]

# Matches logger.info(f"{...}"+str("some msg"))
# Captures the method (info/warning/etc), the message, and the prefix
pattern = re.compile(
    r"""logger\.(?P<method>{})\((?P<msg>[^,]+),\s*prefix\s*=\s*(?P<prefix>[^)]+)\)""".format("|".join(LOG_METHODS))
)

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    matches = list(pattern.finditer(content))
    if not matches:
        return False

    new_content = content
    for match in reversed(matches):  # reversed to not mess up offsets
        method = match.group("method")
        msg = match.group("msg").strip()
        prefix = match.group("prefix").strip()

        replacement = f'logger.{method}(f"{{{prefix}}}"+str({msg}))'
        new_content = (
            new_content[:match.start()] +
            replacement +
            new_content[match.end():]
        )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"✅ Fixed logger prefix usage in: {filepath}")
    return True

def scan_project():
    for dirpath, _, filenames in os.walk(PROJECT_ROOT):
        for filename in filenames:
            if filename.endswith(".py"):
                process_file(os.path.join(dirpath, filename))

if __name__ == "__main__":
    scan_project()
    print("\n🎉 Prefix usage fixed in all logger.* calls.")
