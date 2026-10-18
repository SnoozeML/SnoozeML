#!/usr/bin/env python3
"""SnoozeML PreToolUse safety gate hook.

Blocks dangerous destructive commands (such as deleting .git, forced git reset,
or bulk root file deletion) to protect repository integrity during agent sessions.
"""

import json
import re
import sys

DANGEROUS_PATTERNS = [
    re.compile(r"\brmdir\b.*(?:\.git|\\\.git|/\.git)", re.IGNORECASE),
    re.compile(r"\brm\b\s+-[rf]*\s+.*\.git", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\s+-[xfd]{2,}", re.IGNORECASE),
    re.compile(r"\bRemove-Item\b.*\.git", re.IGNORECASE),
    re.compile(r"\bdel\b\s+/[sfq]+\s+\*\.\*", re.IGNORECASE),
]

def main() -> None:
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print(json.dumps({"decision": "allow"}))
            return
        data = json.loads(raw_input)
    except Exception:
        # Fallback to allow if stdin cannot be parsed
        print(json.dumps({"decision": "allow"}))
        return

    tool_call = data.get("toolCall", {})
    name = tool_call.get("name", "")
    args = tool_call.get("args", {})

    if name == "run_command":
        cmd = args.get("CommandLine", "")
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(cmd):
                print(
                    json.dumps(
                        {
                            "decision": "deny",
                            "reason": f"Command blocked by SnoozeML safety-gate: dangerous command matching '{pattern.pattern}'.",
                        }
                    )
                )
                return

    print(json.dumps({"decision": "allow"}))

if __name__ == "__main__":
    main()
