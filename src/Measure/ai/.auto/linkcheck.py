"""Resolve every relative markdown link/image target in the repo-root docs.

Fenced code blocks and inline code spans are stripped first, so an example such as
``[PNG](x.png)`` written inside backticks is not mistaken for a real link.

Usage:  python .auto/linkcheck.py [baseline.json]
        If the baseline file does not exist it is created from the current state
        and no link is reported as new (self-bootstrapping, no empty argument).
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ["README.md", "RESULTS.md", "SCALING_GUIDE.md", "neq.md", "REPORT.md"]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"```.*?```", re.S)
CODE = re.compile(r"`[^`]*`")

broken = []
checked = 0
for doc in DOCS:
    path = os.path.join(ROOT, doc)
    if not os.path.isfile(path):
        continue
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    text = FENCE.sub("", text)
    text = CODE.sub("", text)
    for target in LINK.findall(text):
        t = target.strip()
        if t.startswith("<") and t.endswith(">"):
            t = t[1:-1]
        t = t.split("#", 1)[0].split("?", 1)[0]
        if not t or "://" in t or t.startswith("mailto:"):
            continue
        checked += 1
        resolved = os.path.normpath(os.path.join(ROOT, t.replace("/", os.sep)))
        if not os.path.exists(resolved):
            broken.append(t)

broken = sorted(set(broken))
new_broken = []

if len(sys.argv) > 1:
    baseline = sys.argv[1]
    if os.path.isfile(baseline):
        with open(baseline, encoding="utf-8") as fh:
            known = set(json.load(fh).get("broken", []))
        new_broken = [b for b in broken if b not in known]
    else:
        with open(baseline, "w", encoding="utf-8") as fh:
            json.dump({"broken": broken}, fh, indent=1)

print("METRIC md_links_checked=%d" % checked)
print("METRIC md_links_broken=%d" % len(broken))
print("METRIC md_new_broken=%d" % len(new_broken))
for b in new_broken[:20]:
    print("  NEW BROKEN:", b)
