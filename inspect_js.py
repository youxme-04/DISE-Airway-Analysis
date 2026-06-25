import re

with open("script.js", "r", encoding="utf-8") as f:
    js = f.read()

# Find occurrences of 'scan-mode' or 'scanMode' in the JS file
lines = js.splitlines()
print("Lines containing 'scan-mode' or 'scanMode':")
for idx, line in enumerate(lines):
    if "scan-mode" in line or "scanMode" in line or "scan_mode" in line:
        print(f"Line {idx+1}: {line.strip()}")
