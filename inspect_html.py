import re

with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Find occurrences of select or dropdown menus, or mode settings
matches = re.findall(r'<select.*?>.*?</select>', html, re.DOTALL)
print("Found dropdown menus:")
for idx, match in enumerate(matches):
    print(f"Dropdown {idx+1}:\n{match}\n")

# Find any references to "Calibration" or "Database" or similar settings
lines = html.splitlines()
print("Lines containing 'Calibration':")
for idx, line in enumerate(lines):
    if "Calibration" in line or "calibration" in line or "โหมดหลัก" in line:
        print(f"Line {idx+1}: {line.strip()}")
