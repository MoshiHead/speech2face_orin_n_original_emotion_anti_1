import json

with open("src/niloys/live/expression.py", "r", encoding="utf-8") as f:
    expr_code = f.read()

with open("tools/apply_smile_expression.py", "r", encoding="utf-8") as f:
    apply_code = f.read()

with open("runpod_speech2face_happy_expression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Build the cells to insert
patch_script = """# Ensure imtalker_try19_student_live.py has clean imports and smile expression support
import os, re

target = os.path.join(REPO_DIR, "src", "niloys", "live", "imtalker_try19_student_live.py")
if os.path.exists(target):
    with open(target, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Ensure no inner 'import sys' inside main() which causes UnboundLocalError
    content = re.sub(
        r'\\n\\s+import sys\\s*\\n\\s+here = Path\\(__file__\\)\\.resolve\\(\\)\\.parent\\s*\\n\\s+if str\\(here\\) not in sys\\.path:\\s*\\n\\s+sys\\.path\\.insert\\(0, str\\(here\\)\\)\\s*\\n\\s+from expression import apply_smile_to_base_frame',
        '\\n                from expression import apply_smile_to_base_frame',
        content
    )

    # 2. Ensure top-level imports include HERE and apply_smile_to_base_frame
    if "from expression import apply_smile_to_base_frame" not in content[:2000]:
        content = content.replace(
            "import os as _os\\n",
            "import os as _os\\nHERE = Path(__file__).resolve().parent\\n"
        )
        content = content.replace(
            "for _p in (str(IMTALKER), str(DESIGN_A)):\\n    if _p not in sys.path:\\n        sys.path.insert(0, _p)\\n",
            "for _p in (str(IMTALKER), str(DESIGN_A), str(HERE)):\\n    if _p not in sys.path:\\n        sys.path.insert(0, _p)\\n\\ntry:\\n    from expression import apply_smile_to_base_frame\\nexcept Exception:\\n    apply_smile_to_base_frame = None\\n"
        )

    # 3. Remove old noisy pred_n delta addition if present
    content = re.sub(
        r'\\s*# ---- expression blend \\(after gain, before denorm\\) ----.*?pred_n = pred_n \\+ _ed\\s*\\n',
        '\\n',
        content,
        flags=re.DOTALL
    )

    # 4. Ensure argument parser has expression_preset
    if "--expression_preset" not in content:
        content = content.replace(
            'pre.add_argument("--expression_alpha", type=float, default=1.0,\\n                     help="expression intensity multiplier (0=off, 1=full, >1=exaggerated)")',
            'pre.add_argument("--expression_alpha", type=float, default=0.0,\\n                     help="expression intensity multiplier (0=off/neutral, 1.0=natural smile)")\\n    pre.add_argument("--expression_preset", default="natural",\\n                     choices=["subtle", "natural", "happy", "broad"],\\n                     help="smile expression preset profile")'
        )

    # 5. Ensure base_frame smile warp in distilled_renderer
    if "apply_smile_to_base_frame" not in content[content.find("mine.distilled_renderer"):content.find("patch_render")]:
        old_render_block = 'D = load_distilled_renderer(mine.distilled_renderer, device)\\n        patch_render(live, D, _bl)'
        new_render_block = '''D = load_distilled_renderer(mine.distilled_renderer, device)
        if mine.expression_alpha > 0:
            if apply_smile_to_base_frame is not None:
                print(f"[try19] applying natural smile expression: alpha={mine.expression_alpha} preset={mine.expression_preset}", flush=True)
                try:
                    with torch.no_grad():
                        warped_base = apply_smile_to_base_frame(D.base_frame, alpha=mine.expression_alpha, preset=mine.expression_preset)
                        D.base_frame.copy_(warped_base)
                except Exception as e:
                    print(f"[try19] WARNING failed to apply smile expression: {e}", flush=True)
            else:
                print("[try19] WARNING apply_smile_to_base_frame module not found; running neutral", flush=True)
        patch_render(live, D, _bl)'''
        content = content.replace(old_render_block, new_render_block)

    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    print("src/niloys/live/imtalker_try19_student_live.py patched and verified.")
"""

cell_expr = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell-write-expression-module",
    "metadata": {},
    "outputs": [],
    "source": ["%%writefile src/niloys/live/expression.py\n"] + [line + "\n" for line in expr_code.split("\n")]
}

cell_apply = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell-write-apply-smile",
    "metadata": {},
    "outputs": [],
    "source": ["%%writefile tools/apply_smile_expression.py\n"] + [line + "\n" for line in apply_code.split("\n")]
}

cell_patch = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell-patch-student-live",
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in patch_script.split("\n")]
}

# Find index of cell-17 (gpu_probe.py)
probe_idx = None
for i, c in enumerate(nb["cells"]):
    if c.get("id") == "cell-17":
        probe_idx = i
        break

# Insert the cells after cell-17 if not already present
existing_ids = {c.get("id") for c in nb["cells"]}
new_cells = []
if "cell-write-expression-module" not in existing_ids:
    new_cells.append(cell_expr)
if "cell-write-apply-smile" not in existing_ids:
    new_cells.append(cell_apply)
if "cell-patch-student-live" not in existing_ids:
    new_cells.append(cell_patch)

if probe_idx is not None and new_cells:
    for idx, c in enumerate(new_cells):
        nb["cells"].insert(probe_idx + 1 + idx, c)
    print(f"Inserted {len(new_cells)} cells after gpu_probe cell.")

with open("runpod_speech2face_happy_expression.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully with embedded files and patcher!")
