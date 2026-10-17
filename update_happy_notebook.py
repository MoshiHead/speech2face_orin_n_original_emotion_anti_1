import json

notebook_path = "runpod_speech2face_happy_expression.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update cell-00 (Title and overview)
for cell in nb["cells"]:
    if cell.get("id") == "cell-00":
        cell["source"] = [
            "# speech2face on RunPod \u2014 Happy Expression Edition\n",
            "\n",
            "Everything from the original `runpod_speech2face.ipynb` \u2014 PersonaPlex 7B + IMTalker real-time\n",
            "talking avatar \u2014 **with natural smiling happy expression** that preserves 100% of the avatar's\n",
            "photorealistic identity and speech lip-sync dynamics.\n",
            "\n",
            "### How Natural Expression Works\n",
            "\n",
            "The avatar renderer is a distilled single-identity architecture:\n",
            "$$\\text{output} = (\\text{base\\_frame} + \\text{residual\\_network}(\\text{motion})).\\text{clamp}(0, 1)$$\n",
            "\n",
            "- **`base_frame`** defines the facial identity, resting geometry, and expression (neutral vs smiling).\n",
            "- **`residual_network`** synthesizes lip opening, closing, and phoneme speech articulation on top of the base frame.\n",
            "\n",
            "By applying an elastic spatial smile deformation to `base_frame` (lifting lip corners, raising cheeks, and gentle eye crinkling), the avatar smiles naturally with:\n",
            "- **100% Identity Preservation**: Zero textures are modified; the avatar's native facial skin, eyes, and hair are intact.\n",
            "- **Accurate Lip Synchronization**: The residual network articulates phonemes seamlessly on top of the smiling face.\n",
            "- **Zero Runtime Overhead**: The deformation is applied once at server startup (0 ms latency during streaming).\n",
            "- **Zero Noise / Mottling**: Eliminates the out-of-distribution pixel noise caused by direct latent injection.\n",
            "\n",
            "### What is included in this notebook\n",
            "\n",
            "| feature | what it does |\n",
            "|---|---|\n",
            "| **Smile Calibration & Preview** | Calibrate and preview the natural happy smile with side-by-side visualization in ~2 seconds |\n",
            "| **Configurable Intensity** | Adjust `EXPR_ALPHA` (0=neutral, 0.6=subtle, 1.0=natural, 1.3=happy) or presets (`subtle`, `natural`, `happy`, `broad`) |\n",
            "| **Real-time Live Server** | Full PersonaPlex 7B voice + video conversational avatar streaming at ~25 fps |\n",
            "| **Offline Audio Test** | Upload any WAV file, synthesize avatar talking-head video with audio, and verify smiling lip sync |\n",
            "| **Side-by-Side Comparison** | Render both Neutral and Happy outputs with the same audio and view the comparison video directly in the notebook |\n",
        ]
        break

# 2. Update cell-13b (markdown for run_x86.sh)
for cell in nb["cells"]:
    if cell.get("id") == "cell-13b":
        cell["source"] = [
            "### `run_x86.sh` \u2014 with smile expression configuration\n",
            "\n",
            "This version supports expression environment variables:\n",
            "- `EXPR_ALPHA` \u2014 expression intensity (default `1.0` = natural smile, `0` = neutral)\n",
            "- `EXPR_PRESET` \u2014 smile preset (`subtle`, `natural`, `happy`, `broad`)\n",
        ]
        break

# 3. Update cell-14 (run_x86.sh code)
with open("run_x86.sh", "r", encoding="utf-8") as f:
    run_x86_lines = f.readlines()
for cell in nb["cells"]:
    if cell.get("id") == "cell-14":
        cell["source"] = ["%%writefile run_x86.sh\n"] + run_x86_lines
        break

# 4. Update cell-5b-md
for cell in nb["cells"]:
    if cell.get("id") == "cell-5b-md":
        cell["source"] = [
            "### 5b \u00b7 Emotion demo videos (Optional)\n",
            "\n",
            "The natural happy expression is now calibrated directly and cleanly from the distilled avatar renderer using `tools/apply_smile_expression.py` in section 7b.\n",
            "Uploading demo videos is optional and no longer required to run the smiling avatar.\n",
        ]
        break

# 5. Update cell-7b-md, cell-7b, cell-7b-inspect-md, cell-7b-inspect
for cell in nb["cells"]:
    if cell.get("id") == "cell-7b-md":
        cell["source"] = [
            "## 7b \u00b7 Calibrate and Preview the Happy Expression (~2 seconds)\n",
            "\n",
            "This cell runs `tools/apply_smile_expression.py` to:\n",
            "1. Load the avatar's base frame from `weights/renderer_narrow.pt`\n",
            "2. Apply the calibrated natural smile deformation (`preset=natural`, `alpha=1.0`)\n",
            "3. Generate a high-resolution side-by-side preview (`assets/expression_preview.jpg`)\n",
            "4. Save the expression configuration to `assets/happy_expression_delta.pt`\n",
        ]
    elif cell.get("id") == "cell-7b":
        cell["source"] = [
            "# Calibrate and preview the natural happy smile expression\n",
            "t0 = time.time()\n",
            'sh(\'PY="$(cat .python_path)" && \' \\\n',
            '   \'"$PY" tools/apply_smile_expression.py \' \\\n',
            '   \'  --renderer_ckpt weights/renderer_narrow.pt \' \\\n',
            '   \'  --alpha 1.0 \' \\\n',
            '   \'  --preset natural \' \\\n',
            '   \'  --preview_path assets/expression_preview.jpg \' \\\n',
            '   \'  --delta_output assets/happy_expression_delta.pt\')\n',
            'print("\\n--- smile expression calibration took %.2f s ---" % (time.time() - t0))\n',
            "\n",
            "delta_path = os.path.join(REPO_DIR, \"assets\", \"happy_expression_delta.pt\")\n",
            "preview_path = os.path.join(REPO_DIR, \"assets\", \"expression_preview.jpg\")\n",
            "if os.path.exists(delta_path) and os.path.exists(preview_path):\n",
            '    print(f"\\n✓ Happy expression configured successfully ({os.path.getsize(preview_path)/1024:.0f} KB preview)")\n',
            "else:\n",
            '    print(f"\\n✗ FAILED: Expression assets were not created properly.")\n',
        ]
    elif cell.get("id") == "cell-7b-inspect-md":
        cell["source"] = [
            "### Inspect configuration & visual preview\n",
            "\n",
            "Quick sanity check: verify expression settings and view the side-by-side comparison (Neutral vs Natural Happy Smile).\n",
        ]
    elif cell.get("id") == "cell-7b-inspect":
        cell["source"] = [
            "# Inspect the calibrated expression and display the side-by-side visual preview\n",
            "import os, torch\n",
            "from IPython.display import Image, display\n",
            "\n",
            'cfg = torch.load("assets/happy_expression_delta.pt", map_location="cpu")\n',
            'print("Expression Configuration:")\n',
            'print(f"  Preset: {cfg.get(\'preset\')}")\n',
            'print(f"  Alpha: {cfg.get(\'alpha\')}")\n',
            'print(f"  Effective scale: {cfg.get(\'effective_scale\')}")\n',
            "\n",
            'preview_path = os.path.join(REPO_DIR, "assets", "expression_preview.jpg")\n',
            "if os.path.exists(preview_path):\n",
            '    print("\\nSide-by-side preview: Left = Original Neutral | Right = Natural Happy Smile")\n',
            "    display(Image(preview_path, width=720))\n",
            "else:\n",
            '    print("Preview not found at", preview_path)\n',
        ]

# 6. Update cell-30 (wait_loaded grep check)
for cell in nb["cells"]:
    if cell.get("id") == "cell-30":
        src = "".join(cell["source"])
        if "natural smile|expression" not in src:
            src = src.replace("expression delta|expr_alpha|Uvicorn running", "natural smile|expression|expr_alpha|Uvicorn running")
            cell["source"] = [line + "\n" for line in src.split("\n")[:-1]]
        break

# 7. Update cell-comparison-video
for cell in nb["cells"]:
    if cell.get("id") == "cell-comparison-video":
        cell["source"] = [
            "# Create side-by-side comparison video\n",
            'happy_mp4 = os.path.join(REPO_DIR, "offline_test_happy", "output.mp4")\n',
            'neutral_mp4 = os.path.join(REPO_DIR, "offline_test_neutral", "output.mp4")\n',
            'comparison_mp4 = os.path.join(REPO_DIR, "offline_test_comparison.mp4")\n',
            "\n",
            "if os.path.exists(happy_mp4) and os.path.exists(neutral_mp4):\n",
            '    sh(\'ffmpeg -y \' \\\n',
            '       \'-i offline_test_neutral/output.mp4 \' \\\n',
            '       \'-i offline_test_happy/output.mp4 \' \\\n',
            '       \'-filter_complex "\' \\\n',
            '       \'[0:v]drawtext=text=NEUTRAL:fontsize=28:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5[v0];\' \\\n',
            '       \'[1:v]drawtext=text=HAPPY (SMILING):fontsize=28:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5[v1];\' \\\n',
            '       \'[v0][v1]hstack=inputs=2[out]" \' \\\n',
            '       \'-map "[out]" -c:v libx264 -crf 18 -pix_fmt yuv420p \' \\\n',
            '       \'offline_test_comparison.mp4\', check=False)\n',
            "    if os.path.exists(comparison_mp4):\n",
            '        print(f"\\n✓ Comparison video created: {comparison_mp4} ({os.path.getsize(comparison_mp4)/1e6:.1f} MB)")\n',
            "        from IPython.display import HTML, display\n",
            '        display(HTML(f"""\n',
            '        <video width="720" controls>\n',
            '            <source src="{comparison_mp4}" type="video/mp4">\n',
            '            Your browser does not support the video tag.\n',
            '        </video>\n',
            '        """))\n',
            "    else:\n",
            '        print("\\n✗ Comparison video creation failed (check ffmpeg output above).")\n',
            "else:\n",
            "    missing = []\n",
            "    if not os.path.exists(happy_mp4):\n",
            '        missing.append("happy")\n',
            "    if not os.path.exists(neutral_mp4):\n",
            '        missing.append("neutral")\n',
            '    print(f"Cannot create comparison: missing {\', \'.join(missing)} video(s).")\n',
            '    print("Run the offline capture cells for both happy and neutral first.")\n',
        ]
        break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")
