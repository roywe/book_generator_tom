# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

TOM generates **3D-printable tactile Hebrew books**. Each page is produced as a single STL file containing three raised layers: an AI-generated line-art image, the Hebrew word, and its Braille transliteration — all on a 150×150mm base plate.

## Running the project

```bash
# Install dependencies
pip install -r requirements.txt

# Download the required Braille font (once, after cloning)
wget -O assets/NotoSansSymbols2-Regular.ttf \
  https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansSymbols2/NotoSansSymbols2-Regular.ttf

# Launch the Gradio web UI (main way to use the app)
python app/app.py

# Convert existing DXFs directly to STL
python src/dxf_3d.py --text text.dxf --braille braille.dxf --image image.dxf -o page1.stl
```

There are no automated tests. Manual verification is done by running the app and inspecting output files in `outputs/`.

## Architecture

The pipeline for one page flows through four stages:

```
Hebrew input
  → src/image_generator.py   (Stable Diffusion → 3 PNGs: image, text, braille)
  → src/image_funcs.py       (PNG → DXF contour extraction, one DXF per PNG)
  → src/dxf_3d.py            (3 DXFs → single STL via CadQuery)
```

**`src/flow_manager.py`** wraps this pipeline for multi-page books. It processes pages atomically — if any step throws, that page's state is not updated. Outputs land in `books/{name}_{timestamp}/`.

**`app/app.py`** is the primary entry point. It is **self-contained**: it does not import from `src/` and replicates the Hebrew→Braille mapping and Braille STL generation inline using `trimesh` (instead of `cadquery`). This makes it independent from the heavier pipeline but means it has its own copy of the Hebrew map.

**`src/dxf_3d.py`** is also a standalone CLI (`python src/dxf_3d.py --text ... --braille ... --image ...`) and the function `create_one_page_stl_from_dxf()` is what `flow_manager.py` calls programmatically.

**`src/language_funcs.py`**'s `add_nikud()` uses blocking `input()` calls — it only works in CLI/notebook contexts. `app/app.py` handles the same nikud disambiguation through Gradio dropdown widgets instead.

## Key constants (src/dxf_3d.py)

STL geometry is controlled by module-level constants:
- `BASE_WIDTH / BASE_HEIGHT` — 150×150mm base plate
- `BASE_THICKNESS` — 1.5mm
- `TEXT_SOLID_HEIGHT` — 1.5mm extrusion for Hebrew text
- `IMAGE_STROKE_WIDTH / IMAGE_STROKE_HEIGHT` — 1.0mm / 1.5mm for image ridges
- `DOME_HEIGHT_RATIO` — braille dome height = radius × 0.5

## Important path note

All imports from `src/` use the `src.` prefix (e.g. `from src import language_funcs as lf`). Run scripts from the repo root, not from inside `src/` or `app/`.
