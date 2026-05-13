# Bioracer VTON Pipeline

AI pipeline that generates realistic model photos from flat-lay garment images.  
Takes a garment + a bank of pre-approved model poses → produces `N garments × M poses` output images via the [Fashn.ai](https://fashn.ai) API.

---

## Architecture

```
inputs/              poses/
(garment images)     (model photos — one per desired pose)
       │                    │
       ▼                    ▼
 GarmentPreprocessor   Image.open() directly
 - bg removal (rembg)
 - resize + pad
 - category detection (CLIP)
       │                    │
       └────────┬───────────┘
                ▼
        BatchOrchestrator  (thread pool)
                │
                ▼
        FashnAPIAdapter
        - sends garment + model photo + category
        - Fashn.ai handles pose & parsing internally
                │
                ▼
        outputs/{run_id}/{product_id}__{pose_id}.png
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export FASHN_API_KEY=your_key_here

# 3. Add garment images
cp your_garment.png inputs/

# 4. Add model/pose photos (one file per pose)
cp model_pose_01.jpg poses/
cp model_pose_02.jpg poses/

# 5. Run the batch
python scripts/run_batch.py
```

Results land in `outputs/<run_id>/`.

---

## CLI Options

```bash
python scripts/run_batch.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `config/pipeline.yaml` | Config file to use |
| `--run-id ID` | timestamp | Human-readable run label |
| `--no-open` | off | Skip auto-opening result images |

---

## Configuration Reference

`config/pipeline.yaml`:

```yaml
pipeline:
  workers: 4          # concurrent jobs
  input_dir: inputs/
  poses_dir: poses/
  output_dir: outputs/

garment:
  target_width: 768
  target_height: 1024

fashn_api:
  base_url: https://api.fashn.ai/v1
  timeout: 120        # seconds per request
```

Any value can be overridden with an environment variable:

```
PIPELINE__<SECTION>__<KEY>=value
```

Example: `PIPELINE__FASHN_API__TIMEOUT=60`

---

## Output Naming

```
outputs/
  run_20260513_143000/
    jersey_001__pose_01.png
    jersey_001__pose_02.png
    bib_shorts_002__pose_01.png
    bib_shorts_002__pose_02.png
```

---

## Tests

```bash
pytest
```

Tests use synthetic images and a mock adapter — no API key required.

---

## Docker

```bash
# API-only (no GPU needed)
docker compose --profile cpu up pipeline-cpu --build
```
