# Bioracer VTON Pipeline

AI pipeline that generates realistic model photos from flat-lay garment images.  
Takes a garment + a bank of pre-approved model poses → produces `N garments × M poses` output images.

---

## Architecture

```
inputs/garments/         poses/ (pre-approved bank)
       │                        │
       ▼                        ▼
 GarmentPreprocessor     PosePreprocessor (cached once)
 - bg removal (rembg)    - human parsing (SCHP)
 - resize + pad          - pose keypoints (DWPose)
 - category detection    - agnostic mask
       │                        │
       └──────────┬─────────────┘
                  ▼
          BatchOrchestrator  (thread pool, JSONL logging)
                  │
                  ▼
          VTONAdapter  ← swap via config, zero code change
          ┌─────────────────────┐
          │ fashn_api  (REST)   │  default — no GPU needed
          │ idm_vton   (local)  │  requires GPU + weights
          │ ootdiffusion (local)│  requires GPU + weights
          └─────────────────────┘
                  │
                  ▼
          outputs/{run_id}/{product_id}/{product_id}__{pose_id}.png
```

---

## Quick Start (Fashn.ai — no GPU required)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env and set FASHN_API_KEY

# 3. Add garment images
cp your_garment.png inputs/

# 4. Add pose images
cp model_pose_01.jpg poses/

# 5. Build the pose cache (run once, or after adding new poses)
python scripts/preprocess_poses.py

# 6. Run the batch
python scripts/run_batch.py --adapter fashn_api
```

Results land in `outputs/<run_id>/`.  
A `results.jsonl` file records every job with its status and output path.

---

## Switching VTON Adapters

Edit `config/pipeline.yaml`:

```yaml
vton:
  adapter: idm_vton   # or: ootdiffusion | fashn_api
```

Or pass `--adapter` at runtime:

```bash
python scripts/run_batch.py --adapter ootdiffusion
```

### IDM-VTON setup

```bash
git clone https://github.com/yisol/IDM-VTON
pip install -r IDM-VTON/requirements.txt
# Download weights from HuggingFace: yisol/IDM-VTON → IDM-VTON/weights/idm_vton/
export IDMVTON_REPO_PATH=IDM-VTON
```

### OOTDiffusion setup

```bash
git clone https://github.com/levihsu/OOTDiffusion
pip install -r OOTDiffusion/requirements.txt
# Download weights per OOTDiffusion README
export OOTD_REPO_PATH=OOTDiffusion
```

---

## Pose Bank Preprocessing

DWPose (keypoints) and SCHP (human parsing) must be installed for local preprocessing.  
Fashn.ai handles parsing server-side — preprocess_poses.py still runs but skips these steps gracefully.

```bash
git clone https://github.com/IDEA-Research/DWPose
git clone https://github.com/GoGoDuck912/Self-Correction-Human-Parsing SCHP
export DWPOSE_PATH=DWPose
export SCHP_PATH=SCHP

python scripts/preprocess_poses.py --force
```

---

## Docker

```bash
# GPU (IDM-VTON / OOTDiffusion)
docker compose up pipeline-gpu --build

# CPU / API-only (Fashn.ai)
docker compose --profile cpu up pipeline-cpu --build
```

---

## Tests

```bash
pytest
```

Tests use synthetic images and a mock adapter — no GPU or API key required.

---

## Output Naming

```
outputs/
  run_20260511_143000/
    results.jsonl
    jersey_001/
      jersey_001__pose_001.png
      jersey_001__pose_002.png
    bib_shorts_002/
      bib_shorts_002__pose_001.png
```

---

## Configuration Reference

All settings live in `config/pipeline.yaml`. Any value can be overridden at runtime with an environment variable:

```
PIPELINE__<SECTION>__<KEY>=value
```

Example: `PIPELINE__VTON__ADAPTER=fashn_api`
