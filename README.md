<div align="center">

<img src="docs/images/logo.svg" width="112" alt="FaceMesh logo">

# FaceMesh

**One PyTorch network that animates 3D faces and turns messy face scans into clean, animation-ready meshes.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-111827?style=flat&logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](facemesh/model.py)
[![OpenUSD](https://img.shields.io/badge/export-OpenUSD-111827?style=flat)](facemesh/export_usd.py)
[![Docker](https://img.shields.io/badge/Docker-CUDA_12.4-2496ED?style=flat&logo=docker&logoColor=white)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/license-MIT-3b82f6?style=flat)](LICENSE)

<a href="#quick-start">⚡️ Quick start</a> • <a href="#how-it-works">🔬 How it works</a> • <a href="#results">📊 Results</a> • <a href="#blender">🎬 Blender</a> • <a href="#gpu-training">☁️ GPU training</a> • <a href="#faq">❓ FAQ</a>

<br>

<img src="docs/images/hero.gif" width="640" alt="Side by side: the model's predicted expression for a person it never saw in training, next to the real motion, colored by how far each part of the face moves.">

</div>

## ✨ What it does

- **Animates a face.** Give it a person's neutral face, an expression and a time from 0 to 1, and it predicts every vertex of the face at that moment. Sixty time steps make an animation.
- **Cleans up scans.** Give it a raw point cloud of a face, in any order, with noise, a hole and stray points, and it returns a clean face on the standard template: fixed vertex count, fixed vertex order, ready to rig and animate.
- Does both with one shared PointNet encoder and two small decoders.
- Exports animated USD files that open in Blender.
- Always reports against a simple baseline, on people it has never seen and on frames it has never seen.
- Trains on an Apple Silicon GPU, an NVIDIA GPU or a CPU, with a Docker image and an AWS script for cloud GPUs.

<a name="quick-start"></a>
## ⚡️ Quick start

```bash
git clone https://github.com/simar-s2/facemesh && cd facemesh
python3 -m venv .venv
.venv/bin/pip install torch -r requirements.txt
```

No dataset yet? Make a fake one in the same format and train on it (about 8 minutes on an M1 Pro):

```bash
.venv/bin/python scripts/make_fake_dataset.py --out data/fake.npz
.venv/bin/python -m facemesh.train --data data/fake.npz --out_dir runs/fake --epochs 100
```

Training prints the baseline first (the number to beat), then the untrained model as epoch 0, then both jobs' error in millimeters every epoch. The best checkpoint goes to `runs/fake/best.pt` and the full history to `runs/fake/metrics.json`.

Export one animation per expression for a person the model never trained on:

```bash
.venv/bin/python -m facemesh.export_usd expressions --checkpoint runs/fake/best.pt \
    --data data/fake.npz --subject fake_subject_11 --out_dir exports/fake_subject_11
```

Clean up a scan (any `.ply` or `.obj`, in meters, roughly aligned with the template):

```bash
.venv/bin/python -m facemesh.export_usd fit --checkpoint runs/fake/best.pt \
    --scan my_scan.ply --out_dir exports/my_scan
```

<a name="how-it-works"></a>
## 🔬 How it works

```mermaid
flowchart LR
    neutral(["neutral face<br/>fixed vertex order"]) --> encoder["PointNet encoder<br/>shared weights"]
    scan(["noisy scan<br/>any order, any size"]) --> encoder
    encoder -->|"identity code"| expression["expression decoder<br/>+ expression + time"]
    encoder -->|"scan code"| fit["fit decoder"]
    expression --> animated(["face at time t"])
    fit --> cleaned(["clean template face"])
    animated --> usd["USD export"]
    cleaned --> usd
    usd --> blender(["Blender"])
```

- **PointNet encoder.** The same small network runs on every point, then a max over all points keeps the strongest response per feature. Max ignores order, so a shuffled scan gives exactly the same result. A test checks this.
- **Two decoders.** Both predict *offsets*: how far each vertex moves from the neutral face (animation) or from the average face (scan cleanup). Their last layer starts at zero, so an untrained model returns "no change", and training starts from a sensible answer.
- **Time features.** Time goes in as `[t, sin(πt), cos(πt), sin(2πt), cos(2πt)]`, because expressions rise and fall over a clip.
- **Loss.** L1 per vertex, plus an edge-length term that keeps the surface smooth instead of crumpled.
- **Baseline.** For each expression and tenth of the clip, the average motion over training people. The model has to beat it to be worth anything.
- **Simulated scans.** Every training sample gets a fresh fake scan of its clean mesh: a hole, 1.5 mm jitter, 2% stray points, shuffled. The clean mesh is the answer key, so scan pairs are free and unlimited.

<a name="results"></a>
## 📊 Results

Mean vertex error in millimeters, lower is better. **New people:** the last 2 people are held out of training. **Seen people:** every 10th frame is held out, everyone is in training. For scan cleanup, the reference is "output the average face".

**CoMA** (20,466 real face meshes, 12 people, 12 expressions, FLAME template):

| | expression: baseline | expression: model | scan cleanup: average face | scan cleanup: model |
|---|---|---|---|---|
| New people | COMA_RESULT | COMA_RESULT | COMA_RESULT | COMA_RESULT |
| Seen people | COMA_RESULT | COMA_RESULT | COMA_RESULT | COMA_RESULT |

**Fake data** (12 people, 12 expressions, 642 vertices, 100 epochs):

| | expression: baseline | expression: model | scan cleanup: average face | scan cleanup: model |
|---|---|---|---|---|
| New people | 0.122 | 0.153 | 3.87 | **0.78** |
| Seen people | 0.166 | **0.014** | 4.70 | **0.30** |

On fake data each person's expression strength is random and invisible in their neutral face, so for a new person nothing can beat the average. The baseline wins that one cell by design. Everywhere else the model wins by a wide margin.

<p align="center">
  <img src="docs/images/scan_cleanup.png" width="640" alt="A noisy, holey point cloud of a face on the left, and the clean template mesh the model produces from it on the right.">
</p>

**What helped**, measured on fake data:

| Change | Effect |
|---|---|
| No BatchNorm in the shared encoder | Held-out error 0.45 to 0.29 mm (earlier version, 3 epochs). It sees clean faces and noisy scans, so its saved averages fit neither. |
| Random head stretch, up to 5% per axis | Scan cleanup on new people 1.89 to 0.78 mm. Ten people become endless slightly different people. |
| 100 epochs instead of 30 | Helps with stretching. Without it, the model overfits after epoch 6. |
| Fair checkpoint choice | Each job scored as a fraction of its simple answer, so the larger scan error no longer picks the epoch alone. |
| Learning a correction on top of the baseline | Did not help: it learned nothing beyond the baseline. Off by default (`--baseline_correction`). |
| Hiding the identity code at random | Worse on both splits. Off by default (`--identity_dropout`). |

<a name="blender"></a>
## 🎬 Open it in Blender

File → Import → Universal Scene Description, pick an expression `.usda`, then press play. Each file is one mesh, Y-up, in meters, at 30 fps, with one time sample per frame (60 frames).

<a name="real-data"></a>
## 🗂️ Real data (CoMA)

Request access at [coma.is.tue.mpg.de](https://coma.is.tue.mpg.de), unzip so you have `COMA_data/<person>/<expression>/*.ply`, then:

```bash
.venv/bin/python scripts/preprocess_coma.py --coma_root path/to/COMA_data --out data/coma.npz
```

It checks that every mesh shares one template (5,023 vertices for FLAME) and stops with the file name if one doesn't. Add `--frame_step 3` to keep every third frame. Neighbouring frames are nearly identical, so training gets about 3x faster for little loss.

<a name="gpu-training"></a>
## ☁️ GPU training

**On a Mac.** Training uses the Apple GPU automatically. At CoMA size, an epoch takes about 50 s on an M1 Pro:

```bash
caffeinate -i .venv/bin/python -m facemesh.train --data data/coma.npz --out_dir runs/coma_new_people --epochs 30 --num_workers 3
caffeinate -i .venv/bin/python -m facemesh.train --data data/coma.npz --out_dir runs/coma_seen_people --epochs 30 --num_workers 3 --split interpolation
```

`caffeinate -i` keeps the Mac awake. The results above were trained this way.

**On AWS.** The Docker image (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`) and `scripts/train_on_aws.sh` train both splits on one GPU machine:

1. Upload the data: `aws s3 cp data/coma.npz s3://YOUR_BUCKET/data/coma.npz`
2. Launch a **g5.xlarge** (one NVIDIA A10G) with the **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu)**, 100 GB of disk, and an IAM role that can read and write the bucket.
3. SSH in, then:
   ```bash
   git clone https://github.com/simar-s2/facemesh && cd facemesh
   BUCKET=YOUR_BUCKET ./scripts/train_on_aws.sh
   ```
   It builds the image, trains both splits with bfloat16 mixed precision, and syncs `runs/` back to S3.
4. **Terminate the instance when it finishes.** It costs about $1 an hour while it runs.

<a name="faq"></a>
## ❓ FAQ

**Why does the model lose to the baseline on new people with fake data?** Each fake person's expression strength is random and has nothing to do with their face shape, so the best guess for a new person is the average, which is exactly what the baseline predicts. On people it has seen, the model learns their strength and is 12x better than the baseline.

**Why no BatchNorm?** The shared encoder sees clean neutral faces and noisy scans. BatchNorm's running averages became a blend of both and matched neither at test time. Removing it cut held-out error by a third.

**Can it clean up any scan?** It needs the scan in meters and roughly lined up with the template (same position and facing). Real scanner output usually needs a rigid alignment step first.

**My meshes came out scrambled.** Load them with `trimesh.load(path, process=False)`. By default trimesh merges and reorders vertices, which silently breaks the vertex-to-vertex match the whole model relies on.

**`torch.load` refuses the checkpoint.** PyTorch 2.6+ only loads plain tensors by default, and the checkpoint also stores NumPy arrays. FaceMesh loads its own checkpoints with `weights_only=False`.

**Training stopped with "Shared memory manager connection has timed out".** The Mac went to sleep under the data loader. Run with `caffeinate -i`, or use `--num_workers 0`.

## 🛠️ Development

```bash
.venv/bin/python -m pytest tests     # 5 tests, about 1 second
```

| Test | What it protects |
|---|---|
| Encoder ignores point order | Same scan, any order, same answer |
| Untrained model returns the neutral face | Zero-initialized decoders and offset prediction |
| Untrained model with the offset table returns the baseline | The baseline-correction option and its time bins |
| Noisy scan has the requested size | The scan simulator always returns N points |
| USD file has every frame | Time samples and point values survive a write and read |

## 📄 License

[MIT](LICENSE). The CoMA dataset has its own license and is not included.
