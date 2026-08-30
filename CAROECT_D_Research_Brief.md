# CAROECT-D — Research Presentation Brief
**CARs On Event Camera sTreams (RGB-Derived)**

---

## One-sentence pitch

> We create a large-scale labeled traffic event-camera dataset from conventional RGB video through physically grounded event simulation, without requiring a real event camera for primary data collection or manual annotation at dataset scale.

---

## One-page overview

```
PROBLEM                     SOLUTION
─────────────────           ──────────────────────────────────────
Event cameras suit           SDR N-RAW capture
  traffic sensing        →      ↓ DaVinci linear Rec.709 export
But:                             ↓ v2e / DVS-Voltmeter simulation
  - Hardware is costly          ↓ SAM3 automatic annotation
  - Annotation is difficult     ↓ Causal geometric label transfer
  - Datasets are scarce         ↓
                           Labeled synthetic event dataset
                               ↓ Model training (RVT, YOLOv8-event)
                               ↓ Sim2real validation (LUCID EVS)
                           Deployed traffic perception model
```

**Paper claim to test:** if RGB preprocessing preserves the relevant radiometric, temporal, and geometric properties, synthetic events may be close enough to real events for competitively transferring trained models. This remains an empirical hypothesis, not a presumed result.

---

---

# PART 1 — MOTIVATION AND PROBLEM

## 1.1 What are event cameras, and why do they matter?

An event camera **does not capture conventional frames**. Instead, each pixel operates independently:
- When the change in log intensity crosses threshold C, the pixel emits an **event** `(x, y, t, polarity)`.
- An ON event represents an increase; an OFF event represents a decrease.
- Temporal resolution: **microseconds** rather than conventional frame-camera milliseconds.
- Event sensors can provide approximately **120 dB** dynamic range, depending on sensor and operating conditions.
- Asynchronous events avoid conventional frame-exposure motion blur, subject to photoreceptor bandwidth.

For **traffic monitoring**, these properties are relevant to fast vehicles, difficult illumination, low latency, real-time detection, and tracking. Actual performance must still be measured.

## 1.2 The problem: insufficient labeled data

Deep learning requires labeled data. For event cameras, this is a severe bottleneck:

| Problem | Detail |
|--------|---------|
| Expensive hardware | High-quality event-camera systems can cost thousands of dollars per unit. |
| Manual annotation does not scale | Event streams are sparse asynchronous points, making direct bounding-box annotation impractical at scale. |
| Complex synchronization | Paired evaluation needs hardware or signal-based synchronization between RGB and event cameras. |
| Few public datasets | Existing traffic datasets cover limited footage, locations, and conditions. |
| Low diversity | A small number of locations cannot represent broad deployment conditions. |

**Gap:** there is no established large-scale, diverse, automatically labeled traffic event-camera dataset with complete radiometric, temporal, geometric, simulator, and split provenance.

## 1.3 Why has this remained difficult?

Prior approaches generally follow these paths:
- **Real collection (eTram, TUMTraf):** deploy event cameras and create annotations; this is valuable but expensive to scale.
- **Simplified simulation:** process ordinary RGB video without controlling transfer, exposure, geometry, and timing; this can enlarge the sim-to-real gap.

**CAROECT-D** systematically combines: (1) SDR N-RAW capture with DaVinci linear Rec.709 export, (2) measurement-backed calibration manifests, (3) physically grounded simulation, (4) bidirectional SAM3 automatic annotation, and (5) causal label transfer in a scalable end-to-end system.

---

---

# PART 2 — HARDWARE SYSTEM

## 2.1 Camera A — RGB Source (Nikon Z6 III)

**Purpose:** collect source footage from which synthetic event streams and RGB-domain labels are generated.

| Setting | Value | Rationale |
|---------|-------|-------|
| Format | 12-bit SDR N-RAW (.NEV), not N-Log | High-bit-depth source decoded and linearized in DaVinci Resolve. |
| Frame rate | 119.88 fps | Dense source observations for event simulation; no label interpolation. |
| ISO | 100–400 | Favor low noise and usable source dynamic range when lighting permits. |
| Shutter | 1/250–1/500 s | Preserve sharp edges and reduce frame-domain motion blur. |
| Aperture | f/5.6–f/8 | Increase depth of field and support stable manual focus. |
| WB/exposure | Manual and fixed | Automatic temporal changes can create false event-like transients. |
| Focus | Manual and locked | Autofocus can change geometry and invalidate calibration assumptions. |
| Storage | CFexpress Type B | Sustain high-frame-rate N-RAW recording. |

**Candidate lenses:** 20 mm f/1.8, 24 mm, and 28 mm, selected to cover the roadway. Each camera/lens/focus configuration needs matching calibration provenance.

## 2.2 Camera B — Event Validation (LUCID Triton2 EVS)

**Purpose:** capture real events for paired and cross-domain validation.

- Sensor: Sony IMX636
- Resolution: 1280×720, also used as the native event-dataset grid.
- It is not the primary large-scale collection camera; it provides real-event validation.
- For paired evaluation it observes the same scene as Camera A with an explicit synchronization estimate.

## 2.3 Hyperfocal distance and stable focus

Focusing near hyperfocal distance H makes the scene acceptably sharp from approximately H/2 to infinity under the usual approximation. For example, H = 10 m suggests a nominal range from about 5 m to infinity. Roadside targets span many distances, so this depth of field is useful.

Focus is then locked to maintain calibration consistency and avoid lens-geometry changes between recordings.

---

---

# PART 3 — DATASET GENERATION PIPELINE

## 3.1 Overview Pipeline

```
.NEV files
  │
  ▼ [DaVinci Resolve]
  ├─ Decode 12-bit SDR N-RAW in DaVinci Resolve (not N-Log)
  ├─ Apply a fixed documented SDR-to-linear Rec.709 transform
  └─ Export already-linear RGB uint16 TIFF at native 119.88 fps
  │
  ▼ [Python / OpenCV]
  ├─ Load + validate uint16
  ├─ Dark frame subtraction        ← sensor thermal noise
  ├─ Flat field correction         ← vignetting + pixel non-uniformity
  ├─ White balance (gray card)     ← illuminant correction
  ├─ Assert input_transfer=linear        ← prevent double linearization
  ├─ RGB → luma Y (Rec.709)       ← event cameras are monochrome
  ├─ Undistort (K, D)             ← geometric accuracy for labels
  ├─ Gentle denoise (σ≤1.0)       ← optional
  ├─ Preserve native 1280×720      ← explicit letterbox only if requested
  └─ Optional stabilization        ← disabled by default and shared by branches
  │
  ▼ [v2e / DVS-Voltmeter]
  ├─ Temporal upsampling (SuperSloMo)
  ├─ log(Y) → ΔL → events per pixel
  ├─ ON/OFF threshold ± C
  └─ Noise models (shot, hot pixels, leak, threshold variation)
  │
  ▼ Event representations
  ├─ Fixed-count images (8.34/16.7/33.3/50.0 ms causal windows)
  ├─ Voxel grids (H × W × T bins)
  └─ Time surfaces (recency map)
```

## 3.2 Why each preprocessing step exists

### Dark residual — electronic baseline correction

A sensor may have non-zero output in darkness because of thermal and fixed-pattern components. If that residual is present, `log(signal + residual)` can alter threshold crossings in dark regions.

*Acquisition:* record multiple lens-cap frames under matching settings, estimate and validate the residual artifact, and enable correction only when the calibration manifest marks it valid.

### Flat field — spatial response correction

Lens vignetting and pixel-response non-uniformity can create a spatial gradient. Object motion across that gradient can then generate position-dependent synthetic events.

*Acquisition:* record a homogeneous non-zero field, derive a bounded gain map, validate it, and leave correction disabled until the artifact is valid.

### Transfer ownership — DaVinci linearization, never twice

Capture uses SDR N-RAW, not N-Log. DaVinci Resolve removes the declared SDR encoding and exports linear Rec.709 before Python. Feeding encoded SDR directly would mis-scale log contrast; applying inverse gamma again in Python would also be wrong:
- Encoded SDR input would distort shadow-region threshold crossings.
- Double linearization would distort both dark and bright values and invalidate simulator calibration.

*Implementation rule:* `camera.input_transfer: linear`; the Python transform is an identity because DaVinci owns linearization.

### RGB → Luminance Y (BT.709)

Event simulation is monochrome. The already-linear Rec.709 channels are reduced with `Y = 0.2126R + 0.7152G + 0.0722B`; the arithmetic channel mean is not used.

### 16-bit TIFF rather than an 8-bit delivery image

Event simulation depends on small log-intensity differences. A high-bit-depth linear working export preserves more numerical precision than an 8-bit delivery image, especially in dark regions. TIFF bit depth alone does not imply that SDR capture matches the event sensor's dynamic range.

## 3.3 v2e vs DVS-Voltmeter

| | v2e | DVS-Voltmeter |
|-|-----|---------------|
| Model | Log-intensity threshold | Stochastic voltage dynamics (Brownian motion) |
| Accuracy | Good, fast | More physically accurate, slower |
| Noise model | Shot noise, hot pixels, threshold variation | Voltage noise, leak events, temporal dynamics |
| Use case | Baseline, high throughput | Higher-fidelity simulation, ablation |
| Output | (x,y,t,polarity) .h5 | (x,y,t,polarity) .h5 |

The paper will compare downstream performance for isolated v2e and DVS-Voltmeter conditions.

## 3.4 Automatic Annotation Pipeline

**In the RGB domain:**
1. **SAM3** receives a text prompt for the target traffic-participant class.
2. Independent forward and backward sessions preserve both directional track artifacts.
3. Same-class trajectories are merged deterministically using overlap, IoU, continuity, confidence, and a narrow receding-object tie-break.

**Transfer to the event domain:**
- Use the shared geometric grid established before the RGB/event branch split.
- At exact observed time `t_k`, copy frame-k geometry to events in `[t_k-window_us, t_k)`.
- Do not use future labels or interpolation in the normal path; real pairs require explicit synchronization provenance.

**Result:** a scalable automatically labeled dataset with auditable timing, geometry, and simulator provenance.

---

---

# PART 4 — MODEL TRAINING

## 4.1 Model architecture

### Baseline: Frame-based Detection
- **Input:** fixed-count ON/OFF event images from causal accumulation windows.
- **Model:** an Ultralytics YOLO event baseline; HSV color augmentation is disabled.
- **Rationale:** familiar tooling and fast iteration provide a reproducible baseline.

### State-of-the-art: RVT (Recurrent Vision Transformer)
- **Input:** Voxel grids (H × W × T)
- **Architecture:** Recurrent attention mechanism captures temporal event dynamics
- **Performance:** a candidate specialized comparison on standard event-detection benchmarks.
- **Rationale:** traffic events have strong temporal structure that a recurrent architecture may exploit better than independent frames. This remains a proposed comparison.

### Tracking Module
- ByteTrack or BoT-SORT can be applied to detector output.
- Event-specific time surfaces may support association when detection confidence is low.

## 4.2 Event representations and architectural implications

| Representation | Construction | Input shape | Typical consumer |
|---------------|---------|------------|-------------|
| Fixed-count ON/OFF image | Accumulate causal events and clip with one shared train-derived scale | [B, 3, H, W] | 2D detector |
| Voxel grid | Polarity-weighted temporal bins | [B, T, H, W] | RVT / temporal convolution |
| Time surface | Most recent event time by polarity | [B, 2, H, W] | Auxiliary input |

**Trade-off for detector window T:**
- 8.34 ms: fewer events and high temporal localization.
- 50.0 ms: denser object shape but more temporal integration. Standard intermediate windows are 16.7 and 33.3 ms.

## 4.3 Dataset split and structure

```
CAROECT-D/
  RAW_NEV/
    site01/   ← highway overpass, 4-lane, daytime
    site02/   ← urban intersection, mixed lighting
    site03/   ← suburban road, nighttime
    ...
  Events_v2e/
    site01/   ← synthetic events from site01
  Events_DVSVolt/
    site01/
  Labels/
    site01/   ← COCO-format JSON annotations
```

**Split by site, never by frame.** Frames from one site are temporally and visually correlated; frame-level random splitting would leak scene identity and undermine validation.

## 4.4 Augmentation Strategies

**Standard:** Random flip, crop, mosaic, scale.

**Event-specific:**
- **Event drop:** Random remove X% events → simulate sensor packet loss, sensor aging
- **Temporal jitter:** Scale timestamps → simulate different motion speeds
- **Polarity noise:** Randomly flip polarities to simulate per-pixel threshold variation.
- **Spatial noise:** Add random spurious events → simulate hot pixel behavior
- **Rate scaling:** Change event density → simulate different scene dynamics

*Why event-specific augmentation matters:* it can expose the model to controlled sensor-like variation. These operations are experimental and must be recorded rather than silently baked into the canonical dataset.

## 4.5 Training Setup

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW + weight decay 1e-4 |
| LR schedule | Cosine annealing with warmup |
| Loss (YOLO-style) | Box regression + classification + objectness |
| Loss (DETR-style) | Hungarian matching |
| Batch size | 8–32, depending on GPU memory and representation |
| Hardware | Declared per run; voxel grids can require substantial memory |

**Label provenance:** every dataset root records the causal timestamp range, event indices, geometry, simulator condition, split, and shared representation scale. COCO export, when used, must preserve this metadata.

---

---

# PART 5 — EVALUATION STRATEGY

## 5.1 Internal Validation: Synthetic vs Real Event Quality

**Question:** how closely do pipeline-generated events match real LUCID EVS events under controlled observations?

**Protocol:**
1. Place the LUCID EVS and Nikon systems on the same observable scene.
2. Estimate RGB/event offset from a visible synchronization signal and store method, residual, and confidence.
3. Generate synthetic events and compare exact metric vectors under matched support.

**Metrics:**
- Event rate (events/pixel/second): synthetic vs real
- Polarity ratio (ON/OFF balance)
- Spatial density maps
- Noise floor estimation
- Estimate contrast threshold C from controlled real observations and compare it with declared simulator parameters.

## 5.2 Detection Performance

**Primary metrics:**
- **mAP@50:** Standard threshold, overall detection accuracy
- **mAP@50:95:** stricter COCO-style localization assessment.
- **AP per class:** Cars, trucks, motorcycles, pedestrians, cyclists
- **FPS:** Real-time capability assessment

**Baselines:**
1. CAROECT-D synthetic trained
2. eTram real-event trained (same architecture)
3. TUMTraf Event trained
4. RGB-only pretrained (no event data) → transfer learning baseline

## 5.3 Tracking Performance

**Primary metrics:**
- **HOTA:** Higher Order Tracking Accuracy — primary, balances detection + association quality
- **MOTA:** Multi-Object Tracking Accuracy — classic metric
- **IDF1:** Identity F1 — how well identities maintained over time
- **MOTP:** Trajectory precision

## 5.4 Sim-to-real validation — the paper's key experiment

**Question:** does a model trained on CAROECT-D synthetic events generalize to real event data?

```
Train: CAROECT-D synthetic events
  ↓ Zero-shot test
  ├── Real LUCID EVS captures (same scenes)
  ├── eTram dataset (different location, different camera)
  └── TUMTraf Event dataset (different country, different setup)

Success: performance gap < 5–10% mAP → synthetic training viable
```

**If the gap is small:** the result supports the core claim under the declared protocol.
**If the gap is large:** controlled ablations should isolate radiometry, timing, geometry, simulator, representation, and scene-domain causes.

## 5.5 Ablation studies — measure each component's effect

### Ablation 1: Preprocessing Quality Impact

| Variant | Dark | Flat | WB | Linearize | Expected |
|---------|------|------|-----|-----------|---------|
| Full pipeline | ✓ | ✓ | ✓ | ✓ | Best mAP |
| No linearization | ✓ | ✓ | ✓ | ✗ | Drop: wrong event rates |
| No dark+flat | ✗ | ✗ | ✓ | ✓ | Drop: spatial noise artifacts |
| Minimal (raw) | ✗ | ✗ | ✗ | ✗ | Worst: all artifacts present |

*Objective:* quantify whether each measurement-backed preprocessing choice improves event fidelity and downstream performance. Unavailable calibration data is not fabricated to fill an ablation cell.

### Ablation 2: Event Representation

| Representation | mAP | Latency |
|---------------|-----|---------|
| Event frames (10ms) | ? | Fast |
| Event frames (50ms) | ? | Fast |
| Voxel grids | ? | Slower |
| Time surfaces | ? | Fast |

### Ablation 3: Simulator Comparison

| Simulator | Event quality score | Model mAP |
|-----------|-------------------|-----------|
| v2e | | |
| DVS-Voltmeter | | |

*If DVS-Voltmeter improves mAP:* the added complexity may be justified. *If results are similar:* the simpler v2e baseline is preferable.

---

---

# PART 6 — RELATED WORK AND POSITIONING

## 6.1 Core papers and datasets

### v2e — simulation foundation

*Yang et al., CVPR 2021*

Event generation from video frames using a log-intensity threshold model:
- Optional temporal upsampling for higher synthetic sampling density.
- `L = log(Y)` with an event when `|ΔL| > C`.
- Noise: shot noise, threshold variation, hot pixels, leak events
- Output: (x,y,t,polarity) stream

**Relation to CAROECT-D:** v2e is the named primary baseline. DaVinci supplies already-linear Rec.709 input so Python can compute matching luminance without another transfer conversion.

### DVS-Voltmeter — Advanced simulation

*Lin et al.*

This model describes photodiode-circuit voltage dynamics as Brownian motion with drift rather than only a deterministic threshold, capturing temporal clustering, refractory behavior, and leakage.

**Relation:** an alternative simulator compared with v2e under disjoint condition roots.

### eTram — Real event traffic dataset

Urban tram tracking with real event-camera observations and ground-truth tracking annotations.

**Relation:** a possible real-event benchmark for evaluating CAROECT-D generalization, subject to compatible classes and protocols.

### TUMTraf Event — Multimodal traffic dataset

RGB + event camera, roadside, TU Munich. Multiple scene types, multiple object classes.

**Relation:** cross-domain validation under a different country, scene, and acquisition setup.

## 6.2 CAROECT-D gap in the literature

| | eTram | TUMTraf | Prophesee Gen1/4 | **CAROECT-D** |
|-|-------|---------|------------------|----------------|
| Scale | Small | Medium | Medium | **Large** |
| Annotation | Manual | Manual | Annotated | **Automatic** |
| Hardware needed | Event cam | Event cam + RGB | Event cam | **RGB only** |
| Diversity | Low | Medium | Medium | **High** |
| Synthetic? | No | No | No | **Yes** |

---

---

# PART 7 — RESEARCH CONTRIBUTIONS

## Contribution 1: Physics-Accurate Preprocessing Pipeline

**Proposed novelty:** a systematic SDR N-RAW → DaVinci-linear Rec.709 → event-simulation-ready luminance workflow with calibration manifests, optional corrections, and shared geometry.

**Evidence:** controlled ablations relating each valid preprocessing choice to event metrics and downstream performance. This is a claim to test rather than assume.

## Contribution 2: CAROECT-D Dataset

Large-scale, diverse, automatically-labeled roadside traffic event dataset:
- Multiple sites, multiple lighting conditions, multiple weather conditions
- Full annotations: bounding boxes, instance masks, tracking IDs
- Both v2e and DVS-Voltmeter variants, with default and calibrated conditions isolated.

## Contribution 3: Zero-Annotation Pipeline

End-to-end automatic annotation: DINO + SAM (RGB detection) → ByteTrack (RGB tracking) → geometric transfer → event domain labels. **Zero human annotation required.** Demonstrates scalability: any RGB traffic footage → labeled event dataset.

## Contribution 4: Sim2Real Validation

Quantitative tests determine whether CAROECT-D-trained models generalize to real event-camera data. Positive results would support synthetic training as an alternative for traffic perception; negative or mixed results remain informative.

---

---

# PART 8 — Q&A PREPARATION

**Q: Why not use only real event cameras?**

A: Cost (hardware $5k–30k), annotation impossibility at scale, limited location flexibility. CAROECT-D scales to any RGB footage — historical traffic footage, multiple cameras simultaneously, diverse worldwide locations — at marginal cost. Real event cameras remain as validation tools, not as primary data collection.

---

**Q: How large is the sim-to-real gap, and how do we know the synthetic data is good enough?**

A: The gap is measured through (1) controlled comparison of exact synthetic and real metric vectors and (2) downstream zero-shot performance on real event benchmarks. Synchronization, metric definitions, weights, simulator condition, and confidence are stored; ablations attribute changes to pipeline components.

---

**Q: Is SAM3 accurate enough to produce training labels?**

A: Accuracy must be measured on a manually reviewed sample from the actual traffic domain. The pipeline reconciles independent directional tracks with temporal overlap, IoU, continuity, confidence, and a narrowly defined receding-object tie-break; external benchmark claims alone are insufficient.

---

**Q: Why use SDR N-RAW rather than ordinary H.264 delivery video?**

A: Delivery video may contain a transfer function, temporal denoising, sharpening, chroma subsampling, lossy compression, and automatic white-balance/exposure changes. CAROECT-D records SDR N-RAW—not N-Log—and performs a fixed SDR-to-linear Rec.709 transform in DaVinci before Python. The pipeline does not claim that this SDR workflow is untouched photosite data.

---

**Q: Is v2e sufficient, or is DVS-Voltmeter necessary?**

A: v2e is the high-throughput named baseline. DVS-Voltmeter provides a more detailed alternative model. Measurements decide whether its added complexity improves real-event transfer; similar results would favor v2e.

---

**Q: Why 16-bit TIFF rather than PNG or JPEG?**

A: TIFF is used as a high-bit-depth lossless container for the already-linear working signal. JPEG is lossy. Common 8-bit PNG has only 256 code values; true 16-bit linear PNG could preserve similar values but would still require transfer and reader validation.

---

**Q: How does the pipeline scale across sites, and how long does one hour take?**

A: Decode, preprocessing, annotation, and simulation can be separated and parallelized where state constraints permit. Runtime depends on source cadence, simulator, GPU, prompts, and storage. Earlier speed figures are planning hypotheses; the paper should report measured throughput on declared hardware.

---

---

# GLOSSARY — Quick Reference

| Term | Definition |
|------|-----------|
| **Event camera / DVS** | A sensor that emits `(x,y,t,polarity)` events rather than conventional frames. |
| **ON / OFF event** | ON represents a sufficient log-intensity increase; OFF represents a decrease. |
| **Threshold C** | Minimum log-intensity change needed to trigger an event under a threshold model. |
| **Log-intensity** | `L = log(I)` — event cameras respond to this, not linear I |
| **Linearization** | Undo gamma: `linear = compressed^2.2` |
| **BT.709 luminance** | `Y = 0.2126R + 0.7152G + 0.0722B` |
| **Dark residual** | Sensor output measured under darkness for optional baseline correction. |
| **Flat field** | Homogeneous non-zero exposure used to estimate spatial response. |
| **NRAW / .NEV** | Nikon proprietary RAW video format |
| **K, D** | Camera intrinsic matrix + distortion coefficients |
| **v2e** | Video-to-Events: log-intensity threshold event simulator |
| **DVS-Voltmeter** | Physically accurate voltage dynamics event simulator |
| **Event frames** | 2D histogram of events accumulated in time window |
| **Voxel grids** | 3D tensor H×W×T time bins |
| **Time surfaces** | 2D map: timestamp of most recent event per pixel |
| **RVT** | Recurrent Vision Transformer — SOTA event-based detector |
| **mAP** | Mean Average Precision — detection accuracy metric |
| **HOTA** | Higher Order Tracking Accuracy — primary tracking metric |
| **Sim-to-real** | Generalization from synthetic training to real deployment data. |
| **SAM3** | Promptable segmentation/tracking model used for RGB-domain labels. |
| **SAM** | Segment Anything Model — pixel-level segmentation |
| **ByteTrack** | SOTA multi-object tracker |
| **Hyperfocal distance** | Focus distance intended to maximize the acceptably sharp depth range. |
| **CFexpress Type B** | High-throughput storage-card format used by the source camera. |
| **IMX636** | Sony sensor used in the LUCID Triton2 EVS reference camera. |

---

---

# TIMELINE — Presentation Flow (20–25 minutes)

| Part | Content | Time |
|------|---------|-----------|
| 1 | Pitch + Motivation (event cameras + problem) | 3 min |
| 2 | Core idea + system overview | 2 min |
| 3 | Camera hardware | 1 min |
| 4 | Dataset pipeline — DaVinci + Python steps | 5 min |
| 5 | Event simulation (v2e, DVS-Voltmeter) + representations | 3 min |
| 6 | Auto-annotation pipeline (DINO+SAM+tracking) | 2 min |
| 7 | Training methodology + architectures | 4 min |
| 8 | Evaluation + ablation studies | 3 min |
| 9 | Contributions + positioning vs related work | 2 min |
| 10 | Q&A buffer | 5 min |

---

*This brief preserves the full CAROECT-D scope from motivation through hardware, preprocessing, simulation, annotation, training, evaluation, and contributions.*
*Version 1.1 — aligned English draft prepared for internal research presentation.*
