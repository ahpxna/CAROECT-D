# CAROECT-D — Research Presentation Brief
**CARs On Event Camera sTreams (RGB-Derived)**

---

## Pitch một câu

> Chúng tôi tạo ra large-scale labeled traffic event-camera dataset từ RGB video thông thường, bằng cách simulate event streams với độ chính xác vật lý học — không cần event camera thật để thu thập data, không cần con người để label.

---

## Bức tranh toàn cảnh — 1 trang

```
VẤN ĐỀ                     GIẢI PHÁP
─────────────────           ──────────────────────────────────────
Event cameras tốt           RGB RAW capture (rẻ, dễ)
  cho traffic sensing   →      ↓ Physics-accurate preprocessing
Nhưng:                         ↓ v2e / DVS-Voltmeter simulation
  - Hardware đắt               ↓ DINO + SAM auto-annotation
  - Annotation khó             ↓ Geometric label transfer
  - Dataset ít và nhỏ          ↓
                           Labeled synthetic event dataset
                               ↓ Model training (RVT, YOLOv8-event)
                               ↓ Sim2real validation (LUCID EVS)
                           Deployed traffic perception model
```

**Key claim của paper:** Nếu RGB preprocessing đúng về mặt physics, synthetic events gần với real events đủ để train models có performance competitive với models trained on real event data.

---

---

# PHẦN 1 — MOTIVATION & PROBLEM

## 1.1 Event Cameras Là Gì và Tại Sao Quan Trọng

Event camera **không chụp frame**. Thay vào đó, mỗi pixel hoạt động độc lập:
- Khi log-intensity tại pixel đó thay đổi vượt threshold C → pixel fires một **event** `(x, y, t, polarity)`
- ON event: sáng lên. OFF event: tối đi
- Temporal resolution: **microseconds** (camera thường: milliseconds)
- Dynamic range: **~120 dB** (camera thường: ~60 dB)
- Zero motion blur do không có exposure time

Với **traffic monitoring**: xe tốc độ cao không bị blur, hoạt động tốt cả ngày lẫn đêm, latency cực thấp — ideal cho real-time vehicle detection và tracking.

## 1.2 Vấn Đề: Không Có Data Để Train

Deep learning cần labeled data. Với event cameras, đây là bottleneck cực kỳ nghiêm trọng:

| Vấn đề | Chi tiết |
|--------|---------|
| Hardware đắt | Event camera tốt: $5,000–$30,000+ mỗi cái |
| Annotation không thể làm tay | Event stream trông như mớ điểm rời rạc, không ai nhìn vào đó mà vẽ bbox được |
| Synchronization phức tạp | Cần phần cứng đặc biệt để sync event camera với reference RGB |
| Dataset công khai cực ít | eTram: vài giờ footage. TUMTraf: vài giờ. Không dataset nào đủ lớn |
| Diversity thấp | Mỗi dataset chỉ cover 1–2 locations, limited conditions |

**Gap:** Không có large-scale, diverse, automatically-labeled traffic event-camera dataset nào.

## 1.3 Tại Sao Chưa Ai Giải Quyết Được

Các approaches trước đó:
- **Thu thập thật (eTram, TUMTraf):** Deploy event camera thật → annotation thủ công → không scale
- **Simple simulation:** Apply basic Gaussian filter lên RGB video → không physically accurate → sim2real gap lớn → model không transfer

**CAROECT-D** là approach đầu tiên systematic kết hợp: (1) RAW RGB capture để preserve physics, (2) full sensor calibration pipeline, (3) physically-grounded simulation, (4) foundation model auto-annotation, vào một end-to-end scalable system.

---

---

# PHẦN 2 — HARDWARE SYSTEM

## 2.1 Camera A — RGB Source (Nikon Z6 III)

**Mục đích:** Thu thập source footage để generate everything else

| Setting | Value | Lý do |
|---------|-------|-------|
| Format | NRAW (.NEV) | RAW sensor data, trước mọi camera processing |
| Framerate | 119.88 fps | Temporal density cao cho event interpolation |
| ISO | 100–400 | Low noise, high dynamic range |
| Shutter | 1/250–1/500 | Sharp edges, giảm motion blur |
| Aperture | f/5.6–f/8 | Large depth of field, stable hyperfocal focus |
| WB/Exposure | Manual, fixed | Auto changes → fake events |
| Focus | Manual, locked | Autofocus thay đổi geometry → break calibration |
| Storage | CFexpress Type B | Đủ nhanh cho NRAW 120fps |

**Lenses dùng:** 20mm f/1.8, 24mm, 28mm — wide để capture cả làn đường

## 2.2 Camera B — Event Validation (LUCID Triton2 EVS)

**Mục đích:** Capture real events để validate synthetic events

- Sensor: Sony IMX636
- Resolution: 1280×720 (target resolution cho toàn bộ pipeline)
- **Không** dùng để collect main dataset — chỉ dùng cho sim2real evaluation
- Deploy cùng scene với Camera A → so sánh synthetic vs real events

## 2.3 Hyperfocal Distance — Tại Sao Quan Trọng

Đặt focus tại hyperfocal distance H → mọi thứ từ H/2 đến infinity acceptably sharp. Ví dụ: nếu H = 10m → 5m đến infinity đều sharp. Với roadside traffic, xe ở các khoảng cách khác nhau → cần depth of field này.

Fixed focus sau đó → calibration consistency, không thay đổi lens geometry giữa các shots.

---

---

# PHẦN 3 — DATASET GENERATION PIPELINE

## 3.1 Overview Pipeline

```
.NEV files
  │
  ▼ [DaVinci Resolve]
  ├─ Decode NRAW (chỉ DaVinci đọc được proprietary format)
  ├─ Rough WB + lens profile
  └─ Export 16-bit TIFF @ 119.88fps, 1280×720
  │
  ▼ [Python / OpenCV]
  ├─ Load + validate uint16
  ├─ Dark frame subtraction        ← sensor thermal noise
  ├─ Flat field correction         ← vignetting + pixel non-uniformity
  ├─ White balance (gray card)     ← illuminant correction
  ├─ Linearization (γ⁻¹ = 2.2)  ← [CRITICAL] restore physical intensity
  ├─ RGB → Luminance Y (BT.709)  ← event cameras monochrome
  ├─ Undistort (K, D)             ← geometric accuracy for labels
  ├─ Gentle denoise (σ≤1.0)       ← optional
  ├─ Resize 1280×720              ← match event camera resolution
  └─ Stabilization (optical flow) ← remove tripod vibration
  │
  ▼ [v2e / DVS-Voltmeter]
  ├─ Temporal upsampling (SuperSloMo)
  ├─ log(Y) → ΔL → events per pixel
  ├─ ON/OFF threshold ± C
  └─ Noise models (shot, hot pixels, leak, threshold variation)
  │
  ▼ Event representations
  ├─ Event frames (2D, 10/20/50ms windows)
  ├─ Voxel grids (H × W × T bins)
  └─ Time surfaces (recency map)
```

## 3.2 Tại Sao Mỗi Preprocessing Step Bắt Buộc

### Dark Frame — Loại bỏ electronic baseline

Sensor luôn có non-zero output kể cả không có ánh sáng (thermal noise + Fixed Pattern Noise). Nếu không subtract: `log(signal + noise)` thay vì `log(signal)` → fake events ở dark regions.

*Cách làm:* Chụp 10–20 frames lens cap → average → subtract khỏi mọi frame.

### Flat Field — Đồng đều brightness spatial

Lens truyền ít ánh sáng hơn ở corners (vignetting). Mỗi pixel có sensitivity khác nhau. Không sửa → spatial brightness gradient → fake position-dependent events khi xe đi từ center ra edge frame.

*Cách làm:* Chụp uniform scene (tường trắng) → gain_map = mean/flat → multiply.

### Linearization — Bước quan trọng nhất

Camera lưu: `pixel ≈ (physical_light)^(1/2.2)` do gamma compression. Event cameras phản hồi với linear physical light. Nếu không undo gamma:
- Vùng tối bị stretched → threshold crossing sai → wrong event rate trong shadows
- Vùng sáng bị compressed → miss events trong bright areas

*Công thức:* `linear = (stored / 65535)^2.2 × 65535`

### RGB → Luminance Y (BT.709)

Event cameras không có màu sắc. Phải collapse 3 channels → 1. Simple average (R+G+B)/3 sai vì Green chiếm ~72% luminance perception. BT.709: `Y = 0.2126R + 0.7152G + 0.0722B` — calibrated cho Rec.709 color space (đúng với Nikon Z6 III).

### 16-bit TIFF (không phải 8-bit)

Event simulation dựa trên differences nhỏ trong log-intensity. 8-bit shadow regions: values 3, 4, 5 — difference của 1 có thể round xuống 0 → event lost. 16-bit: same region values 800, 850, 900 — difference 50 preserved. Direct impact lên event quality ở nighttime footage.

## 3.3 v2e vs DVS-Voltmeter

| | v2e | DVS-Voltmeter |
|-|-----|---------------|
| Model | Log-intensity threshold | Stochastic voltage dynamics (Brownian motion) |
| Accuracy | Good, fast | More physically accurate, slower |
| Noise model | Shot noise, hot pixels, threshold variation | Voltage noise, leak events, temporal dynamics |
| Use case | Baseline, high throughput | Higher-fidelity simulation, ablation |
| Output | (x,y,t,polarity) .h5 | (x,y,t,polarity) .h5 |

Paper sẽ compare model performance khi train trên v2e-generated vs DVS-Voltmeter-generated events.

## 3.4 Automatic Annotation Pipeline

**Trên RGB domain:**
1. **DINO** (vision foundation model) → detect cars, trucks, motorcycles, pedestrians, cyclists
2. **SAM** (Segment Anything) → pixel-level instance segmentation masks
3. **ByteTrack / SORT** → track across frames → consistent tracking IDs

**Transfer sang event domain:**
- Dùng undistortion parameters (K, D) và resize factor
- Map (x₁,y₁,x₂,y₂) coordinates từ RGB space → event frame space
- Masks được warped accordingly

**Kết quả:** Zero-human-annotation labeled dataset at scale.

---

---

# PHẦN 4 — MODEL TRAINING

## 4.1 Kiến Trúc Model

### Baseline: Frame-based Detection
- **Input:** Event frames (single-channel 2D images từ event accumulation)
- **Model:** YOLOv8 adapted cho single-channel input (thay RGB 3ch → event frame 1ch)
- **Lý do:** Familiar, fast iteration, strong baseline cho comparison

### State-of-the-art: RVT (Recurrent Vision Transformer)
- **Input:** Voxel grids (H × W × T)
- **Architecture:** Recurrent attention mechanism captures temporal event dynamics
- **Performance:** Current SOTA trên standard event-detection benchmarks (Gen1, Gen4)
- **Lý do chọn:** Traffic events có strong temporal structure — recurrent architecture khai thác điều này tốt hơn frame-by-frame approaches

### Tracking Module
- ByteTrack hoặc BoT-SORT integrated với detection output
- Event-specific: time surfaces có thể guide association khi detection confidence thấp

## 4.2 Event Representations và Ảnh Hưởng Đến Architecture

| Representation | Cách tạo | Input shape | Phù hợp với |
|---------------|---------|------------|-------------|
| Event frames | Accumulate events trong T ms | [B, 1, H, W] | CNN-based detectors |
| Voxel grids | T time bins, polarity-weighted | [B, T, H, W] | RVT, 3D conv |
| Time surfaces | Timestamp của event gần nhất | [B, 2, H, W] | Auxiliary input |

**Trade-off:** Time window T của event frames:
- T nhỏ (10ms): nhiều frames, ít events per frame, high temporal resolution
- T lớn (50ms): ít frames, dense events, thấy rõ object shape hơn nhưng mất temporal info

## 4.3 Dataset Split và Structure

```
CAROECT-D/
  RAW_NEV/
    site01/   ← highway overpass, 4-lane, daytime
    site02/   ← urban intersection, mixed lighting
    site03/   ← suburban road, nighttime
    ...
  Events_v2e/
    site01/   ← synthetic events từ site01
  Events_DVSVolt/
    site01/
  Labels/
    site01/   ← COCO-format JSON annotations
```

**Split:** Theo site (không theo frame). Frames từ cùng site có temporal correlation cao — split theo frame → data leakage, validation không meaningful.

## 4.4 Augmentation Strategies

**Standard:** Random flip, crop, mosaic, scale.

**Event-specific:**
- **Event drop:** Random remove X% events → simulate sensor packet loss, sensor aging
- **Temporal jitter:** Scale timestamps → simulate different motion speeds
- **Polarity noise:** Random flip polarities → simulate threshold variation giữa pixels
- **Spatial noise:** Add random spurious events → simulate hot pixel behavior
- **Rate scaling:** Change event density → simulate different scene dynamics

*Lý do event-specific augmentation quan trọng:* Tạo diversity trong training data mà model sẽ gặp khi deploy với real camera có noise characteristics khác.

## 4.5 Training Setup

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW + weight decay 1e-4 |
| LR schedule | Cosine annealing với warmup |
| Loss (YOLO-style) | Box regression + classification + objectness |
| Loss (DETR-style) | Hungarian matching |
| Batch size | 8–32 (depends on GPU memory và representation) |
| Hardware | Multi-GPU (voxel grids tốn memory) |

**Label format:** COCO JSON với event-specific metadata (timestamp range per annotation, event density per instance).

---

---

# PHẦN 5 — EVALUATION STRATEGY

## 5.1 Internal Validation: Synthetic vs Real Event Quality

**Câu hỏi:** Synthetic events từ pipeline có gần với real events từ LUCID EVS không?

**Protocol:**
1. Đặt LUCID EVS cạnh Nikon Z6 III, capture cùng scene
2. Generate synthetic events từ RGB
3. So sánh statistical distributions

**Metrics:**
- Event rate (events/pixel/second): synthetic vs real
- Polarity ratio (ON/OFF balance)
- Spatial density maps
- Noise floor estimation
- Threshold C estimation từ real events → so sánh với simulation parameters

## 5.2 Detection Performance

**Primary metrics:**
- **mAP@50:** Standard threshold, overall detection accuracy
- **mAP@50:95:** Stricter, COCO-style, đánh giá localization quality
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

## 5.4 Sim2Real Validation — Key Experiment của Paper

**Câu hỏi:** Model train trên CAROECT-D synthetic có generalize sang real event data không?

```
Train: CAROECT-D synthetic events
  ↓ Zero-shot test
  ├── Real LUCID EVS captures (same scenes)
  ├── eTram dataset (different location, different camera)
  └── TUMTraf Event dataset (different country, different setup)

Success: performance gap < 5–10% mAP → synthetic training viable
```

**Nếu gap nhỏ:** Paper's core claim validated — preprocessing quality is sufficient for sim2real transfer.
**Nếu gap lớn:** Cần ablate preprocessing steps để tìm root cause.

## 5.5 Ablation Studies — Chứng Minh Từng Bước Có Impact

### Ablation 1: Preprocessing Quality Impact

| Variant | Dark | Flat | WB | Linearize | Expected |
|---------|------|------|-----|-----------|---------|
| Full pipeline | ✓ | ✓ | ✓ | ✓ | Best mAP |
| No linearization | ✓ | ✓ | ✓ | ✗ | Drop: wrong event rates |
| No dark+flat | ✗ | ✗ | ✓ | ✓ | Drop: spatial noise artifacts |
| Minimal (raw) | ✗ | ✗ | ✗ | ✗ | Worst: all artifacts present |

*Mục tiêu:* Quantitatively chứng minh rằng physics-accurate preprocessing directly improves downstream model performance. Đây là một trong những novel contributions của paper.

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

*Nếu DVS-Voltmeter → higher mAP:* Justifies using more expensive simulation. *Nếu similar:* v2e sufficient, simpler pipeline.

---

---

# PHẦN 6 — RELATED WORK & POSITIONING

## 6.1 Các Papers Core Phải Biết

### v2e — Foundation của simulation

*Yang et al., CVPR 2021*

Event generation từ video frames bằng log-intensity threshold model:
- SuperSloMo temporal upsampling → high temporal resolution
- `L = log(Y)`, event khi `|ΔL| > C`
- Noise: shot noise, threshold variation, hot pixels, leak events
- Output: (x,y,t,polarity) stream

**Relation to CAROECT-D:** v2e là primary event simulator. Paper cite v2e extensively. Linearization step trong preprocessing pipeline exist specifically để feed đúng format vào v2e.

### DVS-Voltmeter — Advanced simulation

*Lin et al.*

Model voltage dynamics của photodiode circuit (Brownian motion with drift), không chỉ simple threshold. Captures phenomena v2e misses: temporal clustering, refractory period, voltage leak.

**Relation:** Alternative simulator. CAROECT-D compare v2e vs DVS-Voltmeter generated data.

### eTram — Real event traffic dataset

Urban tram tracking với event camera. Ground-truth tracking annotations. Small scale nhưng real.

**Relation:** Primary benchmark cho sim2real evaluation. Test model trained on CAROECT-D → evaluate on eTram → measure generalization.

### TUMTraf Event — Multimodal traffic dataset

RGB + event camera, roadside, TU Munich. Multiple scene types, multiple object classes.

**Relation:** Cross-domain validation. Different country, different camera setup → tests robustness của sim2real.

## 6.2 CAROECT-D Gap Trong Literature

| | eTram | TUMTraf | Prophesee Gen1/4 | **CAROECT-D** |
|-|-------|---------|------------------|----------------|
| Scale | Small | Medium | Medium | **Large** |
| Annotation | Manual | Manual | Annotated | **Automatic** |
| Hardware needed | Event cam | Event cam + RGB | Event cam | **RGB only** |
| Diversity | Low | Medium | Medium | **High** |
| Synthetic? | No | No | No | **Yes** |

---

---

# PHẦN 7 — RESEARCH CONTRIBUTIONS

## Contribution 1: Physics-Accurate Preprocessing Pipeline

**Novel:** Systematic methodology để convert RGB RAW → event-simulation-ready linear luminance, với full sensor calibration (dark frame, flat field, white balance, geometric undistortion).

**Chứng minh bởi:** Ablation studies showing preprocessing quality directly correlates with event simulation fidelity và downstream model performance. Không paper nào trước quantify điều này.

## Contribution 2: CAROECT-D Dataset

Large-scale, diverse, automatically-labeled roadside traffic event dataset:
- Multiple sites, multiple lighting conditions, multiple weather conditions
- Full annotations: bounding boxes, instance masks, tracking IDs
- Both v2e và DVS-Voltmeter variants

## Contribution 3: Zero-Annotation Pipeline

End-to-end automatic annotation: DINO + SAM (RGB detection) → ByteTrack (RGB tracking) → geometric transfer → event domain labels. **Zero human annotation required.** Demonstrates scalability: any RGB traffic footage → labeled event dataset.

## Contribution 4: Sim2Real Validation

Quantitative evidence rằng CAROECT-D-trained models generalize effectively đến real event camera data. Establishes synthetic event training as viable alternative cho event-camera traffic perception — điều chưa được chứng minh ở traffic domain.

---

---

# PHẦN 8 — Q&A PREPARATION

**Q: Tại sao không chỉ dùng event camera thật?**

A: Cost (hardware $5k–30k), annotation impossibility at scale, limited location flexibility. CAROECT-D scales to any RGB footage — historical traffic footage, multiple cameras simultaneously, diverse worldwide locations — at marginal cost. Real event cameras remain as validation tools, not as primary data collection.

---

**Q: Sim2real gap có lớn không? Làm sao biết synthetic đủ tốt?**

A: Validated bằng hai approaches: (1) Statistical comparison với real LUCID EVS captures cùng scene — event rate, polarity ratio, density maps. (2) Downstream task performance — nếu model trained on synthetic achieves competitive mAP on real event benchmarks, pipeline is validated. Ablation studies quantify contribution của từng preprocessing step đến closing the gap.

---

**Q: DINO + SAM có accurate đủ để làm training labels không?**

A: Foundation models achieve >90% mAP trên standard benchmarks. Theo noise-learning literature, high-volume noisy labels often outperform low-volume clean labels cho training. Chúng tôi filter bằng confidence threshold và temporal consistency (tracking) để reduce noise.

---

**Q: Tại sao cần RAW? Không thể dùng H.264 video thông thường?**

A: Consumer video apply: gamma compression (irreversible), aggressive noise reduction (destroys sensor physics), lossy compression (corrupts pixel values), potential dynamic WB changes (temporal brightness shifts → fake events). RAW preserves sensor data trước mọi processing. Linearization pipeline chỉ hoạt động tốt với RAW input.

---

**Q: v2e đủ chưa hay cần DVS-Voltmeter?**

A: v2e là good baseline và significantly faster. DVS-Voltmeter physically more accurate, especially cho low-contrast và static-rich scenes. Paper compare cả hai — nếu DVS-Voltmeter cho better sim2real performance, additional complexity justified. Nếu similar, v2e sufficient.

---

**Q: Tại sao 16-bit TIFF? Không thể dùng PNG hay JPEG?**

A: JPEG: lossy compression distorts pixel values → wrong physics. PNG: typically 8-bit (256 levels). Event simulation cần precision ở dark regions — 16-bit (65535 levels) preserves subtle intensity differences mà 8-bit rounds away. Impact đặc biệt rõ ở nighttime footage.

---

**Q: Làm sao scale lên nhiều sites? Pipeline bao lâu mỗi hour of footage?**

A: DaVinci export: real-time hoặc 2–4x faster. Python preprocessing: parallelizable theo frame. Event simulation: v2e ~ 5–10x real-time trên GPU. Total: roughly 30–60 min processing cho 1 hour footage tại 120fps. Annotation: DINO+SAM chạy parallel, không phải bottleneck.

---

---

# GLOSSARY — Quick Reference

| Term | Định nghĩa |
|------|-----------|
| **Event camera / DVS** | Camera fires (x,y,t,polarity) events thay vì frames |
| **ON / OFF event** | ON = pixel sáng lên vượt threshold, OFF = tối đi |
| **Threshold C** | Min log-intensity change để trigger event (0.2–0.5 typical) |
| **Log-intensity** | `L = log(I)` — event cameras respond to this, not linear I |
| **Linearization** | Undo gamma: `linear = compressed^2.2` |
| **BT.709 luminance** | `Y = 0.2126R + 0.7152G + 0.0722B` |
| **Dark frame** | Lens cap photo → sensor noise baseline |
| **Flat field** | Uniform scene photo → vignetting + pixel non-uniformity map |
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
| **Sim2real** | Generalization từ synthetic training đến real deployment |
| **DINO** | Vision foundation model cho object detection |
| **SAM** | Segment Anything Model — pixel-level segmentation |
| **ByteTrack** | SOTA multi-object tracker |
| **Hyperfocal distance** | Focus point cho max depth of field: H/2 đến ∞ sharp |
| **CFexpress Type B** | Storage card format đủ nhanh cho NRAW 120fps |
| **IMX636** | Sony sensor trong LUCID Triton2 EVS event camera |

---

---

# TIMELINE — Presentation Flow (20–25 phút)

| Phần | Nội dung | Thời gian |
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

*Brief này cover toàn bộ scope của CAROECT-D từ motivation → hardware → preprocessing → training → evaluation → contribution.*
*Version: 1.0 — Prepared for internal research presentation*
