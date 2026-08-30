# CAROECT-D: A Scalable Dataset of RGB-Derived Event Streams for Roadside Traffic Monitoring

*Draft — Abstract, Introduction, Related Work (new), and Methodology (consolidated from prior work, formulas verified)*

---

## Abstract

Event cameras offer microsecond-scale temporal resolution and a dynamic range in excess of 120 dB, properties that make them attractive for roadside traffic perception under the glare, twilight, and low-light conditions that routinely degrade conventional frame cameras. Their adoption is nonetheless limited by the scarcity of large-scale annotated event data: physical event sensors remain far more costly and rare than commodity RGB cameras, and manually labeling asynchronous, colorless event streams at scale is prohibitively expensive. Frame-to-event simulation offers a path to scalable dataset construction, but existing simulators and RGB–event benchmarks suffer from three compounding weaknesses: (i) they are typically driven by 8-bit, gamma-encoded video, which we show mathematically corrupts the log-intensity derivative that every DVS event-generation model depends on; (ii) they are run with simulator parameters tuned on a different sensor and scene distribution than the one being modeled, rather than calibrated against a co-located real event camera; and (iii) when real and RGB annotations must be fused across two physically separated sensors, label transfer requires error-prone targetless calibration. We present CAROECT-D, a roadside dataset-construction pipeline that instead treats geometry, radiometry, and annotation as one shared-by-construction system: a single 16-bit linear scene representation is undistorted once, and only afterward forked into a photometric annotation branch — automatically labeled by Segment Anything Model 3 (SAM 3) — and a synthetic event branch, driven jointly by v2e's deterministic threshold model and the Brownian-motion voltage model of DVS-Voltmeter and Raw2Event, calibrated against a co-located real Sony IMX636 event sensor. Because the geometric transform is frozen prior to the fork, RGB-derived bounding boxes, masks, and track identities map onto the synthetic event stream with no cross-sensor reprojection. This paper formalizes the mathematical pipeline underlying CAROECT-D and positions it relative to ESIM, vid2e, v2e, DVS-Voltmeter, Raw2Event, eTraM, and TUMTraf Event.

---

## I. Introduction

Event cameras (dynamic vision sensors, DVS) depart fundamentally from conventional frame cameras: instead of exposing a full sensor array at a fixed rate, each pixel independently and asynchronously reports a change in log-intensity the instant that change crosses a fixed contrast threshold. The resulting representation is sparse, asynchronous, and colorless, but it inherits two properties directly from that per-pixel, log-domain operation: temporal resolution on the order of microseconds, and a dynamic range exceeding 120 dB — roughly four orders of magnitude beyond a typical RGB sensor. For roadside intelligent transportation systems (ITS), where a single fixed camera must remain useful across full daylight, direct low-sun glare, dusk, and unlit night conditions, this HDR property is not a curiosity but a direct mitigation for the single largest failure mode of RGB-only traffic perception.

The obstacle to exploiting this property at scale is data. Modern detection and tracking architectures are trained on large annotated corpora, and building such a corpus directly on real event hardware is doubly expensive: event cameras with automotive-grade resolution and dynamic range (e.g., the Sony IMX636 sensor used in the Prophesee EVK4 HD and in the LUCID Triton2) remain costly and comparatively rare relative to commodity RGB cameras, which constrains how many physical sites can be instrumented; and because event data has no persistent image to click a bounding box on, manual annotation requires either specialized tooling or a fusion RGB sensor, and is correspondingly slower and more expensive than labeling ordinary video. eTraM [6] illustrates the cost of the fully-real approach directly: even with a dedicated, purpose-built acquisition effort, it required more than ten hours of manually annotated footage to cover just three roadside scene types. TUMTraf Event [7] shows a hybrid alternative — pairing a real event camera with a real RGB camera and transferring RGB-derived pseudo-labels across — but because the two sensors occupy physically distinct viewpoints, this requires a dedicated targetless extrinsic calibration procedure (motion-edge extraction, DBSCAN clustering, and iterative closest point matching), which the authors report introduces 3.37–6.72 pixels of reprojection error even after careful tuning.

Simulating events directly from RGB video is the natural way to scale past this bottleneck: RGB roadside cameras are already ubiquitous, and the annotation problem reduces to running a mature RGB detector or promptable segmentation model rather than labeling raw event streams by hand. However, the existing simulator lineage — ESIM [1], vid2e [2], and v2e [3] — was developed and validated primarily on rendered or consumer 8-bit video, not on footage captured, calibrated, and linearized specifically for the purpose of driving an event simulator. This matters more than a stylistic preference for image quality: because DVS event generation is a threshold on the *derivative* of log-intensity, a nonlinear (gamma) response curve and an 8-bit dynamic range interact to break the simulator at both ends of the exposure scale simultaneously — flooding shadow regions with spurious events from single-bit quantization steps, while going effectively blind to real motion in bright regions where the available quantization is coarser than the trigger threshold. We derive this failure mode explicitly in Section III-B. A second, largely orthogonal weakness is timing fidelity: v2e's deterministic linear interpolation of event timestamps within a frame interval causes synthetic events to cluster at frame boundaries — a "temporal layering" artifact that DVS-Voltmeter [4] and its embedded-systems descendant Raw2Event [5] resolve by modeling the per-pixel photodiode voltage as a Brownian motion with drift, drawing event timestamps from the resulting first-passage-time (Inverse Gaussian) distribution instead.

CAROECT-D is designed around three principles that follow directly from the gaps identified above. First, capture uses 12-bit SDR N-RAW rather than N-Log, and DaVinci Resolve performs the declared SDR-to-linear transform before exporting a 16-bit linear Rec.709 working signal. The Python pipeline treats this input as already linear and never applies a second inverse transfer function. Optional dark, flat-field, white-balance, and exposure corrections remain disabled unless measurement-backed artifacts are valid. Second, lens undistortion and resizing are computed **exactly once**, on that shared 16-bit array, *before* the pipeline forks into an sRGB-toned annotation branch and a linear-luminance event-simulation branch; this single design choice is what lets Section III-E reduce cross-modal label transfer to an identity mapping rather than a calibrated projection. Third, the event-generation stage itself is not a single simulator run at default parameters: it combines v2e's deterministic, intensity-dependent contrast-thresholding model with the stochastic, Brownian-motion voltage model of DVS-Voltmeter/Raw2Event, with both models' thresholds calibrated against a real, co-located Sony IMX636 event camera rather than left at published defaults. Automated annotation of the RGB branch is performed with SAM 3 [10], whose promptable concept segmentation returns masks, boxes, and persistent track identities for an entire traffic-participant vocabulary ("car," "truck," "bus," "motorcycle," "person") in a single pass, requiring no manual event-domain labeling.

The contributions of this paper are: (1) a mathematical account of why 16-bit linear photometry is a hard requirement, not a quality preference, for physically faithful event simulation; (2) a shared-geometry pipeline design that structurally eliminates the cross-sensor reprojection error reported by prior hybrid datasets such as TUMTraf Event; (3) a dual event-generation formulation that couples v2e's deterministic thresholding with DVS-Voltmeter/Raw2Event's stochastic voltage model within one site-calibratable pipeline; and (4) a formal spatiotemporal label-transfer function built on SAM 3's video predictor that requires no manual annotation in the event domain. This paper documents the dataset-construction methodology; quantitative evaluation, including domain transfer to eTraM, is left to a subsequent report once data collection is complete.

The remainder of the paper is organized as follows. Section II reviews physics-based event simulators, real and hybrid event-based traffic datasets, and promptable segmentation foundation models. Section III presents the CAROECT-D methodology in full: radiometric linearization (III-B), geometric calibration (III-C), the dual event-simulation model (III-D), and cross-modal label transfer via SAM 3 (III-E).

---

## II. Related Work

### A. Physics-Based Event Camera Simulators

**ESIM [1]** was the first general-purpose event camera simulator, generating events by applying a fixed contrast threshold — with Gaussian spatial jitter across the sensor — to log-intensity frames produced by a tightly-coupled rendering engine. Because it was designed and validated primarily against synthetic, rendered scenes, ESIM assumes noiseless, arbitrarily-high-frame-rate input and has no mechanism for calibrating its threshold against a physical sensor deployed in a specific outdoor environment.

**vid2e [2]** does not introduce a new event-generation model; it wraps ESIM with a learned frame-interpolation front end (Super-SloMo) that upsamples ordinary video before simulation, addressing the case where consecutive input frames are too far apart in time for ESIM's model to remain accurate. This targets temporal *undersampling* specifically, not radiometric or noise fidelity.

**v2e [3]** models three DVS non-idealities largely absent from ESIM: a pixel-level Gaussian contrast-threshold mismatch that mimics fabrication variance, a finite, intensity-dependent photoreceptor bandwidth (modeled as a first-order low-pass filter whose cutoff frequency drops in dim light — the physical source of "motion blur" in real DVS output), and shot/leak noise. v2e nonetheless still assigns event timestamps by deterministic linear interpolation within each source-video frame interval, which — as we discuss in Section III-D — clusters synthetic timestamps at frame boundaries ("temporal layering").

**DVS-Voltmeter [4]** replaces deterministic thresholding with a physical circuit model: the per-pixel photodiode voltage is modeled as a Brownian motion with drift, and an event is emitted the instant that voltage first crosses an ON or OFF barrier. Event timestamps are consequently drawn from the *first-passage-time distribution* of that process (an Inverse Gaussian), rather than interpolated, which the original authors show reproduces the irregular, non-frame-locked timing statistics of real DVS output more closely than ESIM, vid2e, or v2e. The cost is six sensor-specific parameters that must be calibrated rather than assumed.

**Raw2Event [5]** builds directly on the DVS-Voltmeter voltage model but replaces ISP-processed RGB input with raw Bayer sensor data, preserving dynamic range that a conventional ISP pipeline would otherwise compress, and contributes a search-based, human-interpretable procedure for calibrating DVS-Voltmeter's six parameters on embedded hardware (demonstrated on a Raspberry Pi). Raw2Event is the closest prior work to CAROECT-D's own radiometric philosophy — bypass the ISP, calibrate rather than assume — though it targets a single low-cost, real-time embedded deployment rather than large-scale, multi-site roadside dataset construction.

Across [1]–[5], two gaps recur: none of these simulators, as published, is evaluated within a pipeline that calibrates its threshold and noise parameters against a co-located real event sensor across multiple outdoor sites and lighting conditions; and — with the partial exception of Raw2Event's raw-Bayer input — the majority of reported experiments still operate on 8-bit-derived video, which Section III-B shows breaks the event-triggering condition itself, independent of which of the four threshold/noise models above is used downstream.

### B. Real and Hybrid Event-Based Traffic Datasets

**eTraM [6]** is a fully real, manually annotated dataset built with a Prophesee EVK4 HD (1280×720 px, >10,000 fps effective temporal resolution, >120 dB dynamic range, 0.08 lux low-light cutoff), mounted at roughly 6 m height and a 35° pitch to match typical roadside traffic-camera placement. It spans over ten hours of footage across intersection, roadway, and local-street scenes under varied weather and lighting, with over two million bounding boxes across vehicles, pedestrians, and micro-mobility. Because it is fully real, eTraM is the natural target for evaluating whether a *synthetic*, RGB-derived dataset such as CAROECT-D generalizes to physical event-camera data (Section III-E's label-transfer machinery exists precisely so that CAROECT-D-trained models can later be tested, zero-shot, on eTraM).

**TUMTraf Event [7]** co-locates a real event camera with a real RGB camera and transfers RGB-derived (YOLOv7-based) pseudo-labels into the event domain. Because the two sensors occupy physically separated viewpoints, this transfer requires a dedicated targetless extrinsic calibration pipeline — motion-edge extraction, DBSCAN clustering, and iterative closest point matching — which the authors report yields 3.37–6.72 pixels of reprojection error even after tuning, across 4,111 synchronized frame pairs and 50,496 labeled boxes. TUMTraf Event demonstrates conclusively that RGB→event label transfer is a viable and valuable strategy; it also demonstrates, empirically, that any two-camera baseline imposes a measurable calibration cost that no amount of tuning removes.

CAROECT-D's central structural difference from TUMTraf Event is that it never instantiates two physically separated sensors *within the label-transfer path*. The same undistorted, 16-bit linear scene representation is the shared source for both the SAM-3-annotated branch and the synthetic event branch (Section III-C), so the transform relating an RGB-branch pixel coordinate to the corresponding synthetic-event coordinate is the identity map rather than a calibrated homography — the reprojection error reported in [7] is eliminated by construction rather than reduced by better calibration.

### C. Promptable Visual Foundation Models for Automated Annotation

**SAM [8]** established prompt-based, zero-shot interactive segmentation: given a point, box, or mask prompt, the model returns a single instance mask without task-specific fine-tuning, trained on the large-scale SA-1B dataset. Annotating $N$ objects in a frame still requires $N$ prompts.

**SAM 2 [9]** extends this to video via a memory-based tracker: a prompted mask is propagated frame-to-frame as a persistent object using spatial memory features, but retains the one-prompt-per-object-instance limitation of SAM.

**SAM 3 [10]**, released in November 2025, removes that limitation with *Promptable Concept Segmentation* (PCS): a short noun phrase (e.g., "yellow school bus") or an image exemplar returns segmentation masks and unique identities for **every** matching instance across an image or video in a single pass. Architecturally, SAM 3 couples a DETR-style image-level detector with a SAM-2-style memory-based video tracker on a shared Perception Encoder backbone (≈848M parameters total), and decouples recognition from localization with a learned global *presence token* that predicts whether a concept exists anywhere in the frame independently of any individual object query's localization decision — reported to roughly double accuracy over prior open-vocabulary systems on the authors' SA-Co benchmark. CAROECT-D uses SAM 3's video predictor as the sole source of RGB-branch annotation, queried with a fixed traffic-participant vocabulary ("car," "truck," "bus," "motorcycle," "person"), formalized in Section III-E.

---

## III. Methodology

*(Consolidated from the previously finalized methodology drafts. Formulas below have been checked for internal consistency and, where an equation was stated only as an approximation, that fact is now made explicit — see the note after Eq. 12.)*

### A. Setting and Notation

Traffic scenes are captured with a Nikon Z6 III recording 12-bit SDR N-RAW (not N-Log), rigidly co-located with a LUCID Triton2 event camera built around the Sony IMX636 sensor. DaVinci Resolve decodes and linearizes the SDR material, then exports 16-bit linear Rec.709 RGB at the native 119.88 fps cadence. From this single capture, the pipeline branches into an explicitly modeled sRGB annotation branch and a 16-bit linear event-generation branch. The RGB branch yields, per frame, tracking IDs, bounding boxes, and binary masks (Section III-E). The event branch yields a stream of tuples $e_k=(x_k,y_k,t_k,p_k)$, denoting spatial coordinates, a microsecond timestamp, and polarity $p_k\in\{0,1\}$.

### B. Radiometric Preprocessing and Photometric Linearization

Event cameras operate according to Weber's law: the sensor responds to the temporal derivative of **log**-intensity,

$$L(x,y,t) = \ln\big(I(x,y,t)\big), \tag{1}$$

and an event is triggered at pixel $(x,y)$ the instant

$$\Delta L = \ln I_t - \ln I_{t-1} \;\ge\; \theta \quad(\text{ON}), \qquad \Delta L \le -\theta \quad(\text{OFF}). \tag{2}$$

**Why gamma-encoded 8-bit video breaks (2).** Standard RGB pipelines output frames after a non-linear gamma encode, $I_{\text{sRGB}} \approx I^{1/\gamma}$ with $\gamma\approx 2.2$, so that

$$L_{\text{sRGB}} = \ln\!\big(I^{1/\gamma}\big) = \tfrac{1}{\gamma}\ln I. \tag{3}$$

This fractional scaling compresses $\partial L/\partial t$ non-uniformly across the exposure range, and — independently of the gamma issue — an 8-bit frame quantizes $I$ to only 256 discrete levels. At the extremes of the exposure scale this quantization, not just the gamma curve, is what breaks event generation. In a dark region, the smallest possible digital step is a jump from level 1 to level 2:

$$\Delta L_{\text{dark}} = \ln 2 - \ln 1 = \ln 2 \approx 0.693, \tag{4}$$

which, for a typical contrast threshold $\theta\approx 0.1$–$0.3$, is large enough to trigger an ON event from a single quantization step of sensor noise — flooding shadow regions with spurious events. In a bright region, an equally physically plausible step from level 254 to 255 gives

$$\Delta L_{\text{bright}} = \ln 255 - \ln 254 \approx 0.00393, \tag{5}$$

which is far below $\theta$: real motion under bright, glaring conditions — the exact regime event cameras are meant to help with — can be silently dropped. Both failure modes stem from the same root cause: 8-bit gamma encoding destroys the near-linear relationship between digital value and photon count that Eq. (2) implicitly assumes.

**16-bit linear working photometry.** CAROECT-D records 12-bit SDR N-RAW and uses a fixed DaVinci Resolve transform to export 16-bit linear Rec.709 TIFFs at the native 119.88 fps cadence. This provides 65,536 working code values rather than 256 and declares the linear-light pipeline boundary explicitly. It does not imply that SDR capture matches the physical IMX636 dynamic range or that the TIFF codes are untouched photosite measurements; those are empirical calibration questions.

Operating on the declared DaVinci-linearized signal permits standard radiometric corrections to be evaluated prior to any event-related computation:

$$I_{\text{corr}} = (I_{\text{raw}} - I_{\text{dark}})\cdot M_{\text{gain}} \cdot W_{\text{gain}} \cdot S_{\text{exp}}, \tag{6}$$

where $I_{\text{dark}}$ is the dark-frame mean (lens-cap sensor bias), $M_{\text{gain}}$ is a flat-field gain map removing systematic vignetting, $W_{\text{gain}}$ applies fixed, gray-card-measured white-balance multipliers (not per-frame auto white balance, which would otherwise inject a spurious brightness step — and hence a spurious event — into every pixel simultaneously whenever the camera re-adjusts), and $S_{\text{exp}}$ normalizes exposure across capture sessions. Applying these corrections *before* any thresholding step ensures that spatial artifacts of the lens and sensor (vignetting, fixed-pattern bias) do not masquerade as temporal scene changes.

### C. Geometric Lens Calibration and Shared Undistortion

Automated cross-modal label transfer (Section III-E) requires pixel-perfect spatial alignment between the branch SAM 3 annotates and the branch the event simulator consumes. Raw lenses violate the ideal pinhole model,

$$\begin{bmatrix}u\\v\\1\end{bmatrix} = \begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}\begin{bmatrix}X_c\\Y_c\\Z_c\end{bmatrix}, \tag{7}$$

introducing radial and tangential distortion that warps ideal coordinates $(x,y)$ into observed coordinates $(x_{\text{dist}}, y_{\text{dist}})$. CAROECT-D corrects this with the Brown–Conrady model, calibrating intrinsics $K$ and distortion coefficients $D$ from a checkerboard target:

$$x_{\text{dist}} = x\big(1+k_1r^2+k_2r^4+k_3r^6\big) + \big[2p_1xy+p_2(r^2+2x^2)\big], \tag{8}$$

$$y_{\text{dist}} = y\big(1+k_1r^2+k_2r^4+k_3r^6\big) + \big[p_1(r^2+2y^2)+2p_2xy\big], \tag{9}$$

with $r=\sqrt{x^2+y^2}$, radial coefficients $k_{1,2,3}$, and tangential coefficients $p_{1,2}$.

**Shared architecture.** This undistortion, together with a fixed affine resize, is computed **exactly once**, applied to the shared 16-bit linear array, and only afterward does the pipeline fork: the annotation branch applies an sRGB tone curve for SAM 3 (Section III-E), while the event branch extracts a linear luminance signal for simulation (Section III-D). Because geometry is frozen *before* this photometric split, any residual error in $\Phi$ or in the resize step propagates identically into both branches — it cannot introduce a *relative* misalignment between them. This single design choice is the structural reason Section III-E's label-transfer step reduces to an identity mapping rather than a calibrated cross-sensor projection.

### D. Event Simulation: Deterministic Thresholding and Stochastic Voltage Modeling

CAROECT-D generates events from the shared linear representation using two complementary models, run and calibrated together rather than as alternatives.

#### D.1 Deterministic Thresholding (v2e)

Following v2e [3], the already-linear Rec.709 frame is first reduced to a luma signal $Y = c_R R + c_G G + c_B B$ using the matching coefficients $(c_R,c_G,c_B)=(0.2126,0.7152,0.0722)$. To avoid the $\ln(Y)\to-\infty$ singularity as $Y\to0$, log-intensity is computed with a hybrid mapping about a small cutoff $Y_c$ (nominally 20 DN):

$$L_{\text{in}} = \begin{cases} \ln(Y), & Y \ge Y_c \\[2pt] \dfrac{Y}{Y_c}\ln(Y_c), & Y < Y_c. \end{cases} \tag{10}$$

$L_{\text{in}}$ then passes through a first-order, intensity-dependent low-pass filter modeling the finite bandwidth of a real photoreceptor circuit:

$$L_{\text{lp}}[k] = L_{\text{lp}}[k-1] + \alpha\big(L_{\text{in}}[k]-L_{\text{lp}}[k-1]\big), \qquad \tau = \frac{1}{2\pi f_c(Y)}, \tag{11}$$

with cutoff frequency $f_c(Y)$ increasing monotonically in brightness $Y$ (so that dim pixels respond more sluggishly than bright ones, the physical origin of DVS "motion blur" in low light). The discrete update coefficient is conventionally approximated as

$$\alpha = \Delta t/\tau \tag{12}$$

which is a first-order approximation valid for $\Delta t \ll \tau$; the exact discretization of a continuous-time first-order lag is $\alpha = 1-e^{-\Delta t/\tau}$, which CAROECT-D uses in implementation to avoid instability when $\tau$ is small under bright, high-cutoff conditions.

An event is triggered on the difference between the filtered signal and the value memorized at the pixel's last event, $L_{\text{mem}}$:

$$\Delta L = L_{\text{lp}} - L_{\text{mem}}, \tag{13}$$

$$N_e = \Big\lfloor \frac{|\Delta L|}{\theta} \Big\rfloor \quad \text{events, triggered when } |\Delta L|\ge\theta, \tag{14}$$

$$L_{\text{mem}} \leftarrow L_{\text{mem}} + N_e\cdot\theta\cdot\operatorname{sign}(\Delta L). \tag{15}$$

Three non-idealities are layered on top of (13)–(15): a frozen, per-pixel Gaussian threshold mismatch modeling fabrication variance,

$$\theta_{\text{pixel}} \sim \mathcal{N}\big(\theta_{\text{nom}},\,\sigma_\theta^2\big), \tag{16}$$

a continuous leak that can generate spontaneous events even in a static scene,

$$L_{\text{mem}}(t) = L_{\text{mem}}(t_0) - R_{\text{leak}}(t-t_0), \tag{17}$$

and shot noise, whose per-step firing probability increases as brightness $Y$ falls (consistent with the Poisson statistics of photon arrival, whose signal-to-noise ratio scales as $\sqrt{\lambda}/\lambda = 1/\sqrt{\lambda}$):

$$p = R_n(Y)\cdot\Delta t. \tag{18}$$

Finally, v2e assigns timestamps to the $N_e$ events generated between two consecutive source frames at $t_j$ and $t_{j+1}$ by **deterministic linear interpolation**:

$$t_k = t_j + k\cdot\frac{t_{j+1}-t_j}{N_e}, \qquad k\in\{1,\dots,N_e\}. \tag{19}$$

This forces every synthetic timestamp to align with the artificial video frame rate — a phenomenon termed *temporal layering* — which detectors trained on such data can overfit to, degrading performance when tested against real, non-frame-locked event data.

#### D.2 Stochastic Voltage Modeling (DVS-Voltmeter / Raw2Event)

To resolve temporal layering, CAROECT-D also drives simulation with the DVS-Voltmeter [4] / Raw2Event [5] model, which represents the per-pixel photodiode voltage change directly as a Brownian motion with drift, rather than thresholding a deterministically-sampled signal:

$$\Delta V_d = \mu\,\Delta t + \sigma\,W(\Delta t), \tag{20}$$

where $W(\Delta t)$ is a standard Wiener process. Drift $\mu$ and diffusion scale $\sigma$ are computed from six calibrated parameters $k_1,\dots,k_6$ tying the abstract stochastic process back to physical circuit behavior:

$$\mu = \frac{k_1}{\bar L + k_2}\,k_{dL} + k_4 + k_5\bar L, \tag{21}$$

$$\sigma = \frac{k_3}{\bar L + k_2}\sqrt{\bar L} + k_6, \tag{22}$$

with $\bar L$ the local average brightness, $k_{dL}$ its rate of change, $k_1$ a signal-gain term, $k_2$ a dark-region floor, $k_4,k_5$ thermal and parasitic-photocurrent leakage drift, $k_3$ a shot-noise-driven jitter gain, and $k_6$ a baseline electronic noise floor.

**Domain-matched calibration.** Rather than adopting the published defaults for $\theta$, $\Theta_{\text{ON}}$, $\Theta_{\text{OFF}}$, and $k_1,\dots,k_6$, CAROECT-D estimates the real contrast threshold $C_{\text{real}}$ of the co-located Sony IMX636 sensor by imaging a calibrated gray-gradient target and counting mean events per known log-intensity step $\Delta L$:

$$\bar N(\Delta L) \approx \frac{\Delta L}{C_{\text{real}}} \;\;\Longrightarrow\;\; C_{\text{real}} = \frac{\Delta L}{\bar N(\Delta L)}, \tag{23}$$

fit across multiple $\Delta L$ steps and multiple lighting conditions. $\theta$ (Eq. 14) and $\Theta_{\text{ON}}/\Theta_{\text{OFF}}$ (below) are then set from $C_{\text{real}}$, and $k_1,\dots,k_6$ are subsequently fit to match measured static-scene event rate, motion-triggered event rate, and edge latency between the real and simulated streams. This calibration step — present in Raw2Event's embedded, single-device setting but here extended across multiple roadside sites and lighting conditions — is what Section II identifies as absent from ESIM, vid2e, and v2e as published.

Under (20)–(22), the timestamp of the next event is the **first passage time** of $\Delta V_d$ against a voltage barrier $\Theta_{\text{ON}}$ or $\Theta_{\text{OFF}}$ (in the DVS-Voltmeter circuit convention, an ON event corresponds to $\Delta V_d(t)\le -\Theta_{\text{ON}}$ and an OFF event to $\Delta V_d(t)\ge \Theta_{\text{OFF}}$, owing to the sign inversion of the log-domain voltage stage). This first-passage time is Inverse-Gaussian distributed:

$$\tau_{\text{ON}},\tau_{\text{OFF}} \sim \mathrm{IG}\!\left(\frac{\mp\Theta_{\text{ON/OFF}}}{\mu},\; \frac{\Theta_{\text{ON/OFF}}^2}{\sigma^2}\right), \tag{24}$$

which — unlike Eq. (19) — spreads simulated timestamps continuously rather than clustering them at frame boundaries, directly addressing temporal layering. Polarity is resolved as a classical gambler's-ruin absorption probability — the probability that the drifting, noisy voltage trajectory reaches the ON barrier before the OFF barrier:

$$P(\text{ON}) = \frac{\exp\!\big(-2\mu\Theta_{\text{ON}}/\sigma^2\big) - 1}{\exp\!\big(-2\mu\Theta_{\text{ON}}/\sigma^2\big) - \exp\!\big(2\mu\Theta_{\text{OFF}}/\sigma^2\big)}. \tag{25}$$

#### D.3 Temporal Sampling Rate and Interpolation Fidelity

Both models above ultimately approximate a continuous derivative from discrete frames:

$$\frac{\partial L}{\partial t} \approx \frac{L(t_{i+1}) - L(t_i)}{\Delta t}. \tag{26}$$

At a conventional 30 fps ($\Delta t \approx 33.3$ ms), a fast-moving vehicle can undergo spatial displacement $\Delta x = v\cdot\Delta t$ large enough that the optical-flow assumptions underlying (19) and even (24) begin to break down, producing "ghosting" — smeared, disjointed event clouds with no sharp motion boundary. CAROECT-D captures and processes raw video at 119.88 fps, reducing the temporal step to

$$\Delta t \approx \frac{1}{119.88\ \text{s}^{-1}} \approx 8.34\ \text{ms}, \tag{27}$$

roughly a $4\times$ reduction in $\Delta x$ for the same vehicle speed relative to 30 fps, tightening the approximation in (26) and localizing simulated events around the true physical edges of moving objects.

### E. Cross-Modal Label Transfer and Spatiotemporal Alignment

#### E.1 SAM 3 Annotation of the RGB Branch

CAROECT-D uses SAM 3's Promptable Concept Segmentation [10] on the sRGB-toned annotation branch to obtain, for a fixed traffic-participant vocabulary (car, truck, bus, motorcycle, person), every matching instance's mask, box, and identity in a single pass, rather than one prompt per object as required by SAM [8] or SAM 2 [9]. SAM 3 decouples recognition from localization with a learned presence token; **we formalize this design choice** — as a chain-rule probability decomposition, for use within our own pipeline description, rather than as a literal reproduction of SAM 3's internal loss or architecture — as

$$p(\text{query}_i \text{ matches NP}) = p(\text{query}_i \text{ matches NP} \mid \text{NP appears in image}) \cdot p(\text{NP appears in image}). \tag{28}$$

For video, SAM 3's memory-based tracker propagates each confirmed detection as a spatio-temporal *masklet* $\hat{\mathcal M}_\tau^n$ for object $n$ at frame $\tau$, using periodic re-detection to recover from occlusion. **We similarly formalize** this re-detection/confirmation behavior, for our own bookkeeping rather than as SAM 3's literal internal metric, with an indicator function comparing the propagated masklet against the frame-level detector output $\mathcal D_\tau$ via Intersection-over-Union:

$$\Delta_n(\tau) = \begin{cases} 1, & \exists\, d\in\mathcal D_\tau : \mathrm{IoU}(d,\hat{\mathcal M}_\tau^n) > \text{iou\_threshold} \\ -1, & \text{otherwise}, \end{cases} \tag{29}$$

$$S_n(t,t') = \sum_{\tau=t}^{t'} \Delta_n(\tau), \tag{30}$$

retaining a masklet as confirmed when $S_n \ge 0$. This yields, for each frame $i$ and confirmed track $n$, a binary mask $\mathcal M_{i,n}$, matching SAM 3's video-predictor output convention of `out_obj_ids`, `out_probs`, `out_boxes_xywh`, and `out_binary_masks`.

#### E.2 Spatial Alignment via Frozen Shared Geometry

Because Brown–Conrady undistortion (Eqs. 8–9) and affine resizing are applied exactly once, prior to the photometric fork (Section III-C), a pixel coordinate in SAM 3's output corresponds by construction to the same physical origin as the coordinate frame in which events are generated (Section III-D):

$$\begin{bmatrix} x_{\text{event}} \\ y_{\text{event}} \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} u_{\text{SAM}} \\ v_{\text{SAM}} \end{bmatrix}. \tag{31}$$

This identity mapping — stated formally rather than assumed — is the mechanism by which CAROECT-D avoids the targetless calibration procedure (and its reported 3.37–6.72 px reprojection error) required in TUMTraf Event [7]: no second physical camera viewpoint is ever introduced into the label-transfer path.

#### E.3 Temporal Synchronization

SAM 3 observations carry explicit frame times $t_k$. Synthetic RGB and events share one source clock, so their offset is zero by construction. Physical RGB/event pairs instead use a flash or blinking target to estimate a single offset under

$$t_{\text{event}} = t_{\text{RGB}} + \delta_t, \tag{32}$$

with method, residual, confidence, files, and units stored in sync.json. Physical label transfer refuses a silent zero offset unless an explicit unsynchronized diagnostic override is supplied.

#### E.4 Exact Observations and Causal Detector Windows

For an exact SAM 3 observation $B_k$ at time $t_k$, the detector sample contains only the half-open event history

$$\mathcal W_k(\Delta T)=\{e_j\mid t_k-\Delta T\le t_j<t_k\}, \qquad \mathcal W_k(\Delta T)\longrightarrow B_k. \tag{33}$$

$B_k$ is copied exactly; $B_{k+1}$ and events at or after $t_k$ cannot influence sample $k$. No per-event box/mask interpolation is used in the training path. The standard windows are $\Delta T\in\{8.34,16.7,33.3,50.0\}$ ms. All variants retain identical ordered $t_k$ and labels while changing only the included event history.

#### E.5 Bidirectional SAM 3 Merge

Independent forward and backward sessions are seeded at the first and last frames and both directional artifacts are retained. Same-class trajectories are associated by temporal overlap and mean IoU. The merge uses continuity first, confidence second, and only then a receding-trajectory tie-break derived from decreasing area and motion toward a configured horizon. Propagation direction itself is never treated as receding evidence.

#### E.6 Causal Label Transfer

For sorted event timestamps, two lower-bound searches give

$$a_k=\operatorname{lower\_bound}(t,t_k-\Delta T),\qquad b_k=\operatorname{lower\_bound}(t,t_k). \tag{34}$$

Events with indices $a_k,\ldots,b_k-1$ are rendered into the detector representation and paired with unchanged $B_k$. Shared synthetic geometry makes the spatial transform an identity; physical evaluation additionally records synchronization residual and RGB/event registration error. The resulting labels are frame-observation detector targets with auditable causality, not dense per-event interpolated labels.

---

## References

[1] H. Rebecq, D. Gehrig, and D. Scaramuzza, "ESIM: An open event camera simulator," in *Proc. Conf. Robot Learning (CoRL)*, 2018, pp. 969–982.

[2] D. Gehrig, M. Gehrig, J. Hidalgo-Carrió, and D. Scaramuzza, "Video to events: Recycling video datasets for event cameras," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 3586–3595.

[3] Y. Hu, S.-C. Liu, and T. Delbruck, "v2e: From video frames to realistic DVS events," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2021, pp. 1312–1321.

[4] S. Lin, Y. Ma, Z. Guo, and B. Wen, "DVS-Voltmeter: Stochastic process-based event simulator for dynamic vision sensors," in *Proc. European Conf. Computer Vision (ECCV)*, 2022, pp. 578–593.

[5] Z. Ning, E. Lin, S. R. Iyengar, and P. Vandewalle, "Raw2Event: Converting raw frame camera into event camera," *arXiv preprint arXiv:2509.06767*, 2025.

[6] A. A. Verma, B. Chakravarthi, A. Vaghela, H. Wei, and Y. Yang, "eTraM: Event-based traffic monitoring dataset," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 22637–22646.

[7] C. Creß, W. Zimmer, N. Purschke, B. N. Doan, S. Kirchner, V. Lakshminarasimhan, L. Strand, and A. C. Knoll, "TUMTraf Event: Calibration and fusion resulting in a dataset for roadside event-based and RGB cameras," *IEEE Trans. Intelligent Vehicles*, 2024.

[8] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick, "Segment Anything," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2023.

[9] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer, "SAM 2: Segment Anything in Images and Videos," *arXiv preprint arXiv:2408.00714*, 2024.

[10] N. Carion et al., "SAM 3: Segment Anything with Concepts," *arXiv preprint arXiv:2511.16719*, 2025.

---

### Notes for next revision

- **Formula fixes made in this pass:** rounded $\Delta L_{\text{dark}}$ to the precise value $\ln 2 \approx 0.693$ (was inconsistently written as both 0.69 and 0.693 in earlier drafts); flagged $\alpha=\Delta t/\tau$ (Eq. 12) as a first-order approximation and given the exact discretization $\alpha=1-e^{-\Delta t/\tau}$; explicitly labeled the SAM 3 presence-token formula (Eq. 28) and Masklet Detection Score (Eqs. 29–30) as **this paper's own formalization** of SAM 3's described mechanism, not equations reproduced from the SAM 3 paper — this avoids overclaiming what is publicly known about SAM 3's internals.
- **Not yet written (by request):** Experimental Setup and Results — no capture/benchmark data exists yet. Section headers can be added as `IV. Experimental Setup` and `V. Results` once data collection is complete; the outline document already sketches the intended scene/perspective/lighting splits and the eTraM domain-transfer protocol for that section.
- Equation numbering runs continuously through Section III (1)–(34); renumber if sections are reordered or Experiments/Results are inserted before Methodology.
