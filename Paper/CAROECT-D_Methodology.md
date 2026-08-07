# IV. METHODOLOGY — CAROECT-D

*(Draft methodology section, written to be pasted directly into the IEEE-format LaTeX
source. Display equations use standard LaTeX and are labeled by subsection, e.g.
(B.1), (B.2), so they can be renumbered globally once the paper is assembled. All
formulas below were checked for internal mathematical consistency; see the
**Verification & Correction Log** at the end of this file.)*

**Roadmap.** The pipeline is organized around one governing idea: every downstream
step — event generation, tracking, and automatic annotation — is only physically
meaningful if it operates on *scene-linear*, *geometrically frozen* data. Section B
establishes why the radiometric domain must be 16-bit linear rather than 8-bit
gamma-encoded. Section C establishes a single, shared geometric transform applied
once to that linear data, before the pipeline forks into an sRGB annotation branch
and a linear event-generation branch. Section D uses the corrected, undistorted
linear volume to generate events through two complementary models — a deterministic
one (v2e) and a stochastic one (Raw2Event / DVS-Voltmeter). Section E closes the loop:
because the two branches share identical geometry and a common frame clock, labels
produced by SAM 3 on the sRGB branch can be transferred to the asynchronous event
stream with a simple point-in-mask test, with no cross-sensor calibration.

## Notation Summary

| Symbol | Meaning |
|---|---|
| $I(x,y,t)$ | Scene irradiance (linear, photon-proportional) |
| $L(x,y,t)=\ln I(x,y,t)$ | Log-intensity |
| $\Delta L$ | Change in log-intensity since the last emitted event |
| $\theta_{ON},\theta_{OFF} > 0$ | Positive contrast thresholds |
| $N_e$ | Number of events fired in one update |
| $e_k = (x_k,y_k,t_k,p_k)$ | $k$-th generated event, polarity $p_k\in\{0,1\}$ |
| $(u,v)$ | Pixel coordinates in the undistorted, resized sRGB (SAM 3) branch |
| $(x,y)$ | Pixel coordinates in the undistorted, resized linear-luminance (event) branch |
| $K$ | Camera intrinsic matrix |
| $D=(k_1,k_2,k_3,p_1,p_2)$ | Brown–Conrady distortion coefficients |
| $\bar L,\ k_{dL}$ | Local mean brightness and its rate of change |
| $k_1,\dots,k_6$ | Six calibrated DVS-Voltmeter circuit parameters |
| $\mu,\sigma$ | Drift and diffusion of the pixel-voltage Brownian motion |
| $W(\Delta t)$ | Wiener process |
| $IG(m,\lambda)$ | Inverse Gaussian (Wald) distribution, mean $m$, shape $\lambda$ |
| $M_{i,n}$ | SAM 3 binary mask for object $n$ at frame $i$ |
| $i$ | Discrete frame index (119.88 fps clock) |
| $\Delta t_{frame}$ | $1/119.88~\text{fps}\approx 8.34~\text{ms}$ |

---

## A. Problem Setting and Notation

Traffic scenes are captured with two rigidly co-located, time-synchronized cameras:
a Nikon Z6 III recording 4K N-RAW video with a 20 mm lens, and a LUCID Vision Labs
Triton2 event camera (Sony IMX636 sensor, active area $6.2208\times3.4992$ mm) with a
6 mm lens. Both lenses' horizontal fields of view follow the standard pinhole relation

$$
\text{HFOV} = 2\arctan\!\left(\frac{w}{2f}\right), \tag{A.1}
$$

which reproduces the two operating points used throughout this paper: $84.0^{\circ}$
for the RGB rig ($w=36$ mm, $f=20$ mm) and $54.8^{\circ}$ for the event rig
($w=6.2208$ mm, $f=6$ mm). Because the event sensor's FOV is strictly a subset of the
RGB sensor's FOV, spatial alignment (Section C) is fundamentally a cropping and
registration problem rather than a stereo-baseline problem.

Raw video is processed at the event camera's native frame rate of 119.88 fps. The
pipeline branches into (i) an sRGB annotation branch that is tone-mapped for SAM 3,
and (ii) a 16-bit linear event-generation branch. The event branch ultimately yields
an asynchronous stream of tuples

$$
e_k = (x_k, y_k, t_k, p_k), \qquad p_k \in \{0,1\}, \tag{A.2}
$$

denoting pixel location, microsecond timestamp, and polarity.

---

## B. Radiometric Preprocessing and Photometric Linearization

### B.1 Weber's Law and the Log-Intensity Trigger

Physical event sensors respond to the temporal derivative of *log*-intensity
(Weber's Law):

$$
L(x,y,t) = \ln I(x,y,t). \tag{B.1}
$$

An ON event is emitted when the accumulated log-intensity change since the last
event exceeds a positive contrast threshold, and an OFF event when it falls below
the negative of a (possibly different) positive threshold:

$$
\Delta L = \ln I_t - \ln I_{t-1}
\quad\Rightarrow\quad
\begin{cases}
\Delta L \ge \theta_{ON} & \text{(ON event)}\\
\Delta L \le -\theta_{OFF} & \text{(OFF event)}
\end{cases} \tag{B.2}
$$

with $\theta_{ON},\theta_{OFF}>0$. *(Note: the working draft occasionally wrote the
OFF condition as $\Delta L\le\theta_{OFF}$ with an implicitly signed threshold; (B.2)
fixes this by keeping both thresholds positive magnitudes and making the sign
explicit — see the correction log.)*

### B.2 Failure Mode: 8-bit Gamma-Encoded Video

Consumer RGB pipelines output 8-bit frames after a gamma encoding
$I_{sRGB}\approx I^{1/\gamma}$, $\gamma\approx 2.2$. Since $\ln(I^{1/\gamma}) =
\tfrac1\gamma\ln I$, naively feeding this signal into an event simulator scales the
usable log-intensity range:

$$
L_{sRGB} = \ln\!\left(I^{1/\gamma}\right) = \frac{1}{\gamma}\ln I
\quad\Rightarrow\quad
\frac{\partial L_{sRGB}}{\partial t} = \frac{1}{\gamma}\,\frac{\partial L}{\partial t}. \tag{B.3}
$$

This compresses the true contrast derivative by a constant factor $1/\gamma$,
de-calibrating any fixed threshold $\theta$ against the sensor's native response.

A second, independent failure comes from **quantization**. With $I\in\{0,\dots,255\}$,
the smallest representable step near a code value $I$ produces a log-intensity jump
of approximately

$$
\Delta L_{step}(I) \;\approx\; \frac{d}{dI}\ln I \;=\; \frac{1}{I}, \tag{B.4}
$$

a first-order expansion of $\ln(I{+}1)-\ln I$. Equation (B.4) is not merely
qualitative — it predicts both extremes exactly:

- **Shadow "hallucination."** Near $I=1$ (darkest representable step above black),
$$
\Delta L_{dark} = \ln 2 - \ln 1 = \ln 2 \approx 0.693, \tag{B.5}
$$
a 100% relative jump that exceeds any realistic $\theta$ (typically $0.1$–$0.3$),
forcing the simulator to hallucinate a flood of spurious ON events from
imperceptible sensor noise.

- **Highlight "blindness."** Near saturation, e.g. $I:254\to255$,
$$
\Delta L_{bright} = \ln 255 - \ln 254 = \ln\!\frac{255}{254} \approx 0.00393, \tag{B.6}
$$
too small to ever cross $\theta$ — the simulator goes blind to real motion in bright
regions.

### B.3 The 16-bit Linear Solution

CAROECT-D bypasses the ISP entirely and exports 16-bit linear TIFFs directly from
N-RAW, so $I\in\{0,\dots,65535\}$, a $256\times$ larger quantization space than 8-bit.
Equation (B.4) shows exactly why this matters: for the *same absolute scene
darkness* that produced $I{=}1$ at 8-bit, the equivalent 16-bit code value is on the
order of $I\approx256$ (an 8-to-16-bit scale factor of $65536/256=256$), giving

$$
\Delta L_{step}(256) = \ln 257 - \ln 256 \approx 0.0039, \tag{B.7}
$$

i.e. the *shadow* regime at 16-bit now has the same fine log-intensity resolution
that only the *brightest* regime enjoyed at 8-bit (compare to (B.6)). Because the
data also bypasses the non-linear gamma curve, pixel values remain strictly
proportional to photon count, which is required for (B.3) to reduce to the identity
$L_{sRGB}=L$ ($\gamma=1$). Together, (B.4)–(B.7) are the mathematical justification
for the $>120$ dB dynamic range CAROECT-D targets, matching the physical Sony IMX636
sensor.

### B.4 Physical Corrections on Linear Data

Because the exported data remains scene-linear, standard radiometric calibration is
valid prior to any tone mapping:

$$
I_{corr} = (I_{raw} - I_{dark}) \cdot M_{gain} \cdot W_{gain} \cdot S_{exp}, \tag{B.8}
$$

where:

- $I_{dark}$ is the dark-frame bias (mean of lens-cap exposures), removing
sensor black level;
- $M_{gain}(x,y) = \dfrac{\overline{I_{flat}-I_{dark}}}{I_{flat}(x,y)-I_{dark}(x,y)}$
is the flat-field gain map — the mean corrected flat-frame level divided by its
per-pixel value — which normalizes vignetting and fixed-pattern sensitivity so a
uniformly lit scene yields a uniform image;
- $W_{gain}$ is a per-channel gray-card white-balance gain,
$W_{gain,c} = \bar g_{gray}/\bar c_{gray}$ for $c\in\{R,G,B\}$, normalizing a neutral
patch to equal channel response;
- $S_{exp}$ is a scalar (or per-channel gain) normalizing residual exposure/ISO
differences across capture sessions to a common reference.

Executing (B.8) **before** the pipeline forks guarantees that static spatial
gradients (e.g. vignette corners) do not masquerade as temporal log-intensity
change and falsely trigger events under minor camera vibration.

---

## C. Geometric Lens Calibration and Shared Undistortion

### C.1 Ideal Pinhole Projection

$$
\begin{bmatrix} u\\ v\\ 1 \end{bmatrix}
=
\begin{bmatrix} f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix} X_c\\ Y_c\\ Z_c \end{bmatrix}
= K \begin{bmatrix} X_c\\ Y_c\\ Z_c \end{bmatrix}, \tag{C.1}
$$

where $f_x,f_y$ are the focal lengths in pixels and $(c_x,c_y)$ the principal point.

### C.2 Brown–Conrady Distortion Model

Real lenses violate (C.1) through radial and tangential distortion, warping ideal
coordinates $(x,y)$ into observed coordinates $(x_{dist},y_{dist})$:

$$
\begin{aligned}
x_{dist} &= x\left(1+k_1 r^2+k_2 r^4+k_3 r^6\right) + \left[2p_1 xy + p_2\left(r^2+2x^2\right)\right] \\
y_{dist} &= y\left(1+k_1 r^2+k_2 r^4+k_3 r^6\right) + \left[p_1\left(r^2+2y^2\right) + 2p_2 xy\right]
\end{aligned} \tag{C.2}
$$

with $r=\sqrt{x^2+y^2}$, $k_{1,2,3}$ the radial coefficients and $p_{1,2}$ the
tangential coefficients, all calibrated via checkerboard against the intrinsic
matrix $K$ from (C.1). This is the standard (OpenCV) formulation and was verified
term-by-term against the working draft with no corrections required.

### C.3 Shared-by-Construction Geometry

The undistortion in (C.2), followed by an affine resize to the event sensor's
native array size,

$$
\begin{bmatrix} x'\\ y' \end{bmatrix}
=
\begin{bmatrix} 1280/W_0 & 0\\ 0 & 720/H_0 \end{bmatrix}
\begin{bmatrix} x\\ y \end{bmatrix}, \tag{C.3}
$$

is computed **exactly once** on the shared 16-bit linear array $I_{corr}$, *before*
the pipeline forks into (i) an sRGB-encoded branch for SAM 3 and (ii) a linear
luminance branch for event simulation. Because both branches are derived from the
identical post-(C.2)–(C.3) pixel grid at the same $1280\times720$ resolution, the
spatial mapping between the two domains is, by construction, the identity:

$$
\begin{bmatrix} x_{event}\\ y_{event} \end{bmatrix}
=
\begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix}
\begin{bmatrix} u_{SAM}\\ v_{SAM} \end{bmatrix}. \tag{C.4}
$$

This is the geometric precondition that Section E exploits to avoid any targetless,
error-prone cross-sensor calibration.

---

## D. High-Fidelity Event Generation from Linear Radiance

Both event-generation models below consume the *same* upstream artifact: the
corrected, undistorted, resized linear volume $I_{corr}$ from Sections B and C. Only
the temporal event-generation logic differs between D.2 and D.3.

### D.1 Temporal Sampling Rate and Finite-Difference Fidelity

Both models approximate the continuous derivative $\partial L/\partial t$ from
discrete frames via a forward difference,

$$
\left.\frac{\partial L}{\partial t}\right|_{t=t_i} \approx
\frac{L(x,y,t_{i+1}) - L(x,y,t_i)}{\Delta t_{frame}}, \tag{D.1}
$$

whose truncation error is $O(\Delta t_{frame})$. Standard video datasets operate at
30 fps ($\Delta t_{frame}\approx33.3$ ms); CAROECT-D instead records at the event
camera's native 119.88 fps ($\Delta t_{frame}\approx8.34$ ms), a $\sim4\times$
reduction in the sampling interval. A smaller $\Delta t_{frame}$ both tightens the
approximation error in (D.1) and reduces the inter-frame spatial displacement
$\Delta x = v\cdot\Delta t_{frame}$ of moving objects, mitigating the optical-flow
breakdown that produces smeared, "ghosted" event clouds under coarse frame rates.

### D.2 The v2e Deterministic Pipeline

**Step 1 — Luma conversion.** The linear RGB volume is reduced to a single-channel
luma signal $Y$:

$$
Y = c_R R + c_G G + c_B B. \tag{D.2}
$$

*Note:* the reference v2e implementation defaults to BT.601/BT.709 weights; because
CAROECT-D operates on wide-gamut, 16-bit linear N-RAW data rather than tone-mapped
video, this pipeline instead adopts ITU-R BT.2020 weights
$(c_R,c_G,c_B)=(0.2627, 0.6780, 0.0593)$, which sum to unity and are consistent with
the sensor's native color primaries. Frame upsampling via Super-SloMo (v2e's optional
interpolation step) is disabled, since native capture already occurs at 119.88 fps —
enabling it would risk hallucinated motion detail.

**Step 2 — Hybrid lin-log mapping.** A pure logarithm diverges as $Y\to0$, so v2e
uses a piecewise mapping with transition point $I_{lin}$ (typically 20 DN):

$$
L_{in} =
\begin{cases}
\ln Y, & Y \ge I_{lin}\\[4pt]
\dfrac{Y}{I_{lin}}\ln I_{lin}, & Y < I_{lin}
\end{cases} \tag{D.3}
$$

Both branches agree at $Y=I_{lin}$ (value continuity: $\ln I_{lin}=\tfrac{I_{lin}}{I_{lin}}\ln I_{lin}$),
so (D.3) is continuous, and the linear branch avoids the numerical blow-up and
quantization noise of $\ln(0^+)$.

**Step 3 — Intensity-dependent low-pass filter.** Physical DVS pixels have finite
analog bandwidth that grows with incident light, producing motion blur in low light.
This is modeled as a discretized first-order RC low-pass:

$$
L_{lp}[k] = L_{lp}[k-1] + \alpha\left(L_{in}[k]-L_{lp}[k-1]\right),
\qquad \alpha=\frac{\Delta t}{\tau},\quad \tau=\frac{1}{2\pi f_c(Y)}, \tag{D.4}
$$

with cutoff frequency $f_c(Y)$ monotonically increasing in $Y$ (bright pixels track
$L_{in}$ almost instantly, $\alpha\approx1$; dark pixels lag, $\alpha\to0$).

**Step 4 — Event generation and memory update.**

$$
\Delta L = L_{lp}-L_{mem},\qquad
N_e = \left\lfloor \frac{|\Delta L|}{\theta} \right\rfloor,\qquad
L_{mem} \leftarrow L_{mem} + N_e\,\theta\,\mathrm{sign}(\Delta L), \tag{D.5}
$$

fired whenever $\Delta L\ge\theta_{ON}$ or $\Delta L\le-\theta_{OFF}$ per (B.2), with
a frozen, per-pixel Gaussian threshold mismatch
$\theta_{pixel}\sim\mathcal N(\theta_{nominal},\sigma_\theta^2)$ modeling silicon
manufacturing tolerance.

**Step 5 — Non-idealities.** Leak events model charge leakage as a continuous drift
of the memorized state,

$$
L_{mem}(t) = L_{mem}(t_0) - R_{leak}\cdot(t-t_0), \tag{D.6}
$$

and shot noise is modeled as a Bernoulli draw per time step with rate increasing as
$Y$ decreases,

$$
p = R_n(Y)\cdot\Delta t,\qquad u\sim\mathcal U(0,1),\ \text{fire if } u<p \text{ or } u>1-p. \tag{D.7}
$$

**Step 6 — Deterministic timestamping.** Given $N_e$ events between frames at $t_j$
and $t_{j+1}$, v2e distributes them evenly:

$$
t_k = t_j + k\cdot\frac{t_{j+1}-t_j}{N_e},\qquad k\in\{1,\dots,N_e\}. \tag{D.8}
$$

Because (D.8) locks every timestamp to a rigid, frame-rate-derived grid, the
resulting event stream exhibits **temporal layering**: events cluster at artificial,
evenly spaced positions rather than the continuous microsecond jitter of a physical
sensor. This motivates Section D.3.

### D.3 The Raw2Event / DVS-Voltmeter Stochastic Pipeline

Raw2Event couples the DVS-Voltmeter stochastic voltage model to the same raw
linear ingestion established in Section B — i.e. it also operates directly on
$I_{corr}$, bypassing the ISP, rather than on tone-mapped video frames as in the
generic DVS-Voltmeter formulation.

**Voltage as Brownian motion with drift.** Instead of a deterministic per-frame
difference, the internal pixel voltage change over a time step $\Delta t$ is modeled
as a Wiener process with drift:

$$
\Delta V_d = \mu\,\Delta t + \sigma\,W(\Delta t). \tag{D.9}
$$

Drift and diffusion are parametrized by six physically interpretable, calibrated
constants:

$$
\mu = \frac{k_1}{\bar L + k_2}\,k_{dL} + k_4 + k_5\bar L,
\qquad
\sigma = \frac{k_3}{\bar L + k_2}\sqrt{\bar L} + k_6, \tag{D.10}
$$

where $\bar L$ is local mean brightness, $k_{dL}=d\bar L/dt$ its rate of change,
$k_1$ signal gain, $k_2$ a darkness-limiting offset, $k_4,k_5$ thermal/parasitic
leakage drift, $k_3$ photon-shot-noise-driven jitter, and $k_6$ the baseline
electronic noise floor.

**First-passage-time sampling.** Rather than interpolating timestamps as in (D.8),
the timestamp of the *next* event is sampled as the first time the drifted Brownian
motion (D.9) hits an absorbing barrier at $+\theta_{ON}$ or $-\theta_{OFF}$. This is
a classical result for Brownian motion with drift: the first hitting time of a level
$a>0$ by a process with drift $\mu>0$ and diffusion $\sigma$ follows an Inverse
Gaussian (Wald) distribution with mean $a/\mu$ and shape $a^2/\sigma^2$. Applying
this symmetrically to both barriers (with $\mu$'s sign determining which barrier is
approached),

$$
\tau_{ON} \sim IG\!\left(\frac{\theta_{ON}}{\mu},\ \frac{\theta_{ON}^2}{\sigma^2}\right)
\ (\mu>0),
\qquad
\tau_{OFF} \sim IG\!\left(\frac{\theta_{OFF}}{|\mu|},\ \frac{\theta_{OFF}^2}{\sigma^2}\right)
\ (\mu<0), \tag{D.11}
$$

reducing to a Lévy distribution in the driftless case $\mu=0$. *(This restates the
source draft's $IG(\mp\Theta/\mu,\ \Theta^2/\sigma^2)$ notation with both thresholds
kept as positive magnitudes and the sign resolved by which barrier is drifted
toward, removing the ambiguity of the original $\mp$ shorthand — see the correction
log.)*

Because timestamps in (D.11) are continuous random variables rather than points on a
fixed frame-derived grid, this model produces microsecond jitter consistent with a
physical DVS circuit and removes the temporal layering artifact of (D.8) by
construction, at the cost of requiring six calibrated circuit parameters instead of
one contrast threshold.

---

## E. Cross-Modal Spatiotemporal Label Transfer

### E.1 SAM 3 as the Label Source $^{\dagger}$

*($^{\dagger}$ Formulas in this subsection are reconstructed from general knowledge
of SAM 3's publicly described design — decoupled recognition/localization via a
global presence token, and a memory-based video tracker in the spirit of SAM 2 —
**not transcribed from the primary source**, since this session has no web/search
access. Verify notation, exact terms, and citation details against the official SAM
3 paper before submission.)*

SAM 3 performs Promptable Concept Segmentation (PCS): given a short noun phrase
(e.g. "car", "pedestrian"), it detects, segments, and tracks every instance of that
concept in a video. A central design choice is decoupling whether the concept is
*present at all* from *where* it is located, via a learned global presence token:

$$
p(\text{query}_i \text{ matches NP}) =
p(\text{query}_i \text{ matches NP} \mid \text{NP present in image}) \cdot
p(\text{NP present in image}). \tag{E.1}
$$

(This is a standard conditional-probability decomposition; the modeling
contribution is that the second factor is predicted once, globally, per frame,
rather than being entangled with each local query's spatial localization
objective — reducing false positives when the concept is genuinely absent.)

For video, SAM 3 propagates spatio-temporal masklets $\hat{\mathcal M}_\tau^n$ for
object $n$ at frame $\tau$, using a memory bank, and periodically re-confirms tracks
against a per-frame detector output $\mathcal D_\tau$ to recover from occlusion. A
plausible confirmation rule (approximate; verify against source) is an IoU-gated
indicator accumulated over a tracking window $[t,t']$:

$$
\Delta_n(\tau) =
\begin{cases}
+1, & \exists\, d\in\mathcal D_\tau \text{ s.t. } \mathrm{IoU}(d,\hat{\mathcal M}_\tau^n) > \text{iou\_threshold}\\
-1, & \text{otherwise}
\end{cases},
\qquad
S_n(t,t') = \sum_{\tau=t}^{t'} \Delta_n(\tau), \tag{E.2}
$$

retaining a masklet as confirmed when $S_n \ge 0$ and outputting a binary mask
$M_{i,n}$ (with corresponding bounding box and tracking ID) for object $n$ at
discrete frame $i$.

### E.2 Spatial Alignment via the Frozen Shared Geometry

By Section C.3, a SAM 3 pixel coordinate $(u,v)$ in the sRGB branch and the physical
event coordinate $(x,y)$ refer to the same undistorted, resized grid — the identity
map of (C.4). This differs from prior cross-modal datasets such as TUMTraf Event,
where a physical baseline between two separately positioned sensors necessitates
targetless extrinsic calibration (e.g., motion-edge extraction, DBSCAN clustering,
Iterative Closest Point matching), reported to introduce reprojection error on the
order of a few pixels *(exact figure to be re-verified against the TUMTraf Event
paper before citing)*. CAROECT-D avoids this class of error entirely by
construction rather than by calibration accuracy.

### E.3 Temporal Synchronization

SAM 3 produces discrete, per-frame annotations, while both event models of Section D
produce continuous timestamps $t_k$. Both, however, are driven from the same
underlying 119.88 fps frame clock:

$$
t_i = i \cdot \Delta t_{frame}, \qquad \Delta t_{frame} \approx 8.34~\text{ms}. \tag{E.3}
$$

Because both v2e (D.8) and Raw2Event/DVS-Voltmeter (D.11) generate events strictly
from the log-intensity transition between frame $i$ and frame $i{+}1$, every event
resulting from that transition satisfies

$$
t_k \in [t_i, t_{i+1}). \tag{E.4}
$$

### E.4 The Label-Assignment Operator

Combining the spatial identity of (C.4)/(E.2) with the temporal window of (E.4)
yields a single spatiotemporal inclusion test. An event $e_k=(x_k,y_k,t_k,p_k)$ is
assigned object identity $ID_n$ if and only if it falls within the corresponding
frame's confirmed SAM 3 mask, and inside the correct temporal window:

$$
\mathrm{Assign}(e_k) =
\begin{cases}
ID_n, & t_k \in [t_i, t_{i+1}) \ \wedge\ (x_k, y_k) \in M_{i,n} \\[4pt]
\text{Background}, & \text{otherwise}
\end{cases} \tag{E.5}
$$

Because the spatial term reduces to a point-in-mask test (no reprojection) and the
temporal term reduces to a shared-clock interval lookup (no cross-modal
resynchronization), (E.5) can be evaluated directly on every generated event,
yielding dense, pixel-accurate class labels, bounding boxes, and consistent tracking
IDs for the synthetic event stream without manual annotation.

---
---

# INTERNAL NOTES — remove before submission

## Verification & Correction Log

1. **(B.2)** Standardized $\theta_{ON},\theta_{OFF}$ as positive magnitudes and
   corrected the OFF-event condition to $\Delta L \le -\theta_{OFF}$. The source
   drafts were inconsistent, sometimes writing $\Delta L\le\theta_{OFF}$ as if
   $\theta_{OFF}$ were already signed.
2. **(B.5)** Verified $\ln 2-\ln 1=\ln 2\approx 0.6931$; standardized rounding to
   0.693 (source alternated between 0.69 and 0.693 across drafts).
3. **(B.6)** Verified $\ln 255-\ln 254=\ln(255/254)\approx0.003930$ (source's
   $\approx0.0039$ confirmed correct).
4. **(B.4), (B.7)** Added the general first-order quantization bound
   $\Delta L_{step}(I)\approx1/I$, not present in the source material, derived here
   from $d(\ln I)/dI$, to unify the two worked numeric examples into one argument
   and to show explicitly why 16-bit shadow regions inherit the fine resolution
   that only 8-bit highlights previously had.
5. **(A.1)** Verified the horizontal-FOV formula reproduces both tabulated values in
   the project outline: $84.0^{\circ}$ (Nikon Z6III, 20 mm, 36 mm sensor width) and
   $54.8^{\circ}$ (Triton2/IMX636, 6 mm, 6.2208 mm sensor width).
6. **(D.2)** Verified the Rec.2020 luma weights sum to unity
   ($0.2627+0.6780+0.0593=1.0000$); flagged explicitly as a deliberate CAROECT-D
   deviation from vanilla v2e's default BT.601/709 weights, with justification
   added to the prose (wide-gamut 16-bit linear input rather than tone-mapped video).
7. **(C.2)** Brown–Conrady equations checked term-by-term against the standard
   (OpenCV) formulation — correct as given in the source, no changes made.
8. **(C.1)** Pinhole intrinsic matrix checked — correct as given, no changes made.
9. **(D.3)** Verified value-continuity of the hybrid lin-log mapping at $Y=I_{lin}$.
10. **(D.11)** Rewrote the Inverse Gaussian first-passage-time formula using two
    magnitude-positive thresholds and an explicit sign convention on $\mu$, instead
    of the source's ambiguous $IG(\mp\Theta_{ON/OFF}/\mu, \dots)$ shorthand.
11. **(B.8)** Made $M_{gain}$ and $W_{gain}$ mathematically explicit (flat-field
    ratio and gray-card channel ratio, respectively); the source only asserted
    these as unspecified "maps"/"multipliers."
12. **(C.3)/(C.4)** Added the explicit resize matrix (C.3) and grounded the identity
    claim (C.4) in the fact — stated in the project outline, not previously used to
    justify this step — that both branches are resampled to the same 1280×720
    resolution matching the event sensor's native array, which is *why* the identity
    map holds rather than simply asserting it.
13. **(E.1), (E.2)** SAM 3 formulas: (E.1) is a generically true conditional-probability
    identity, safe regardless of exact SAM 3 internals. (E.2), the masklet
    confirmation score, is a plausible reconstruction consistent with public
    descriptions of SAM 3's decoupled-recognition design and SAM 2-style memory
    tracking, but **is not verified against the primary source** — no web/search
    access was available this session. Please confirm exact notation/mechanism
    before using in a submission.
14. **Citations.** Bibliographic details for v2e, DVS-Voltmeter, Raw2Event, TUMTraf
    Event, eTraM, and SAM 3 (authors/venue/year) were not independently confirmed
    this session for the same reason. Suggested best-recollection references, to be
    verified once Related Work is written:
    - v2e: Hu, Liu, Delbruck, *"v2e: From Video Frames to Realistic DVS Events,"* CVPR Workshops, 2021.
    - DVS-Voltmeter: Lin et al., *"DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors,"* ECCV, 2022.
    - ESIM: Rebecq, Gehrig, Scaramuzza, *"ESIM: an Open Event Camera Simulator,"* CoRL, 2018.
    - vid2e: Gehrig et al., *"Video to Events: Recycling Video Datasets for Event Cameras,"* CVPR, 2020.
    - Raw2Event, TUMTraf Event, eTraM, SAM 3: titles/venues/years to be confirmed — I do not have confident, independently-checked recall of these citations.
15. **TUMTraf Event reprojection error** (E.2): the "3.37–6.72 px" figure appeared in
    earlier drafts; kept out of the main text pending verification and flagged
    explicitly as unverified rather than restated as fact.

## Open items before this is submission-ready
- Confirm SAM 3 equations/terminology against the primary source.
- Confirm all citations above.
- Decide whether $S_{exp}$'s calibration procedure should be spelled out here or
  deferred to Experiments (currently deferred).
- Section D currently omits Event Representation (Histogram of Events / Time
  Surfaces) since that is training-input machinery for Experiments, not part of
  event *generation* — confirm this scoping matches your intended structure before
  merging with Section V.
