# Synthetic streaming SL benchmarks for StreamSLST

Synthesize streaming sign-language benchmarks for **PHOENIX-2014T** (DGS, German), **CSL-Daily** (CSL, Chinese), and **How2Sign** (ASL, English) from offline sentence-level pose clips, all in BOBSL-compatible layout.

**One synthesis pathway for all three datasets.** Each stream is a same-signer concatenation of _K_ sentence clips joined by Hermite C¹ bridges, whose gap durations are sampled from the empirical BOBSL inter-subtitle distribution, with exact frame-level oracle boundaries by construction. The datasets differ only in their source layout and a few per-dataset constants:

| Dataset       | Source layout                         | Signer unit  | Src → proc fps | Canvas    | Lang    |
| ------------- | ------------------------------------- | ------------ | -------------- | --------- | ------- |
| PHOENIX-2014T | one big pickle per split              | broadcast    | 25 → 12.5      | 210 × 260 | `de_DE` |
| CSL-Daily     | per-clip `.pkl` (+ `start`/`end`)     | `P<id>`      | 30 → 15        | 512 × 512 | `zh_CN` |
| How2Sign      | per-clip `.pkl` (Uni-Sign RTMPose)    | YouTube ID   | 30 → 15        | 1280 × 720 | `en_XX` |

Pick a benchmark via `DATASET={PHOENIX,CSL,H2S}`; `config.py` resolves all paths, fps, canvas, language code, and the trimmed mBART for that dataset. (An earlier CSV-timeline How2Sign path was removed — H2S now uses this same pathway, because the >10 s instructional dead-time in real H2S timing produced too many empty training windows.)

## Why synthetic streams

BOBSL's **auto-aligned** subtitles introduce annotation noise. Synthetic streams give us:

- three more sign languages (DGS, CSL, ASL) covering cross-language generalization;
- **oracle event boundaries** by construction (we _know_ exactly when each sentence starts and ends), letting us re-run the alignment / learned-vs-GT ablations to disentangle "model can't localize" from "BOBSL labels are noisy";

## Drop-in compatibility

Outputs match BOBSL's `loader.DVCDataset` directory contract:

```
data/synth/<lang>/
├── poses/<stream_id>.npy           # (T, 133, 3) float32 at the processing fps, native pixel coords
├── vtt/<stream_id>.vtt             # WEBVTT, one sentence per cue (single-line text)
├── subset2episode.json             # {"train": [...], "val": [...], "test": [...]}
└── manifest.json                   # provenance: clip ids per stream + seed + pause/k_range used
```

`DVCDataset._build_video_metadata` supports both pose layouts: BOBSL's multi-segment `POSE_ROOT/<video_id>/*.npy` and the synth flat `POSE_ROOT/<stream_id>.npy`. All datasets share `facebook/mbart-large-cc25` (its 25-language vocabulary covers en/de/zh natively), trimmed per language.

## Source layouts

The synthesizer reads two input layouts, auto-selected from `config.SYNTH_META`:

- **Big pickle** (PHOENIX): one `{.train,.dev,.test}` pickle per split, each a dict `{name: {'keypoint': (T,133,3), 'text': str, 'num_frames': int}}` in native pixel coords.
- **Per-sample pickle** (CSL, H2S): `poses/<name>.pkl` with `{'keypoints': (T,1,133,2) normalized [0,1], 'scores': (T,1,133)}` plus a gzipped `labels.<split>` dict. **CSL** pickles additionally carry `start`/`end` frame indices marking the actual signing segment — the loader slices `[start:end]` automatically, so the clip is already tight to signing. **H2S** (Uni-Sign RTMPose) has no `dev` split; use `--val_frac` to carve one at synthesis time.

---

# 1. Problem statement

Each input dataset is a pickle that maps a clip identifier to one isolated, sentence-aligned record:

```
clip_record  := {
  keypoint   : float32 (T, 133, 3)     # COCO-WholeBody-133, NATIVE pixel coordinates
  text       : str                     # German subtitle (PHOENIX) | Chinese subtitle (CSL)
  num_frames : int                     # = T at the source frame rate
  name       : str                     # canonical clip identifier (encodes signer ID)
}
```

The downstream streaming model expects an **untrimmed** pose stream that contains multiple sentences interleaved with non-signing background. Naive concatenation of these isolated clips introduces several artifacts that an event-localization head learns to exploit instead of learning real sign-content boundaries:

| Artifact                                    | Cause                                                                      | Trivial cue model can latch onto        |
| ------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------- |
| Hands-down → pause → hands-up at every join | Each clip recorded in isolation: signer rests before/after every utterance | Boundary = neutral-pose detector        |
| Long pause between every pair of clips      | Naive synth inserts a sampled pause between every clip                     | Pause length = localization signal      |
| Frozen BG_pre / BG_post                     | Held first/last frame as background                                        | BG = "no motion" Boolean                |
| Discontinuous velocity at the seam          | Last frame of A and first of B chosen independently                        | C0 discontinuity in keypoint trajectory |

The algorithm below removes all 4 artifacts. Each rule is **either pure geometry or sampled directly from BOBSL manual annotations**.

---

# 2. Algorithm overview

A synthetic stream is a temporal sequence assembled from $K$ sentence-level pose clips drawn from a single signer. The intuition behind every design choice is "_reproduce the joint distribution of co-articulation, pause length, and stream length that real continuous broadcast signing exhibits, while preserving oracle event boundaries that the offline corpus already gives us for free_".

Concretely, a stream $S$ has the schematic structure

```
S  =  [ B_pre ][ c_1 ][ Δ_1 ][ c_2 ][ Δ_2 ] ... [ c_K ][ B_post ]
       \_____/  \___/  \___/  \___/  \___/       \___/  \______/
         BG      clip  bridge  clip  bridge       clip     BG
```

where each $c_k$ is a real signer-pose clip from the offline corpus (after rest-trimming, §4.4) and each connector $\Delta_k, B_\text{pre}, B_\text{post}$ is a Hermite-interpolated bridge whose duration is sampled from the empirical BOBSL inter-subtitle gap distribution. Because ~74% of that distribution's mass is at zero, most seams are **co-articulated**: the bridge there is a short movement-epenthesis transition that is *signing content*, absorbed into the two adjacent sentences so their oracle boundaries touch ($e_k = s_{k+1}$; §4.1). Only the ~26% of seams with a positive sampled gap leave a non-transcribed background pause between sentences.

---

# 3. Notation

| Symbol                                         | Definition                                                                     | Source / value                                               |
| ---------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| $C$                                            | Pose clip pool, $C = \\{c_i = (P_i, t_i, s_i, n_i)\\}$                         | Input pickle                                                 |
| $P_i \in \mathbb{R}^{T_i \times 133 \times 3}$ | Keypoint tensor for clip $i$ — $T_i$ frames, 133 joints, $(x, y, c)$ per joint | clip_record["keypoint"]                                      |
| $t_i$                                          | Subtitle text for clip $i$                                                     | clip_record["text"]                                          |
| $s_i$                                          | Signer ID for clip $i$                                                         | parsed from clip_record["name"] (§4.2)                       |
| $\mathcal{P}_s$                                | Same-signer clip subset, $\mathcal{P}_s = \\{c_i \in C : s_i = s\\}$           | derived                                                      |
| $f$                                            | Target frame rate                                                              | $12.5$ Hz (matches StreamSLST `config.FPS`)                  |
| $W$                                            | Model training window length                                                   | $15$ s (matches StreamSLST `config.WINDOW_DURATION_SECONDS`) |
| $W_\text{stream}$                              | Stream design target window                                                    | $4 W = 60$ s (see §4.9)                                      |
| $G$                                            | BOBSL empirical inter-subtitle gap array, $g=\max(0,\cdot)$ (Eq 10)            | `bobsl_gap_samples.npy` (≈74% zero)                         |
| $G_+ \subset G$                                | Positive-only subset of $G$                                                    | `bobsl_gap_samples.npy`, $G_i>0$                             |
| $K$                                            | Number of sentence clips per stream                                            | sampled from $[K_\text{lo}, K_\text{hi}]$                    |
| $K_\text{lo}, K_\text{hi}$                     | $p_{10}, p_{90}$ of `subs_per_60s_window`(BOBSL)                               | from `bobsl_gap_stats.json`                                  |
| $n_\text{min}$                                 | Numerical floor for the Hermite bridge                                         | $2$ frames (§4.5)                                            |
| $\rho_\text{min}$                              | Minimum signer-pool size (need ≥1 phantom + ≥1 chosen)                         | $2$                                                          |

---

# 4. Algorithm components

This section walks through every step of the synthesis pipeline. Each subsection states **why** the rule exists, gives a **mathematical** definition, and shows the **code** implementation. The complete end-to-end algorithm is assembled in §5.

## 4.1 Stream skeleton

A stream is the temporal concatenation of $2K + 1$ segments:

$$S \ =\  B_\text{pre} \ \Vert\  c_1 \ \Vert\  \Delta_1 \ \Vert\  c_2 \ \Vert\  \Delta_2 \ \Vert\  \cdots \ \Vert\  c_K \ \Vert\  B_\text{post}$$

where $\Vert$ denotes frame-axis concatenation. The **clip segments** $c_k$ contribute the actual subtitle events; the **background segments** $B_\text{pre}, B_\text{post}$ are non-event. A **bridge** $\Delta_k$ is attributed depending on the gap it was sampled for (see below).

**Oracle cue attribution (co-articulation vs. pause).** A seam is _co-articulated_ when its sampled BOBSL gap is exactly $0$ — ~74% of seams (§4.6). In real continuous signing a sentence can start immediately after another: there is no non-signing frame between them, only movement epenthesis. So a co-articulated $\Delta_k$ is **signing content**, not background, and is split at its midpoint and absorbed into the two adjacent sentences, making their oracle boundaries **frame-adjacent** ($e_k = s_{k+1}$). Only a _pause_ seam (sampled gap $> 0$) keeps $\Delta_k$ as a non-transcribed background gap. Writing $a_k = \lfloor |\Delta_k|/2 \rfloor,\ b_k = \lceil |\Delta_k|/2 \rceil$ for co-articulated seams (and $a_k = b_k = 0$ for pause seams):

$$s_k = \frac{1}{f}\Big(\lvert B_\text{pre}\rvert + \sum_{j < k}(\lvert c_j\rvert + \lvert \Delta_j\rvert) - b_{k-1}\Big), \qquad e_k = \frac{1}{f}\Big(\lvert B_\text{pre}\rvert + \sum_{j < k}(\lvert c_j\rvert + \lvert \Delta_j\rvert) + \lvert c_k\rvert + a_k\Big)$$

Because $A$ is computed from segment lengths chosen by the synthesizer, the boundaries are **oracle** — exact to the frame.

> **Fixed bug.** An earlier version set $e_k = s_k + |c_k|/f$ unconditionally, leaving _every_ $\Delta_k$ — including the ~74% zero-gap ones — as a non-transcribed sliver, so no two sentences were ever frame-adjacent. That contradicted the co-articulation goal below and handed the localization head a trivial "there is always a small gap between sentences" cue absent from real streams (real BOBSL has ~74% touching subtitle boundaries). A physical $\ge 2$-frame Hermite bridge is still inserted at every seam for $C^1$ continuity (§4.5) — a hard concat would teleport, an even more trivial cue — but co-articulated bridges are now _labelled as signing_, not as a gap.

## 4.2 Same-signer clip pool

Before any stream is synthesized, the clip pool is partitioned by signer:

$$\mathcal{P}_s = \\{c_i \in C : s_i = s\\}, \quad \text{drop pools with } \lvert \mathcal{P}_s \rvert < \rho_\text{min}$$

A stream draws all of its $K$ clips from a single $\mathcal{P}_s$. Cross-signer concatenation is avoided because it introduces two leakage hazards:

1. **Skeleton-scale jumps** at clip boundaries (different signers have different limb lengths and camera-relative scales) → free signal that the localization head can use as a boundary detector ("learn the seam, not the sign").
2. **Signer-identity change** becomes a confound for the boundary cue.

The signer ID is parsed deterministically from each clip's `name` field (no external metadata, no pseudo mapping):

- **PHOENIX**: name is `<split>/<broadcast>-<sentence_idx>` (e.g. `train/11August_2010_Wednesday_tagesschau-1`). One broadcast = one signer; group by the broadcast.
- **CSL**: name is `Sxxx_P<id>_Tyy`; group by the `P<id>` token.
- **H2S**: name is `<YouTubeID>_<sentence>-<take>-rgb_<view>.mp4`; sentences from the same source video share a signer + setup, so group by the YouTube ID.

```python
def signer_id(dataset, clip_name):
    if dataset == 'PHOENIX': return re.sub(r'-\d+$', '', clip_name.split('/', 1)[-1])
    if dataset == 'CSL':     return f"P{re.search(r'_P(\\d+)_', clip_name).group(1)}"
    if dataset == 'H2S':     return re.match(r'^(.+?)_\d+-\d+-rgb_(?:front|side)', clip_name.rsplit('/',1)[-1]).group(1)
```

## 4.3 Frame-rate resampling

Source clips arrive at the source dataset's native frame rate (25 Hz PHOENIX; 30 Hz CSL/H2S). They are resampled to the per-dataset processing rate $f$ (12.5 Hz PHOENIX; 15 Hz CSL/H2S) by per-joint linear interpolation on $(x, y)$ and true nearest-neighbour on the confidence channel:

$$P^{(f)}_t \ =\  \text{interp}(P^{(f_\text{src})}, t / f, \  t \in 0 .. \lfloor f \cdot T_\text{src} / f_\text{src} \rfloor)$$

Confidence is taken nearest-neighbour rather than averaged because averaging confidences is semantically meaningless (a 50%-confident average of two predictions does not mean "half-confident at the average position").

## 4.4 `trim_rest` — removal of held preparation / retraction

Each isolated clip begins with a _preparation_ phase (signer raises hands from lap to first sign location) and ends with a _retraction_ phase (hands fall back to lap). These artifacts are absent in real continuous broadcast signing — signers move directly between adjacent sentences without returning to rest. Concatenating non-trimmed clips creates an obvious "hands-down → long pause → hands-up" boundary that the localization head can latch onto trivially.

A rest pose is a **held low pose**: the hands are below the shoulder line AND essentially still. A frame is "rest" iff both conditions hold:

$$\text{rest}(t) \ =\  \mathbb{1}\ \Big[\ \underbrace{\tfrac{1}{|W|}\!\sum_{j \in W} P^{j,y}_t \ >\  \tfrac{1}{|S|}\!\sum_{j \in S} P^{j,y}_t}_{\text{hands below shoulders}}\ \Big]\ \cdot\ \mathbb{1}\ \Big[\ \underbrace{\tfrac{\lVert \bar w_t - \bar w_{t-1}\rVert}{\text{shoulder width}} \ <\  \rho}_{\text{near-still}}\ \Big]$$

where $W = {9, 10}$ (wrists), $S = {5, 6}$ (shoulders), $\bar w_t$ is the mean wrist position, and $\rho$ (`rest_speed`, default $0.015$) is the still-motion threshold in shoulder-widths per frame. The rule is **signer-relative** (both the position and motion terms are normalized by the signer's own shoulders), so it auto-scales across canvases with no per-dataset threshold.

> **Fixed bug.** An earlier rule used the **position term only** (`hands below shoulders`). But signers — CSL-Daily for example — produce most of their signing _below_ shoulder level (measured: 87–97% of frames are below-shoulder and moving at full signing speed). The position-only rule therefore classified active signing as rest and deleted it: on the CSL test split it trimmed **every** clip, up to **~9 s of a ~10 s signing segment** (`S001667`: 10.77 s clip → 1.07 s span), while the caption stayed the full sentence — corrupting the translation supervision at the source. Requiring **near-zero motion** (a held pose) fixes this: on the same worst-case clips the trim drops to 0.00–0.07 s. A per-side cap `max_trim_s` (default 0.75 s) additionally guarantees a mis-classified slow passage can never delete more than that many seconds.

The trim interval is the contiguous non-rest core, capped per side; a safety floor preserves the clip if trimming would leave fewer than 3 frames (needed for Hermite endpoint velocities). $\rho$ and `max_trim_s` are physical **calibration** constants (real rest is near-still), not fit parameters.

After this transformation, each clip starts and ends _during signing activity_. Concatenating two trimmed clips produces a kinematically continuous signing-to-signing transition rather than a rest-pose-mediated one — exactly the **co-articulation** phenomenon real signers exhibit (also called _movement epenthesis_ in the sign-linguistics literature).

## 4.5 Hermite bridge — C¹-continuous seam

Even after trimming, the last frame of clip $A$ and the first frame of clip $B$ are kinematically independent — joining them directly would create a position discontinuity (teleport) and a velocity discontinuity at the seam. A linear interpolation between $\mathbf{p}_0 = c_A[-1]$ and $\mathbf{p}_1 = c_B[0]$ would remove the position teleport but still has $C^0$ continuity (velocity discontinuity remains). A cubic Hermite spline achieves $C^1$ continuity by additionally matching the per-frame velocity at each endpoint.

Per joint, on $(x, y)$ only (the confidence channel is propagated separately, see below), the spline is parameterized by $s \in [0, 1]$:

$$\mathbf{h}(s) \ =\  h_{00}(s)\ \mathbf{p}_0 \ +\  h_{10}(s)\ \mathbf{m}_0 \ +\  h_{01}(s)\ \mathbf{p}_1 \ +\  h_{11}(s)\ \mathbf{m}_1$$

with the standard cubic Hermite basis polynomials

$$h_{00}(s) = 2s^3 - 3s^2 + 1, \qquad h_{10}(s) = s^3 - 2s^2 + s$$

$$h_{01}(s) = -2s^3 + 3s^2, \qquad h_{11}(s) = s^3 - s^2$$

The **endpoint tangents** $\mathbf{m}_0, \mathbf{m}_1$ are derived from the actual per-frame velocities at the seam, scaled to per-unit-$s$ via $\text{span} = n + 1$:

$$\mathbf{v}_0 = c_A[-1] - c_A[-2], \qquad \mathbf{v}_1 = c_B[1]  - c_B[0]$$

$$\mathbf{m}_i = \mathbf{v}_i \cdot (n+1) \cdot \alpha_n$$

where $\alpha_n$ is the **length-aware tangent damping** factor

$$\boxed{\ \alpha_n = \frac{n_\text{min}}{\max(n, n_\text{min})}\ }$$

with $n_\text{min} = 2$. This factor is the key to making the spline behave correctly across both ends of the bridge-length spectrum:

- **Short bridges** ($n \leq n_\text{min}$): $\alpha_n = 1$ → full clip-end momentum is preserved → the spline reproduces the natural movement-epenthesis trajectory between two signing poses (co-articulation).
- **Long bridges** ($n \gg n_\text{min}$): $\alpha_n \to 0$ → tangent contribution vanishes → the spline degenerates to a position-only interpolation between $\mathbf{p}_0$ and $\mathbf{p}_1$ → the hand smoothly settles into rest, no overshoot.

Tangent magnitude is additionally clamped per joint to prevent overshoot when endpoint velocities happen to be locally large:

$$\lVert\mathbf{m}_i\rVert \ \leftarrow\  \min\ (\lVert\mathbf{m}_i\rVert,\  2\ \lVert\mathbf{p}_1 - \mathbf{p}_0\rVert)$$

Confidence is the element-wise minimum of the two endpoint confidences, $c_\text{out}^{(j)} = \min(c_0^{(j)}, c_1^{(j)})$, marking interpolated frames as no more confident than the worst of the two real endpoints.

The bridge length itself is sampled in seconds and converted to integer frames with the floor:

$$n \ =\  \max\ (n_\text{min},\  \lfloor \ell \cdot f \rceil)$$

The floor $n_\text{min} = 2$ is a **numerical** floor, not a tuning knob — a Hermite spline needs at least two interior frames to define a transition (one frame is just a point).

## 4.6 Inter-clip pause sampling

One attempt is to fit a LogNormal $\ell \sim \text{LogNormal}(\mu, \sigma)$ to BOBSL inter-subtitle gaps and sample from it. This fails badly because the real distribution is _bimodal_: a large point-mass at $\ell = 0$ (touching or overlapping subtitles = continuous co-articulated signing) plus a heavy-tailed positive component (real sentence-break pauses). No symmetric or unimodal parametric distribution captures the zero spike. We instead sample directly from the empirical CDF of BOBSL's manual-aligned inter-subtitle gaps:

$$\hat{\mathbb{P}}(\ell) \ =\  \frac{1}{|G|} \sum_{i=1}^{|G|} \delta(\ell - G_i), \qquad \ell \sim \hat{\mathbb{P}}$$

```python
def sample_pause_s(rng, G):
    return float(rng.choice(G))
```

The array $G$ is every adjacent-subtitle gap $g = \max(0,\ b.\text{start} - a.\text{end})$ over all BOBSL manual VTT files (per Eq 10 of the paper). Overlapping subtitles have a **negative** raw gap, but they are the *most* co-articulated boundaries, so they clamp to $0$ — they must **not** be dropped, or the zero-mass is understated. The exact counts are written by `analyze_bobsl_gaps.py`; on the current run **≈74% of gaps are exactly zero**, with a heavy positive tail (median ≈2 s).

So **≈74% of inter-clip joins concatenate co-articulated** by construction — no LogNormal fit, no `pause_min_s` / `pause_max_s` knob. The empirical CDF _is_ the model. The remaining ~26% of joins receive positive bridge durations from the right tail of $G$.

## 4.7 BG_pre / BG_post sampling

**Why a different sampler for BG**. The two background segments $B_\text{pre}$ and $B_\text{post}$ are conceptually distinct from inter-clip pauses: they represent the silent broadcast lead-in/lead-out, when there is no caption and the signer may not yet (or no longer) be signing. Sampling them from the full empirical $G$ would collapse ~74% of streams to a 2-frame BG (invisible). BG must always be **visibly present** so that the model gets exposure to genuine no-signing regions.

We therefore sample BG durations from the **conditional distribution** $\hat{\mathbb{P}}(\ell \mid \ell > 0)$, equivalently from the positive-only subset

$$G_+ = \\{ G_i \in G : G_i > 0 \\}$$

```python
def sample_bg_s(rng, G_plus):
    return float(rng.choice(G_plus))
```

This enforces "BG is always positive" by construction without introducing a min-duration knob: the floor is whatever the smallest positive BOBSL inter-subtitle gap is.

**Phantom clips.** For the bridge endpoint of the BG segment that is not adjacent to a chosen clip (i.e. the "outside" endpoint of $B_\text{pre}$ or $B_\text{post}$), we sample one additional clip from the same signer's spare pool — the **phantom clip** $\varphi$. This animates the BG region as a real signer-pose-to-real-signer-pose transition rather than a frozen frame. Phantom clips are **not** trimmed: BG segments represent the broadcast lead-in / lead-out, where retaining the phantom's natural rest pose is realistic. Formally:

$$\varphi_L, \varphi_R \ \sim\  \text{Uniform}(\mathcal{P}_s \setminus \\{c_1, \ldots, c_K\\})$$

If the spare pool is empty, the synthesizer falls back to using the first/last selected clip itself; the seam is still Hermite-interpolated, just less varied.

## 4.8 K-per-stream from a 60-second BOBSL window

**Why 60 s, not 15 s.** The model's training window is $W = 15$ s. If the synthesized streams were also $\sim 15$ s long, then evaluation would amount to single-window decoding — not streaming inference. To genuinely exercise streaming behaviour (cross-window event handling, state propagation across windows), streams must span _multiple_ training windows. We choose the design ratio

$$W_\text{stream} = 4\ W = 60 \text{ s}$$

as a deliberate compromise: long enough that streaming inference must fire, short enough that signer pools (especially PHOENIX's tight ones) can fill them.

**K range from BOBSL.** For each BOBSL manual VTT we slide a window of length $W_\text{stream}$ with a $1$-s step and count the number of subtitles that fall entirely within each window position. Let $\\{N_w\\}$ denote the resulting count distribution (over $\sim 161$k window positions across all VTT files). Then

$$K \ \sim\  \text{Uniform}\ \left(\\{K_\text{lo}, K_\text{lo}+1, \ldots, K_\text{hi}\\}\right)$$

with

$$K_\text{lo} = Q_{10}(\\{N_w\\}), \qquad K_\text{hi} = Q_{90}(\\{N_w\\})$$

In our current run, $K_\text{lo} = 6, K_\text{hi} = 19$. So a synthesized stream contains 6 to 19 sentences, mirroring how densely real BOBSL packs them in a 60-s window. For comparison, the same computation on a 15-s window gives $[Q_{10}, Q_{90}] = [0, 5]$ — confirming that the 60-s anchor produces $\sim 4 \times$ as many sentences per stream, exactly the design intent.

The per-stream $K$ is additionally clamped to the signer's pool size, $K \leftarrow \min(K, |\mathcal{P}_s|)$, since we sample without replacement.

## 4.9 Stream count

Each split contains a target number of streams chosen so that every clip is used about _once_ on average:

$$N_S \ =\  \text{round}\ \left(\frac{|\bigcup_s \mathcal{P}_s|}{\bar{K}}\right), \qquad \bar{K} = \frac{K_\text{lo} + K_\text{hi}}{2}$$

This preserves the offline split's contract: no clip duplication beyond what's statistically inevitable. We deliberately do **not** inflate this with a permutation-multiplier knob — on small signer pools (PHOENIX broadcasts have only 2-3 clips each) any multiplier above 1 saturates the $K!$ permutation space and biases training toward duplicated streams. (`rng.choice(..., replace=False)` already returns the chosen elements in a random order, so each stream is implicitly a random permutation of its chosen subset — but we let the permutation-space exploration arise naturally from the per-stream sampling, not from explicit replication.)

---

# 5. Putting it all together — full pseudocode

The two algorithms below assemble the components into the end-to-end pipeline.

## Algorithm 1 — synthesize one stream

```
Inputs:  Pool[s], G, G_+, K_lo, K_hi, FPS, MIN_BRIDGE_FRAMES, rng
Output:  pose tensor P (T, 133, 3),  cue list A = [(start_s, end_s, text)]

 1: K         ← rng.uniform_int(K_lo, min(K_hi, |Pool[s]|))
 2: chosen    ← rng.choice(Pool[s], K, replace=False)         # ORDER IS RANDOM (= permutation)
 3: resampled ← [trim_rest(resample_to_FPS(P_i, src_fps)) for c_i in chosen]   # trim_rest = held-rest only (§4.4)
 4: drop entries shorter than 1·FPS frames; if resampled = ∅ return empty stream
 5: spare ← Pool[s] \ chosen
 6: if |spare| ≥ 2:  φ_L, φ_R ← rng.choice(spare, 2)          # phantom clips for BG
    else: φ_L, φ_R ← (resampled[-1], resampled[0])            # fall back to chosen

 7: ℓ_pre  ← rng.choice(G_+)                                  # BG_pre  duration (positive-only)
 8: ℓ_post ← rng.choice(G_+)                                  # BG_post duration (positive-only)
 9: B_pre  ← Hermite(φ_L[-1], v(φ_L, "last"),
                     resampled[0][0], v(resampled[0], "first"),
                     n = max(MIN_BRIDGE_FRAMES, ⌊ℓ_pre · FPS⌉))

10: segments ← [B_pre]; cur ← |B_pre|; spans ← []; joins ← []      # lay out clips + inter-clip bridges
11: for k = 0 ... K-1:
12:     c0 ← cur; segments.append(resampled[k]); cur ← cur + |resampled[k]|; spans.append((c0, cur))
13:     if k < K-1:
14:         ℓ ← rng.choice(G)                                 # FULL empirical, ~74% zeros
15:         coart ← (⌊ℓ · FPS⌉ == 0);  n ← max(MIN_BRIDGE_FRAMES, ⌊ℓ · FPS⌉)
16:         Δ ← Hermite(resampled[k][-1], resampled[k+1][0], …, n)
17:         segments.append(Δ); cur ← cur + n; joins.append((n, coart))

18: B_post ← Hermite(resampled[-1][-1], φ_R[0], …, n = max(MIN_BRIDGE_FRAMES, ⌊ℓ_post · FPS⌉))
19: segments.append(B_post)

20: cues ← []                                                 # oracle cues (§4.1): absorb a co-articulated
21: for k, (c0, c1) ∈ enumerate(spans):                       #   (zero-gap) bridge into the two sentences so
22:     left  ← c0 − (⌈joins[k−1].n / 2⌉ if k>0   and joins[k−1].coart else 0)   #   they are frame-adjacent;
23:     right ← c1 + (⌊joins[k].n   / 2⌋ if k<K−1 and joins[k].coart   else 0)   #   a pause bridge stays a gap
24:     cues.append( (left/FPS, right/FPS, t_k) )
25: return (concat(segments), cues)
```

`v(P, end)` returns the per-frame velocity at the requested clip endpoint: $\mathbf{v}(P, \text{last}) = P[-1] - P[-2]$, $\mathbf{v}(P, \text{first}) = P[1] - P[0]$. Confidence channel set to $0$ (unused for tangent computation).

## Algorithm 2 — driver across splits

```
Inputs:  dataset D ∈ {PHOENIX, CSL, H2S}, seed σ
Output:  per-split stream files + manifest.json + subset2episode.json

 1: G        ← np.clip( load("bobsl_gap_samples.npy"), 0, ∞ )           # already g=max(0,·)
 2: G_+      ← G[G > 0]
 3: K_lo, K_hi ← p10, p90 of bobsl_stats["subs_per_stream_window"]      # 60-s sliding window
 4: for split ∈ {train, val, test}:
 5:     C_split ← load_split(D, split)     # big-pickle (PHOENIX) | per-sample .pkl (CSL/H2S; slice [start:end])
        drop clips with |P_i| < 1·src_fps or empty text
 6:     Pool ← group C_split by signer (§4.2);  drop pools with size < ρ_min
 7:     N_S ← max(1, round(|⋃ Pool| / mean(K_lo, K_hi)))                # stream count (§4.9)
 8:     for j = 1 ... N_S:
 9:         pick s uniformly from active signer IDs
10:         (P_j, A_j) ← Algorithm-1(Pool[s], G, G_+, K_lo, K_hi, FPS, MIN_BRIDGE_FRAMES, rng_j)
11:         if A_j ≠ ∅: write poses/<j>.npy, vtt/<j>.vtt, manifest entry
```

The construction `rng_j = np.random.default_rng([base_seed, j])` makes every stream's sampling deterministic and independent of stream ordering — re-running with the same seed reproduces every stream byte-for-byte.

---

# 6. Provenance of every parameter

Every value the synthesizer touches is either pure geometry or measured directly from the BOBSL manual annotations (`data/BOBSL/manual_annotations/signing_aligned_subtitles/*.vtt`, $4\,468$ files). Auto-aligned BOBSL is **not** used — that's the very source of annotation noise the original paper was criticised for.

| Parameter                  | Source                                                                 | Computed value                                                      |
| -------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------- |
| $f$ (FPS)                  | StreamSLST `config.FPS`                                                | $12.5$ Hz                                                           |
| $W$ (training window)      | StreamSLST `config.WINDOW_DURATION_SECONDS`                            | $15$ s                                                              |
| $W_\text{stream}$          | Design ratio $4 W$                                                     | $60$ s                                                              |
| $K_\text{lo}, K_\text{hi}$ | $p_{10}, p_{90}$ of `subs_per_60s_window`(BOBSL)                       | from `bobsl_gap_stats.json`                                        |
| $G$                        | All inter-subtitle gaps from BOBSL manual VTTs, $g=\max(0,\cdot)$      | `bobsl_gap_samples.npy` (≈74% zero, $p_{90}\approx 2$ s)          |
| $G_+$                      | $G \setminus \\{0\\}$                                                  | `bobsl_gap_samples.npy`, positive subset                          |
| $n_\text{min}$             | Numerical floor for Hermite (1 frame is just a point)                  | $2$                                                                 |
| $\rho_\text{min}$          | Need ≥1 phantom + ≥1 chosen                                            | $2$                                                                 |
| trim_rest rule             | Hands-below-shoulders AND near-still, signer-relative (§4.4)           | $\rho=0.015$ sw/frame, cap 0.75 s/side (calibration)                |
| Tangent damping $\alpha_n$ | $n_\text{min} / \max(n, n_\text{min})$                                 | reuses existing constant, no new knob                               |

---

# 7. Design decisions vs alternatives

| Choice we made                                     | Alternative considered         | Why we picked ours                                                                                                                                                                                        |
| -------------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Empirical pause sampling (§4.6)                    | LogNormal fit to $G_+$         | Real BOBSL is bimodal (74% zero + 26% positive heavy tail); no parametric distribution captures the zero spike. Empirical CDF _is_ the truth.                                                             |
| Positive-only $G_+$ for BG (§4.7)                  | Same $G$ as inter-clip         | With $G$, ~74% of streams collapse to a 2-frame BG (invisible). $G_+$ enforces visible BG without a min-duration knob.                                                                                    |
| `trim_rest` = below-shoulder AND near-still (§4.4) | Position-only (below-shoulder) | Position-only deletes low signing (CSL signs below the shoulder line): it trimmed ~9 s of a 10 s clip. Rest is a _held_ pose, so gating on near-zero motion is required; both terms stay signer-relative. |
| Hermite cubic spline (§4.5)                        | Linear interpolation           | Linear has $C^0$ continuity at the seam → visible velocity discontinuity. Hermite achieves $C^1$.                                                                                                         |
| Tangent damping by $1/n$ (§4.5)                    | Fixed tangent magnitude        | Fixed tangent overshoots on long bridges. $1/n$ damping makes long bridges relax to position-only interpolation.                                                                                          |
| Same-signer per stream (§4.2)                      | Cross-signer                   | Cross-signer introduces scale-jump and identity-change leakage; same-signer matches BOBSL.                                                                                                                |
| $K$ from 60-s window (§4.8)                        | $K$ from 15-s window           | 15 s = model window; streams must span ≥ 4× window for streaming inference to actually fire.                                                                                                              |
| Stream count = $\lvert C\rvert / \bar K$ (§4.9)    | Multiplier-based inflation     | On small pools (PHOENIX, ~3 clips/broadcast) any multiplier $>1$ saturates the $K!$ permutation space and biases training toward duplicated streams. Each clip used $\sim$ once is principled.            |

---

# 8. Failure modes & safeguards

| Failure mode                                  | Safeguard in code                                                                 |
| --------------------------------------------- | --------------------------------------------------------------------------------- |
| `trim_rest` deletes signing                   | Rest requires below-shoulder **and** near-still; per-side cap `max_trim_s`; 3-frame floor (§4.4) |
| Hermite called with $n \leq 0$                | Returns empty array of correct shape                                              |
| Hermite tangent magnitude explodes            | Clamped to $2 \lVert \mathbf{p}_1 - \mathbf{p}_0\rVert$ per joint per side        |
| Sampled gap = 0 (co-articulated)              | Bridge floored to $n_\text{min}=2$ for $C^1$ continuity **and** absorbed into the two sentences (frame-adjacent, §4.1) |
| Signer pool too small for phantoms            | Fall back to using first/last chosen clip as phantom (still Hermite-interpolated) |
| Subtitle is empty / clip < 1 s after resample | Drop the clip; skip stream if no clips remain                                     |
| Negative inter-subtitle gap (overlap)         | Clamped to 0 ($g=\max(0,\cdot)$) — the most co-articulated boundary, **not** dropped (§4.6) |
| Pose confidence values $> 1.0$                | Clipped to $[0, 1]$ at clip-load time                                             |
| **Signing silently deleted vs caption**       | `verify_stream_integrity.py` gate: each cue span must ≈ its source signing segment (§11) |

---

# 9. Workflow

The pipeline is the same for every dataset — only the `DATASET` env var changes (`PHOENIX` | `CSL` | `H2S`).

```bash
# 0. (one-time) compute the gap + K stats from BOBSL MANUAL annotations. Writes bobsl_gap_stats.json
#    (summary) and bobsl_gap_samples.npy (empirical gap array, g=max(0,·)). If skipped, the synthesizer
#    falls back to an analytic distribution of the same shape.
python -m data_synth.analyze_bobsl_gaps \
    --vtt_dir data/BOBSL/manual_annotations/signing_aligned_subtitles \
    --out data_synth/stats/bobsl_gap_stats.json

# 1. Synthesize streams (everything derived from the data + BOBSL stats). H2S has no dev split -> --val_frac.
DATASET=PHOENIX python -m data_synth.synthesize_streams --out_root data/synth/phoenix
DATASET=CSL     python -m data_synth.synthesize_streams --out_root data/synth/csl
DATASET=H2S     python -m data_synth.synthesize_streams --out_root data/synth/h2s --val_frac 0.05

# 2. Integrity gate — every cue span must contain (nearly) the full source signing segment. Non-zero exit on FAIL.
DATASET=CSL python -m data_synth.verify_stream_integrity --root data/synth/csl --split test

# 3. Trim the mBART tokenizer + decoder to this dataset's vocabulary (writes captioners/trimmed_mbart_<ds>/).
DATASET=CSL python -m captioners.trim_mbart

# 4. Dataset stats (BOBSL-paper style; reports word + BPE vocab).
DATASET=CSL python -m data_synth.dataset_stats --root data/synth/csl --out data_synth/stats/csl_stats.json

# (optional) visualize a stream: renders the 77 model-input keypoints as MP4 with subtitle overlay.
DATASET=CSL python -m data_synth.visualize_stream \
    --pose data/synth/csl/poses/test_00000.npy --vtt data/synth/csl/vtt/test_00000.vtt \
    --out data_synth/examples/csl_test_00000.mp4
```

---

# 10. Resulting benchmark sizes

$K$ is derived from a **60-second** BOBSL window ($=4W$, the model's training window), so streams span multiple model windows and the streaming inference behaviour actually fires at eval time. Stream length depends on the signer-pool size: PHOENIX dev/test broadcasts have only ~2–3 clips each (short streams; the underlying biological signer count is 9 across 629 broadcasts), while CSL's 10 signers and H2S's per-video pools are much larger.

Regenerate the exact per-split numbers after synthesis with `dataset_stats` — the trim and boundary fixes changed them, so any previously-recorded sizes are **stale**:

```bash
DATASET=CSL python -m data_synth.dataset_stats --root data/synth/csl --out data_synth/stats/csl_stats.json
```

| split | streams | hours | cues/stream | stream dur p50/p90 (s) | pause med/p90 (s) | density |
| ----- | ------: | ----: | ----------: | ---------------------: | ----------------: | ------: |
| _populate per split by running `dataset_stats`_ |||||||

---

# How2Sign — upstream caveats

How2Sign uses [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign)'s released RTMPose data via the per-sample layout above. Three upstream quirks are handled by the loader:

- **[Uni-Sign #2](https://github.com/ZechengLi19/Uni-Sign/issues/2)** — per-split counts differ from the Uni-Sign paper. We trust the gzipped labels file and print the actual loaded count.
- **[Uni-Sign #34](https://github.com/ZechengLi19/Uni-Sign/issues/34)** — no H2S dev split. Use `--val_frac` to carve one (deterministic given `--seed`).
- **[how2sign-data #4](https://github.com/how2sign/how2sign-data/issues/4)** — ~117 sample drift across CVPR21 / CSV / clip folders. The loader silently skips label entries whose pose `.pkl` is absent on disk and logs the drop count.

---

# 11. Integrity check

`verify_stream_integrity.py` guards against the class of bug where `trim_rest` / slicing silently deletes signing while the caption keeps the full sentence — i.e. the pose span no longer matches the reference text. For each stream it pairs the VTT cues (in order) with the source clips in `manifest.json` and checks `cue_span_s ≈ source_signing_s`; it prints a JSON report and exits non-zero on `FAIL`, so it can gate a re-synthesis. Run it on every dataset/split before trusting any numbers:

```bash
DATASET=CSL python -m data_synth.verify_stream_integrity --root data/synth/csl --split test
```

---

# 12. Reproducibility

Per-stream RNG is `np.random.default_rng([base_seed, stream_idx])` so re-running with the same `--seed` reproduces every stream byte-for-byte. Default seeds: train=42, val=43, test=44. Override via `--seed`. `manifest.json` records per stream: chosen clip names, phantom clip names, sampled pause durations, signer ID (YouTube ID for H2S), _K_, total duration, plus the `--val_frac` value and seed. The dataset can be reconstructed from the source poses + manifest alone.

---

# Files

- `synthesize_streams.py` — unified synthesis entry for PHOENIX, CSL, H2S (Algorithms 1 + 2: BOBSL-derived gaps, Hermite bridges, co-articulation-absorbed oracle cues).
- `analyze_bobsl_gaps.py` — writes `bobsl_gap_stats.json` (summary + K-range) and `bobsl_gap_samples.npy` (empirical gap array, `g=max(0,·)`, consumed by `synthesize_streams.py`).
- `verify_stream_integrity.py` — integrity gate: every cue span must ≈ its source signing segment; non-zero exit on FAIL (§11).
- `verify_synth.py` — round-trip sanity check (load → normalize → threshold → parse_vtt).
- `dataset_stats.py` — BOBSL-paper-style stats; reports word + BPE vocab; consumes `manifest.json`.
- `visualize_stream.py` — renders the 77 model-input keypoints as MP4 with subtitle overlay; segments labelled `[BG/PAUSE]` vs subtitle text.
