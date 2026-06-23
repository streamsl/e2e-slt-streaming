'''Synthesize streaming sign-language datasets from offline pre-segmented pose data.

Single entry-point for all four corpora. Two source layouts are supported, auto-selected from
`SYNTH_META`:

  (A) Legacy big-pickle  -- one {.train,.dev,.test} pickle per split. Each is a dict
    `{name: {'keypoint': (T,133,3), 'text': str, 'num_frames': int}}`. Used by PHOENIX.
    Triggered when `SYNTH_META['pickle_prefix']` is present.

  (B) Per-sample pose `.pkl` (normalized [0,1] x,y + confidence + `w_h`, with optional `start`/`end` 
    for CSL) + a gzipped label dict keyed by sample name. Triggered when `pickle_prefix` is absent.

Both paths feed the **same BOBSL-style synthesis**: same-signer (CSL: `P\\d+`; H2S: VIDEO_ID
prefix) concatenation, BOBSL-derived pause sampling, Hermite C¹ bridges. The previous
CSV-driven H2S pathway (synthesize_h2s.py + op2coco.py) is removed -- the >10s instructional
dead-time in How2Sign produced too many empty windows.

Co-articulated streams via four minimal-knob mechanisms (unchanged from prior version):
  1. **trim_rest** strips leading/trailing rest frames (wrists below shoulders).
  2. **Empirical pause sampling** from BOBSL manual annotations (~74% zero, heavy positive tail).
  3. **BG_pre / BG_post** sampled from positives only (silent broadcast lead-in / lead-out).
  4. **K-per-stream from a 60s sliding window** (= 4x training window) so streaming actually fires.

A short Hermite bridge (>= MIN_BRIDGE_FRAMES) connects every seam smoothly even when the
sampled pause is 0; tangent magnitude is damped inversely with bridge length so long BG
segments do not extrapolate clip-end momentum into visible overshoot.

Outputs (drop-in BOBSL-style):
    POSE_ROOT/<stream_id>.npy               # (T, 133, 3) float32 at FPS in pixel coords (canvas = SYNTH_META src_w/h)
    VTT_DIR/<stream_id>.vtt                 # WEBVTT, one sentence per cue (single-line text)
    SUBSET_JSON                             # {"train": [...], "val": [...], "test": [...]}
    manifest.json                           # provenance: signer/clip ids per stream + seed

Run:
    DATASET=PHOENIX python -m data_synth.synthesize_streams --out_root data/synth/phoenix
    DATASET=CSL     python -m data_synth.synthesize_streams --out_root data/synth/csl
    DATASET=H2S     python -m data_synth.synthesize_streams --out_root data/synth/h2s

Shorter / easier streams (3..5 sentences each; also yields more total streams):
    DATASET=CSL     python -m data_synth.synthesize_streams --out_root data/synth/csl --k_range 3 5

Carve a val split from train (H2S has no Uni-Sign dev set; CSL/PHOENIX already have one but you
can override with this flag for a custom split):
    DATASET=H2S     python -m data_synth.synthesize_streams --out_root data/synth/h2s --val_frac 0.05

Caveats from the Uni-Sign / How2Sign upstreams (handled here):
  - ZechengLi19/Uni-Sign#2  : per-split counts differ from the Uni-Sign paper -- we trust the
                              gzipped labels file as source-of-truth and print the actual counts.
  - ZechengLi19/Uni-Sign#34 : Uni-Sign has no H2S dev split (loader raises NotImplementedError);
                              use --val_frac to carve val from train if you need one.
  - how2sign/how2sign-data#4: ~117-sample drift across CVPR21 / CSV / clip folders. We trust the
                              labels file but skip entries whose pose .pkl is absent on disk
                              (user-downloaded subsets are common) and log the count.
'''
import argparse, gzip, json, pickle, random, re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET, SYNTH_META, WINDOW_DURATION_SECONDS, FPS

# Minimum signer-pure pool size to be included in synthesis 
#   (otherwise the signer is dropped entirely since we can't form a multi-clip stream from them). 
# This is a tuning knob to trade off diversity (# signers) vs co-articulation realism (multi-clip streams per signer). 
# Setting it to 2 is a minimal requirement for co-articulation, 
#   and already excludes some very small pools with only one clip. 
# Setting it higher would exclude more signers and reduce diversity, 
#   but might improve co-articulation if those signers had many single-clip pools. 
# Setting it to 1 would include all signers but allow some single-clip streams with no co-articulation.
MIN_POOL_SIZE = 2  
MIN_BRIDGE_FRAMES = 2  # Numerical floor for Hermite interp to define a transition (not a tuning knob)

# Trim-rest geometry: COCO-WholeBody-133 indices for shoulders and wrists. A frame is "rest" when
# both wrists sit BELOW (image y-down: larger y) the shoulder line. Pure geometric rule, scales
# automatically across canvases by being relative to each signer's own shoulders.
SHOULDER_IDS = [5, 6]
WRIST_IDS = [9, 10]

# --- Loading & signer grouping ---------------------------------------------

def signer_id(dataset: str, clip_name: str) -> str:
    if dataset == 'PHOENIX': return re.sub(r'-\d+$', '', clip_name.split('/', 1)[-1])
    if dataset == 'CSL':
        m = re.search(r'_P(\d+)_', clip_name)
        return f'P{m.group(1)}' if m else clip_name
    if dataset == 'H2S':
        # How2Sign sentence names follow `<YouTubeID>_<sentence>-<take>-rgb_<view>.mp4`.
        # Sentences from the same source YouTube ID share a signer + recording setup, so the
        # YouTube ID is the natural "signer" pool unit. Stripping `_<digits>-...` isolates it
        # whether or not the .mp4 suffix is present.
        base = clip_name.rsplit('/', 1)[-1]
        m = re.match(r'^(.+?)_\d+-\d+-rgb_(?:front|side)(?:\.mp4)?$', base)
        return m.group(1) if m else base.rsplit('.', 1)[0]
    raise ValueError(dataset)


def group_by_signer(clips: Dict[str, dict], dataset: str) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = defaultdict(list)
    for name in clips.keys():
        g[signer_id(dataset, name)].append(name)
    return g


# --- Pose handling: native pixel space throughout --------------------------

def to_numpy_kpts(rec: dict) -> np.ndarray:
    kp = rec['keypoint']
    if hasattr(kp, 'cpu'): kp = kp.cpu().numpy()
    kp = np.asarray(kp, dtype=np.float32)
    if kp.ndim != 3 or kp.shape[1] != 133 or kp.shape[2] != 3:
        raise ValueError(f'Unexpected keypoint shape {kp.shape}')
    kp[..., 2] = np.clip(kp[..., 2], 0.0, 1.0)
    return kp


def trim_rest(kpts: np.ndarray) -> np.ndarray:
    '''Drop contiguous leading/trailing "rest" frames at clip boundaries.

    Real continuous signing exhibits **co-articulation**: signers do not return to a full rest pose between adjacent 
    sentences in same utterance. Each pose clip is recorded in isolation, so it begins with a "preparation" phase (hands 
    rising from lap to 1st sign) & ends with a "retraction" phase (hands falling back to lap). When clips are concatenated 
    naively, these isolation artefacts produce an obvious "hands-down -> long pause -> hands-up" boundary that localization 
    head can latch onto trivially. By trimming them, the synthesized stream looks like continuous broadcast signing where 
    sentence boundaries are *kinematically continuous* & must be inferred from sign content, not from a free pause cue.

    Rule (zero hyperparameters): a frame is "rest" iff both wrists sit BELOW the shoulder line in image-y coordinates. 
    Pure geometry, signer-relative, scales automatically across canvases. Trimming is contiguous-only (we never cut mid-clip) 
    and respects a 3-frame floor so the residual clip remains usable for Hermite endpoint-velocity estimation.
    '''
    T = kpts.shape[0]
    if T < 3: return kpts
    sh_y = kpts[:, SHOULDER_IDS, 1].mean(axis=1)
    wr_y = kpts[:, WRIST_IDS, 1].mean(axis=1)
    sh_c = kpts[:, SHOULDER_IDS, 2].min(axis=1)
    wr_c = kpts[:, WRIST_IDS, 2].min(axis=1)
    valid = (sh_c > 0) & (wr_c > 0)
    is_rest = (wr_y > sh_y) & valid  # image y-down: larger y == lower in image == hands below shoulders
    start = 0
    while start < T and is_rest[start]: start += 1
    end = T
    while end > start and is_rest[end - 1]: end -= 1
    if end - start < 3: return kpts  # safeguard: don't reduce clip below 3 frames
    return kpts[start:end]


def resample_to_fps(kpts: np.ndarray, src_fps: float, tgt_fps: float = FPS) -> np.ndarray:
    '''Vectorized linear resample on (x, y), nearest-neighbour on confidence.

    Replaces a 133-joint Python loop (~266 per-clip np.interp calls) with a single set of
    NumPy index/broadcast ops. Result is mathematically identical to the prior per-joint
    implementation (linear interp on x/y with side='left' nearest-neighbour for confidence).
    '''
    if abs(src_fps - tgt_fps) < 1e-6 or kpts.shape[0] < 2: return kpts
    T_src = kpts.shape[0]
    duration = T_src / src_fps
    T_tgt = max(1, int(round(duration * tgt_fps)))
    src_t = np.arange(T_src, dtype=np.float64) / src_fps
    tgt_t = np.clip(np.arange(T_tgt, dtype=np.float64) / tgt_fps, src_t[0], src_t[-1])

    # Linear interp on x, y: locate bracketing source indices and per-target weight.
    lo = np.clip(np.searchsorted(src_t, tgt_t, side='right') - 1, 0, T_src - 2)
    hi = lo + 1
    span = src_t[hi] - src_t[lo]
    span[span == 0] = 1.0
    frac = ((tgt_t - src_t[lo]) / span).astype(np.float32)[:, None, None]
    out = np.empty((T_tgt, 133, 3), dtype=np.float32)
    out[..., :2] = (1.0 - frac) * kpts[lo][..., :2] + frac * kpts[hi][..., :2]
    # Confidence: nearest-neighbour, matching prior `searchsorted side='left'` semantics.
    near = np.clip(np.searchsorted(src_t, tgt_t, side='left'), 0, T_src - 1)
    out[..., 2] = kpts[near, :, 2]
    return out


# --- Stream synthesis ------------------------------------------------------

def hermite_interp_segment(p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray, n: int) -> np.ndarray:
    '''Cubic Hermite spline interpolation per joint on (x, y); confidence = element-wise min.

    Endpoints `p0`, `p1` are real frames from the data (last frame of clip A, first frame of clip B). Tangents `v0`, `v1` 
    are the per-frame displacements at those endpoints (clip_A[-1] - clip_A[-2] and clip_B[1] - clip_B[0]) so interpolation 
    matches the signer's actual velocity at each side of the seam. This gives C1 continuity at boundaries: the hand motion
    at the end of the spoken sentence flows smoothly into the pause and out into the next sentence.

    Tangents are scaled to the spline parameter s in [0, 1] (so segment length-aware) and clamped to at most 2 * |p1 - p0| 
    per joint to prevent overshoot when endpoint velocities are large. 

    Zero free parameters: everything is derived from the data.
    '''
    if n <= 0: return np.zeros((0, 133, 3), dtype=np.float32)
    span = (n + 1.0)  # parameter step is 1/span between frames

    # Damp endpoint-velocity contribution for LONG bridges. Short bridges (n <= MIN_BRIDGE_FRAMES) keep full clip-end momentum 
    # -- this IS the co-articulation continuity we want when the sampled pause was 0. For longer bridges (e.g. multi-second BG 
    # segments) the full per-frame velocity propagated across many frames extrapolates the clip's last motion far beyond where
    # the hand should physically end up, producing visible overshoot. Damping inversely with n makes the spline smoothly converge 
    # to a position-only interpolation as the bridge grows.
    v_scale = float(MIN_BRIDGE_FRAMES) / max(n, MIN_BRIDGE_FRAMES)
    m0 = (v0[:, :2] * span * v_scale).astype(np.float32)
    m1 = (v1[:, :2] * span * v_scale).astype(np.float32)

    # Clamp tangent magnitude to 2 * |p1 - p0| per joint to prevent overshoot
    delta = (p1[:, :2] - p0[:, :2]).astype(np.float32)
    max_mag = 2.0 * np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-3
    for m in (m0, m1):
        mag = np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9
        scale = np.minimum(1.0, max_mag / mag)
        m *= scale  # in-place clamp

    out = np.zeros((n, 133, 3), dtype=np.float32)
    conf = np.minimum(p0[:, 2], p1[:, 2]).astype(np.float32)
    for t in range(n):
        s = (t + 1.0) / span  # s in (0, 1)
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        out[t, :, :2] = h00 * p0[:, :2] + h10 * m0 + h01 * p1[:, :2] + h11 * m1
        out[t, :, 2] = conf
    return out


def endpoint_velocity(clip: np.ndarray, end: str) -> np.ndarray:
    '''Per-frame (x, y, conf) velocity at a clip endpoint. `end` in {'last', 'first'}.

    Returns a (133, 3) array with v[:, :2] = displacement, v[:, 2] = 0 (unused).
    '''
    if clip.shape[0] < 2: return np.zeros((133, 3), dtype=np.float32)
    if end == 'last': v = (clip[-1] - clip[-2]).astype(np.float32)
    elif end == 'first': v = (clip[1] - clip[0]).astype(np.float32)
    else: raise ValueError(end)
    v[:, 2] = 0.0
    return v


def synth_one_stream(
    rng: np.random.Generator, signer_pool: List[str],
    clips: Dict[str, dict], src_fps: float, pause: dict, k_range: Tuple[int, int],
) -> Tuple[np.ndarray, List[Tuple[float, float, str]], dict]:
    k_lo, k_hi = k_range
    K_target = int(rng.integers(min(k_lo, len(signer_pool)), min(k_hi, len(signer_pool)) + 1))

    # Top-up loop: many CSL clips are very short (<1s after [start:end] slicing + trim_rest), so a plain rng.choice(size=K) 
    # frequently lands on clips that get dropped by the <1s filter, producing streams with fewer than k_lo sentences. We 
    # instead shuffle the whole signer pool once and walk it in random order, keeping every clip that passes the filters 
    # until we hit K_target (or exhaust the pool). This guarantees `len(keep_names) >= min(K_target, n_valid)`
    # while preserving determinism (rng seeded per stream).
    pool_order = list(np.asarray(signer_pool)[rng.permutation(len(signer_pool))])
    resampled, texts, keep_names = [], [], []
    attempted: set = set()
    min_frames_after_resample = int(1.0 * FPS)  # matches loader MIN_SUB_DURATION
    for name in pool_order:
        if len(resampled) >= K_target: break
        if name in attempted: continue
        attempted.add(name)
        rec = clips[name]
        text = str(rec.get('text', '')).strip()
        if not text: continue
        kp = trim_rest(resample_to_fps(to_numpy_kpts(rec), src_fps))
        if kp.shape[0] < min_frames_after_resample: continue
        resampled.append(kp)
        texts.append(text)
        keep_names.append(name)
    if not resampled: return np.zeros((0, 133, 3), dtype=np.float32), [], {'clips': [], 'K': 0}

    # Phantom clips for BG_pre and BG_post: random clips from this signer's pool that were NOT selected as kept content. 
    # We deliberately do NOT trim_rest the phantoms -- BG segments represent the broadcast lead-in / lead-out where signer 
    # naturally holds a rest pose, so retaining the rest frames at phantom endpoint is realistic. If the pool has no spare 
    # clips, fallback to 1st/last selected clip itself (the seam will still be Hermite-interpolated rather than a held frame).
    chosen_set = set(keep_names)
    spare = [n for n in signer_pool if n not in chosen_set]
    if spare:
        phantom_left_name = str(rng.choice(spare))
        phantom_right_name = str(rng.choice(spare))
        phantom_left = resample_to_fps(to_numpy_kpts(clips[phantom_left_name]), src_fps)
        phantom_right = resample_to_fps(to_numpy_kpts(clips[phantom_right_name]), src_fps)
    else:
        phantom_left_name = phantom_right_name = '<self>'
        phantom_left = resampled[-1] if len(resampled) > 1 else resampled[0]
        phantom_right = resampled[0] if len(resampled) > 1 else resampled[-1]

    segments: List[np.ndarray] = []
    cues: List[Tuple[float, float, str]] = []
    durations: dict = {'pre_s': 0.0, 'pauses_s': [], 'post_s': 0.0}
    cur = 0

    # BG_pre = Hermite interp from phantom_left[-1] -> resampled[0][0] (signer "transitioning into" the first sentence). 
    # Animated, biomechanically grounded in real signer poses. Sampled from the POSITIVE-only subset of BOBSL gaps so BG 
    # segments are always visibly present in stream -- representing the silent broadcast lead-in, not an inter-sentence join.
    L_pre_s = float(rng.choice(pause['bg_samples_s']))
    n_pre = max(MIN_BRIDGE_FRAMES, int(round(L_pre_s * FPS)))
    bg_pre = hermite_interp_segment(
        phantom_left[-1], resampled[0][0],
        endpoint_velocity(phantom_left, 'last'),
        endpoint_velocity(resampled[0], 'first'),
        n_pre,
    )
    segments.append(bg_pre)
    durations['pre_s'] = L_pre_s
    cur += bg_pre.shape[0]

    for i, (clip, text) in enumerate(zip(resampled, texts)):
        segments.append(clip)
        cues.append((cur / FPS, (cur + clip.shape[0]) / FPS, text))
        cur += clip.shape[0]
        if i < len(resampled) - 1:
            L_pause_s = float(rng.choice(pause['samples_s']))
            n_pause = max(MIN_BRIDGE_FRAMES, int(round(L_pause_s * FPS)))
            pause_seg = hermite_interp_segment(
                clip[-1], resampled[i + 1][0],
                endpoint_velocity(clip, 'last'),
                endpoint_velocity(resampled[i + 1], 'first'),
                n_pause,
            )
            segments.append(pause_seg)
            cur += pause_seg.shape[0]
            durations['pauses_s'].append(L_pause_s)

    # BG_post = Hermite interp from resampled[-1][-1] -> phantom_right[0] 
    # (broadcast lead-out; same positive-only sampling as BG_pre).
    L_post_s = float(rng.choice(pause['bg_samples_s']))
    n_post = max(MIN_BRIDGE_FRAMES, int(round(L_post_s * FPS)))
    bg_post = hermite_interp_segment(
        resampled[-1][-1], phantom_right[0],
        endpoint_velocity(resampled[-1], 'last'),
        endpoint_velocity(phantom_right, 'first'),
        n_post,
    )
    segments.append(bg_post)
    cur += bg_post.shape[0]
    durations['post_s'] = L_post_s

    poses = np.concatenate(segments, axis=0).astype(np.float32)
    # Hermite-spline bridges can overshoot endpoint bounds by ~2% even with the tangent-magnitude clamp, so a clip just inside 
    # the source clips can produce a stream frame just outside it. Clamp xy to the unified canvas so downstream consumers can 
    # rely on `poses[..., :2]` staying in [0, src_w] x [0, src_h]. The PHOENIX big-pickle path already lands within canvas so 
    # this is a no-op there; for CSL/H2S Uni-Sign it cleans up the bridge artefacts.
    poses[..., 0] = np.clip(poses[..., 0], 0.0, float(SYNTH_META['src_w']))
    poses[..., 1] = np.clip(poses[..., 1], 0.0, float(SYNTH_META['src_h']))
    prov = {
        'clips': keep_names, 'K': len(cues),
        'phantom_left': phantom_left_name, 'phantom_right': phantom_right_name,
        'duration_s': float(poses.shape[0] / FPS), **durations,
    }
    return poses, cues, prov


# --- Output writers --------------------------------------------------------

def fmt_vtt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


def write_vtt(path: Path, cues: List[Tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['WEBVTT', '']
    for s, e, text in cues:
        lines.append(f'{fmt_vtt_time(s)} --> {fmt_vtt_time(e)}')
        lines.append(' '.join(text.split()))
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_pose(out_dir: Path, stream_id: str, poses: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'{stream_id}.npy', poses.astype(np.float32))


# --- Per-split orchestration -----------------------------------------------

def load_pause_dist(stats_path: str) -> dict:
    '''Load BOBSL empirical inter-subtitle gap samples (replaces the old LogNormal parameters).

    Looks for `bobsl_gap_samples.npy` next to the stats JSON. If absent, falls back to an
    analytic emulation that reproduces the real BOBSL shape (~74% zero + LogNormal positive
    tail) so the pipeline still runs without the optional samples file.

    Returned dict: {'samples_s': np.ndarray, 'frac_zero': float, 'median_s': float, 'p90_s': float}
    '''
    samples_path = Path(stats_path).with_name('bobsl_gap_samples.npy')
    if samples_path.exists():
        samples = np.load(samples_path).astype(np.float32)
        samples = np.clip(samples, 0.0, None)  # negative gaps (overlapping subs) = co-articulated, treat as 0
        src = f'BOBSL empirical n={samples.size}'
    else:
        rng = np.random.default_rng(0)
        n = 10000
        is_pos = rng.random(n) > 0.74  # match observed BOBSL ~26% positive-gap fraction
        samples = np.where(is_pos, rng.lognormal(np.log(2.0), 0.83), 0.0).astype(np.float32)
        src = f'fallback (no {samples_path.name}; emulating 74% zero + LogNormal positive tail)'

    # Positive-only subset: used for BG_pre/BG_post (broadcast lead-in / lead-out). Inter-clip pauses use full empirical (which is 
    # ~74% zeros, capturing co-articulation), but BG segments are conceptually different -- they represent silent broadcast preamble 
    # where there is no caption, not an inter-sentence join. Sampling them from full empirical collapses ~74% of streams to a 2-frame 
    # BG (invisible). Sampling from positives only keeps them broadcast-realistic (median ~2s, heavy tail) without adding new knobs.
    bg_samples = samples[samples > 0]
    if bg_samples.size == 0: bg_samples = np.array([2.0], dtype=np.float32)  # degenerate fallback
    info = {
        'samples_s': samples,
        'bg_samples_s': bg_samples,
        'frac_zero': float((samples == 0).mean()),
        'median_s': float(np.median(samples)),
        'p90_s': float(np.percentile(samples, 90)),
        'bg_median_s': float(np.median(bg_samples)),
        'bg_p90_s': float(np.percentile(bg_samples, 90)),
    }
    print(f"(pause samples: {src}; frac_zero={info['frac_zero']:.3f}, "
          f"median={info['median_s']:.2f}s, p90={info['p90_s']:.2f}s; "
          f"BG (positive only): median={info['bg_median_s']:.2f}s, p90={info['bg_p90_s']:.2f}s)")
    return info


def load_k_range(stats_path: str) -> Tuple[int, int]:
    '''Sentences-per-stream range derived from BOBSL "subs per 15s sliding window" stats.

    If `bobsl_gap_stats.json` does not contain `subs_per_window_p10` / `subs_per_window_p90`
    (older versions), fall back to deriving K from `sub_dur_median_s` + `gap_median_s`:
        K_typical = WINDOW_DURATION_SECONDS / (sub_dur_median + max(gap_median_pos, 0))
        K_lo = max(1, K_typical // 2);  K_hi = K_typical * 2
    Otherwise use the empirical p10..p90 range from the BOBSL distribution.
    '''
    p = Path(stats_path)
    if p.exists():
        s = json.loads(p.read_text())
        # Prefer subs_per_STREAM_window (60s) — anchors K to a stream length that spans multiple
        # 15s training windows so the streaming inference behaviour actually fires at eval time.
        if 'subs_per_stream_window_p10' in s and 'subs_per_stream_window_p90' in s:
            lo = max(2, int(s['subs_per_stream_window_p10']))
            hi = max(lo + 1, int(s['subs_per_stream_window_p90']))
            sw = float(s.get('stream_window_seconds', 60.0))
            print(f'(BOBSL K range: [{lo}, {hi}] from subs_per_{sw:.0f}s_window p10..p90)')
            return lo, hi
        if 'subs_per_window_p10' in s and 'subs_per_window_p90' in s:
            lo = max(1, int(s['subs_per_window_p10']))
            hi = max(lo + 1, int(s['subs_per_window_p90']))
            print(f'(BOBSL K range: [{lo}, {hi}] from subs_per_15s_window p10..p90 [no 60s stats found])')
            return lo, hi
        sub_med = float(s.get('sub_dur_median_s', 4.0))
        # gap_median_s is often 0 in BOBSL because most subs are adjacent; use exp(log_pos_mean) as positive median
        gap_med_pos = float(np.exp(s.get('log_pos_mean', np.log(2.0))))
        cycle = max(2.0, sub_med + gap_med_pos)
        k_typ = max(2, int(round(WINDOW_DURATION_SECONDS / cycle)))
        lo, hi = max(1, k_typ // 2), max(k_typ + 1, k_typ * 2)
        print(f'(K range derived from sub_dur_median={sub_med:.1f}s + gap_median_pos={gap_med_pos:.1f}s '
              f'-> K_typical={k_typ}, range [{lo}, {hi}])')
        return lo, hi
    print('(no BOBSL stats; K range fallback [2, 10])')
    return 2, 10


def _load_per_sample_split(split_name_on_disk: str) -> Dict[str, dict]:
    '''Load a per-sample split (CSL or H2S) into the same `{name: {keypoint, text, ...}}` schema as the legacy big-pickle path:

        - parse gzipped labels.<split> (sample dict)
        - read poses/<name>.pkl (RTMPose normalized x,y in [0,1] + per-frame score + w_h)
        - CSL only: slice keypoints[start:end]
        - merge (T,1,133,2) xy with (T,1,133) confidence -> (T,133,3) on a unified canvas (multiplied by SYNTH_META src_w/src_h 
          so cross-clip endpoints are scale-comparable for Hermite bridging; group-relative normalization in poses/preprocessing.py 
          is scale- invariant downstream, so the choice of canvas only affects bridge endpoint deltas).
        - clip confidence to [0,1] (RTMPose occasionally returns ~1.1)
        - skip samples with empty text, missing pose .pkl, or fewer than 1s of frames after slicing.
    '''
    pickle_dir = Path(SYNTH_META['pickle_dir'])
    pose_dir = pickle_dir / SYNTH_META.get('pose_dir_name', 'poses')
    label_path = pickle_dir / f"{SYNTH_META.get('label_prefix', 'labels')}.{split_name_on_disk}"
    with gzip.open(label_path, 'rb') as f: labels = pickle.load(f)

    canvas = np.asarray([SYNTH_META['src_w'], SYNTH_META['src_h']], dtype=np.float32)
    min_src_frames = int(1.0 * SYNTH_META['src_fps'])
    out: Dict[str, dict] = {}
    n_missing_pose = n_empty_text = n_short = 0

    for sample_name, entry in tqdm(labels.items(), desc=f'[{DATASET}/{split_name_on_disk}] load Uni-Sign'):
        text = str(entry.get('text', '')).strip()
        if not text: n_empty_text += 1; continue
        # H2S keys carry .mp4; CSL keys do not. The pose file is always <name without .mp4>.pkl.
        pose_stem = sample_name[:-4] if sample_name.endswith('.mp4') else sample_name
        pose_path = pose_dir / f'{pose_stem}.pkl'
        if not pose_path.exists(): n_missing_pose += 1; continue
        try:
            with open(pose_path, 'rb') as f: pose_pkl = pickle.load(f)
        except Exception as e:
            print(f'  WARN: failed to read {pose_path.name}: {e}'); n_missing_pose += 1; continue

        kpts = np.asarray(pose_pkl['keypoints'])  # (T, 1, 133, 2) normalized [0, 1] xy
        scores = np.asarray(pose_pkl['scores'])   # (T, 1, 133) confidence (can exceed 1.0)
        if kpts.ndim == 4 and kpts.shape[1] == 1: kpts = kpts[:, 0]      # (T, 133, 2)
        if scores.ndim == 3 and scores.shape[1] == 1: scores = scores[:, 0]  # (T, 133)
        if 'start' in pose_pkl and 'end' in pose_pkl:
            s, e = int(pose_pkl['start']), int(pose_pkl['end'])
            if 0 <= s < e <= kpts.shape[0]:
                kpts = kpts[s:e]; scores = scores[s:e]
        if kpts.shape[0] < min_src_frames: n_short += 1; continue

        # Scale normalized [0,1] to a unified pixel-equivalent canvas. Per-sample w_h varies in H2S; rescaling each sample 
        # to (src_w, src_h) makes endpoint deltas (clip[-1] vs phantom[0]) comparable across signers without recovering each 
        # video's native resolution. We clip normalized coords to [0,1] before scaling: RTMPose occasionally predicts keypoints 
        # outside the frame (face/hands leaving view, or extrapolated low-confidence detections), which would otherwise produce 
        # pose values > canvas size. Clipping keeps output strictly within [0, src_w] x [0, src_h] -- the few clipped points are 
        # typically low-confidence anyway and get zeroed by threshold_confidence() downstream.
        kpts_norm = np.clip(kpts.astype(np.float32), 0.0, 1.0)
        xy = kpts_norm * canvas[None, None, :]
        conf = np.clip(scores.astype(np.float32), 0.0, 1.0)[..., None]
        keypoint = np.concatenate([xy, conf], axis=-1).astype(np.float32)  # (T, 133, 3)

        out[sample_name] = {
            'name': sample_name,
            'text': text,
            'keypoint': keypoint,
            'num_frames': int(keypoint.shape[0]),
            'gloss': entry.get('gloss', ''),
        }
    print(f'  loaded {len(out)} samples (skipped: {n_missing_pose} missing pose, '
          f'{n_empty_text} empty text, {n_short} <1s after slicing)')
    return out


def _load_legacy_pickle_split(split_name_on_disk: str) -> Dict[str, dict]:
    pickle_path = SYNTH_META['pickle_dir'] / f"{SYNTH_META['pickle_prefix']}.{split_name_on_disk}"
    print(f'[{DATASET}/{split_name_on_disk}] loading legacy pickle {pickle_path}')
    with open(pickle_path, 'rb') as f: return pickle.load(f)


def synthesize_split(
    split: str, out_pose_dir: Path, out_vtt_dir: Path,
    base_seed: int, pause: dict, k_range: Tuple[int, int],
) -> Tuple[List[str], List[dict]]:
    # Map our split name to the on-disk split name (e.g. CSL/PHOENIX -> 'dev', H2S has no 'dev' key).
    split_map = SYNTH_META.get('splits', {'train': 'train', 'val': 'dev', 'test': 'test'})
    if split not in split_map:
        # Caller asked for a split this dataset's source does not provide (e.g. H2S 'val' when
        # --val_frac is 0). Synthesize an empty split so the orchestrator's output stays uniform.
        print(f'[{DATASET}/{split}] no source split mapping -- skipping')
        return [], []

    load_split = split_map[split]
    if 'pickle_prefix' in SYNTH_META: clips = _load_legacy_pickle_split(load_split)
    else: clips = _load_per_sample_split(load_split)

    min_src_frames = int(1.0 * SYNTH_META['src_fps'])
    clips = {k: v for k, v in clips.items() if str(v.get('text', '')).strip() and v.get('num_frames', 0) >= min_src_frames}
    groups = group_by_signer(clips, DATASET)
    groups = {sid: lst for sid, lst in groups.items() if len(lst) >= MIN_POOL_SIZE}
    n_usable = sum(len(v) for v in groups.values())
    k_avg = (k_range[0] + k_range[1]) / 2.0
    # Each clip used ~once on average across the split (n_usable / K_avg). Preserves the offline split's contract: no clip 
    # duplication beyond what's statistically inevitable. We deliberately do NOT inflate this with a permutation-multiplier knob 
    # -- on small signer pools (PHOENIX broadcasts have only 2-3 clips each) any multiplier above 1 saturates the K! permutation
    # space and biases training toward duplicated streams.
    n_streams = max(1, int(round(n_usable / k_avg)))
    print(f'[{DATASET}/{split}] usable clips: {n_usable}, signer-pure groups: {len(groups)} -> {n_streams} streams')
    if not groups: raise RuntimeError(f'No signer groups for {DATASET}/{split}')

    rng = np.random.default_rng(base_seed)
    signer_ids = list(groups.keys())
    stream_ids: List[str] = []
    manifest: List[dict] = []
    
    for i in tqdm(range(n_streams), desc=f'synth {split}'):
        stream_rng = np.random.default_rng([base_seed, i])
        sid = signer_ids[int(rng.integers(0, len(signer_ids)))]
        poses, cues, prov = synth_one_stream(stream_rng, groups[sid], clips, SYNTH_META['src_fps'], pause, k_range)
        if not cues: continue
        stream_id = f'{split}_{i:05d}'
        write_pose(out_pose_dir, stream_id, poses)
        write_vtt(out_vtt_dir / f'{stream_id}.vtt', cues)
        prov.update({'stream_id': stream_id, 'signer': sid, 'split': split})
        manifest.append(prov)
        stream_ids.append(stream_id)
    return stream_ids, manifest


def _carve_val_from_train(subset: Dict[str, List[str]], full_manifest: Dict[str, List[dict]], val_frac: float, seed: int) -> None:
    '''Move a deterministic random fraction of train STREAMS into val, in-place.

    Mutates subset and full_manifest. Useful when the source corpus has no dev split (H2S) or when a custom split ratio is desired. 
    We carve at the STREAM level (not the source-clip level) so each carved stream is internally consistent and the per-window 
    evaluation logic needs no special-case handling.
    '''
    if val_frac <= 0.0: return
    if not (0.0 < val_frac < 1.0): raise ValueError(f'--val_frac must be in (0, 1); got {val_frac}')
    train_ids = list(subset.get('train', []))
    n_move = int(round(len(train_ids) * val_frac))
    if n_move <= 0:
        print(f'(--val_frac={val_frac:g} on {len(train_ids)} train streams rounds to 0 -- no carve)')
        return

    rng = random.Random(seed)
    move_set = set(rng.sample(train_ids, n_move))
    subset['train'] = [sid for sid in train_ids if sid not in move_set]
    subset['val']   = sorted(list(subset.get('val', [])) + [sid for sid in train_ids if sid in move_set])

    # Manifest: rebuild train+val entry lists keyed by stream_id (preserves provenance).
    train_manifest = full_manifest.get('train', [])
    keep, moved = [], []
    for m in train_manifest:
        (moved if m.get('stream_id') in move_set else keep).append(m)

    full_manifest['train'] = keep
    full_manifest['val'] = list(full_manifest.get('val', [])) + moved
    print(f'(carved {n_move}/{len(train_ids)} train streams into val [seed={seed}, frac={val_frac:g}])')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_root', required=True)
    p.add_argument('--bobsl_gap_stats', default='data_synth/stats/bobsl_gap_stats.json',
                   help='JSON of BOBSL manual-aligned pause statistics; if missing, use fallback.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k_range', type=int, nargs=2, metavar=('LO', 'HI'), default=None,
                   help='Override sentences-per-stream range [LO, HI]. e.g. --k_range 3 5 for shorter, '
                        'easier streams (also yields more total streams since n_streams = n_usable/K_avg).')
    p.add_argument('--val_frac', type=float, default=0.0,
                   help='Fraction of TRAIN streams to move into val after synthesis (default 0 = no carve). '
                        'Useful for H2S where the Uni-Sign labels file has no dev split. Deterministic '
                        'given --seed. Applied AFTER per-split synthesis so each carved stream is intact.')
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_pose_dir = out_root / 'poses'
    out_vtt_dir = out_root / 'vtt'
    out_pose_dir.mkdir(parents=True, exist_ok=True)
    out_vtt_dir.mkdir(parents=True, exist_ok=True)

    pause = load_pause_dist(args.bobsl_gap_stats)
    if args.k_range is not None:
        lo, hi = int(args.k_range[0]), int(args.k_range[1])
        if lo < 1 or hi < lo: raise ValueError(f'invalid --k_range {args.k_range}: need 1 <= LO <= HI')
        k_range = (lo, hi)
        print(f'(K range overridden by --k_range: [{lo}, {hi}])')
    else:
        k_range = load_k_range(args.bobsl_gap_stats)

    splits_cfg = [('train', args.seed), ('val', args.seed + 1), ('test', args.seed + 2)]
    subset: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}
    full_manifest: Dict[str, List[dict]] = {'train': [], 'val': [], 'test': []}
    for split, seed in splits_cfg:
        ids, m = synthesize_split(split, out_pose_dir, out_vtt_dir, seed, pause, k_range)
        subset[split] = ids
        full_manifest[split] = m

    # Optional train -> val carve (default 0% = no carve). Done after synthesis so each carved
    # stream is fully written to disk and only the subset2episode.json mapping is rerouted.
    _carve_val_from_train(subset, full_manifest, args.val_frac, args.seed)
    (out_root / 'subset2episode.json').write_text(json.dumps(subset, indent=2))
    (out_root / 'manifest.json').write_text(json.dumps({
        'dataset': DATASET,
        'src_meta': {k: (str(v) if isinstance(v, Path) else v) for k, v in SYNTH_META.items()},
        'target_fps': FPS,
        # Drop the raw samples arrays from manifest -- store only the summary statistics that
        # describe the empirical pause distribution actually used during synthesis.
        'pause': {k: v for k, v in pause.items() if k not in ('samples_s', 'bg_samples_s')},
        'k_range': list(k_range), 'min_pool_size': MIN_POOL_SIZE, 'window_seconds': WINDOW_DURATION_SECONDS,
        'min_bridge_frames': MIN_BRIDGE_FRAMES, 'trim_rest': True,
        'val_frac': args.val_frac, 'seed': args.seed,
        'streams': full_manifest,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nDone. Wrote {sum(len(v) for v in subset.values())} streams to {out_root}")
