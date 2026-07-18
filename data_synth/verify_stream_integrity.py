'''Integrity gate: every synthesized stream cue must contain (nearly) the full source signing segment.

Catches the class of bug where trim_rest / slicing silently deletes signing while the caption stays the
full sentence — i.e. the pose span no longer matches the reference text. For each stream we pair its VTT
cues (in order) with the source clips recorded in manifest.json and compare:

    cue_span_s        = cue.end - cue.start                          (what the model actually sees)
    source_signing_s  = (end - start) / src_fps   [CSL/H2S start,end] (the signing the text refers to)
                      = num_frames    / src_fps   [PHOENIX, no start,end]

A small deficit is expected (legitimate held-rest trim, <~0.75 s/side). A large deficit means signing was
deleted. Exit code is non-zero on FAIL so this can gate a re-synthesis.

Run:  DATASET=CSL python -m data_synth.verify_stream_integrity --root data/synth/csl --split test
'''
import argparse, gzip, json, pickle, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYNTH_META, DATASET
from utils import parse_vtt


def source_signing_seconds(name: str) -> float:
    # Duration (s) of the source clip's signing segment, matching what the loader feeds the synthesizer.
    src_fps = float(SYNTH_META['src_fps'])
    if 'pickle_prefix' in SYNTH_META:  # PHOENIX big-pickle: no start/end, full clip is the segment
        return None  # resolved from the pickle cache below
    pose_dir = Path(SYNTH_META['pickle_dir']) / SYNTH_META.get('pose_dir_name', 'poses')
    stem = name[:-4] if name.endswith('.mp4') else name
    with open(pose_dir / f'{stem}.pkl', 'rb') as f:
        p = pickle.load(f)
    n = np.asarray(p['keypoints']).shape[0]
    if 'start' in p and 'end' in p:
        s, e = int(p['start']), int(p['end'])
        if 0 <= s < e <= n: n = e - s
    return n / src_fps


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--tol_s', type=float, default=0.8, help='max acceptable per-cue deficit (s)')
    args = ap.parse_args()
    root = Path(args.root)

    manifest = json.loads((root / 'manifest.json').read_text())
    streams = manifest['streams'][args.split]
    src_fps = float(SYNTH_META['src_fps'])

    # PHOENIX big-pickle: preload the split to look up num_frames per clip
    pheonix_cache = {}
    if 'pickle_prefix' in SYNTH_META:
        split_on_disk = SYNTH_META.get('splits', {}).get(args.split, args.split)
        pk = Path(SYNTH_META['pickle_dir']) / f"{SYNTH_META['pickle_prefix']}.{split_on_disk}"
        pheonix_cache = pickle.load(open(pk, 'rb'))

    n_spans = n_trimmed = frames_lost = 0
    worst = []
    for st in streams:
        sid = st['stream_id']
        clips = st.get('clips', [])
        cues = parse_vtt(root / 'vtt' / f'{sid}.vtt')
        if len(cues) != len(clips):  # defensive: order/length must line up
            print(f'  WARN {sid}: {len(cues)} cues vs {len(clips)} manifest clips — skipping')
            continue
        for name, cue in zip(clips, cues):
            if pheonix_cache: src_s = pheonix_cache[name]['keypoint'].shape[0] / src_fps
            else: src_s = source_signing_seconds(name)
            span_s = cue['end'] - cue['start']
            deficit = src_s - span_s
            n_spans += 1
            if deficit > args.tol_s:
                n_trimmed += 1
                frames_lost += int(round(deficit * SYNTH_META['src_fps']))
                worst.append({'clip': name, 'stream': sid, 'source_s': round(src_s, 3),
                              'span_s': round(span_s, 3), 'deficit_s': round(deficit, 3)})

    worst.sort(key=lambda w: -w['deficit_s'])
    frac = n_trimmed / max(1, n_spans)
    verdict = 'PASS' if frac < 0.02 else 'FAIL'
    out = {'dataset': DATASET, 'split': args.split, 'spans': n_spans, 'over_trimmed_spans': n_trimmed,
           'over_trimmed_fraction': round(frac, 4), 'frames_lost_estimate': frames_lost,
           'tol_s': args.tol_s, 'verdict': verdict, 'worst': worst[:15]}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    sys.exit(0 if verdict == 'PASS' else 1)
