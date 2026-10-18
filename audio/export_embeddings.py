"""Export per-second audio outputs for the fusion work (team contract: 10 x D per sample, keyed by sample_id).

Writes to EMBED_ROOT/audio_<run_id>/ (all files share the row order of {split}_ids.csv):
  {split}_emb.npy    (N, 10, D)  float16  embedding per second (D = 256)
  {split}_probs.npy  (N, 10, 29) float16  predicted class probabilities per second; columns follow
                                          class_names.json (28 events alphabetically, then "background")
  class_names.json
Run: python -m audio.export_embeddings --run-id <id>
     python -m audio.export_embeddings --run-id <id> --final   # also publish to EMBED_ROOT/audio_final/ for the team
"""
import argparse
import json
import shutil

import numpy as np
from torch.utils.data import DataLoader

from audio import config as C
from audio.dataset import make_dataset, norm_stats
from audio.evaluate import load_model, pick_device, predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--final", action="store_true", help="also copy to embeddings/audio_final/ (the team's fixed path)")
    a = ap.parse_args()
    device = pick_device()
    model, fdir = load_model(C.ARTIFACT_DIR / a.run_id, device)
    out = C.EMBED_ROOT / f"audio_{a.run_id}"
    out.mkdir(parents=True, exist_ok=True)
    for split in C.SPLITS:
        P, _, _, E = predict(model, DataLoader(make_dataset(split, model.kind, fdir), batch_size=32), device)
        np.save(out / f"{split}_emb.npy", E.astype(np.float16))
        np.save(out / f"{split}_probs.npy", P.astype(np.float16))
        shutil.copy(fdir / f"{split}_ids.csv", out / f"{split}_ids.csv")
        print(f"{split}: emb {E.shape}, probs {P.shape}")
    (out / "class_names.json").write_text(json.dumps(norm_stats(fdir)[2] + ["background"], indent=1))
    (out / "SOURCE.txt").write_text(f"audio run: {a.run_id}\nfeatures: {fdir.name}\n")
    print(f"-> {out}")
    if a.final:
        publish_final(out, a.run_id, fdir)


def publish_final(src, run_id: str, fdir) -> None:
    """Copy an export to the fixed team path with a README describing the files."""
    final = C.EMBED_ROOT / "audio_final"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(src, final)
    res = C.ARTIFACT_DIR / run_id / "results.json"
    test = json.loads(res.read_text())["test"] if res.exists() else {}
    score = lambda k: f"{test[k]:.3f}" if k in test else "n/a"
    n = {s: len(open(final / f"{s}_ids.csv").read().splitlines()) - 1 for s in C.SPLITS}
    (final / "README.md").write_text(f"""# Audio outputs for fusion (final)

Source run: `{run_id}` (features: `{fdir.name}`). Test scores: clip accuracy {score('clip_accuracy')}, clip macro-F1 {score('clip_macro_f1')}, per-second accuracy {score('second_accuracy')}, per-second macro-F1 {score('second_macro_f1')}.

| File | Shape | Meaning |
|---|---|---|
| `{{split}}_ids.csv` | one row per sample | `sample_id` (= `clip_id__label`), `clip_id` (YouTube id), `category`, `start`, `end`. **Row i of every array below is row i of this file.** |
| `{{split}}_emb.npy` | (N, 10, 256) float16 | audio embedding for each of the 10 seconds |
| `{{split}}_probs.npy` | (N, 10, 29) float16 | predicted class probabilities per second; columns follow `class_names.json` (28 AVE events alphabetically, then `background`); each row sums to 1 |
| `class_names.json` | 29 names | column order of the probabilities |

Splits (AVE official): train {n['train']}, val {n['val']}, test {n['test']} samples. 45 AVE videos carry two annotations, so join on `sample_id`, not `clip_id`.

Second t covers audio from t to t+1 s, matching frame t when video is sampled at 1 fps.
For the label-agreement baseline, the audio label of second t is `class_names[probs[i, t].argmax()]`; the clip-level audio label is the argmax of `probs[i, :, :28].mean(0)`.
""")
    print(f"published -> {final}")


if __name__ == "__main__":
    main()
