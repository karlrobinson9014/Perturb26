"""Does semantic similarity to the true class predict the class the best competitor chose? (offline, CPU)

Needs packages outside requirements.txt (nltk + WordNet corpus, open_clip_torch, timm, regex, ftfy) and
imagenet_class_index.json. Install them into a separate directory, not .venv (the loop integrity-checks it):
    PYTHONPATH=<scratch>/pkgs HF_HOME=<scratch>/dl/hf python tools/semantic_classes.py <repo> <scratch>
<scratch>/dl/nltk holds the WordNet corpus, <scratch>/dl/idx/imagenet_class_index.json the class index.
The openai ViT-B-32 TorchScript .pt crashes on this CPU (illegal instruction); weights load from the timm safetensors.
Result 2026-10-22: results/semantic-classes-dev.json, research_log 2026-10-22.
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch

repo, scratch = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "upstream"))

import nltk  # noqa: E402

nltk.data.path.insert(0, str(scratch / "dl" / "nltk"))
from nltk.corpus import wordnet as wn  # noqa: E402

idx = json.loads((scratch / "dl" / "idx" / "imagenet_class_index.json").read_text())
wnids = [idx[str(i)][0] for i in range(1000)]
names = [idx[str(i)][1].replace("_", " ") for i in range(1000)]
syn = [wn.synset_from_pos_and_offset("n", int(w[1:])) for w in wnids]

# CLIP text embeddings
import open_clip  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

clip = open_clip.create_model("ViT-B-32", pretrained=None, quick_gelu=True)
clip.load_state_dict(load_file(hf_hub_download("timm/vit_base_patch32_clip_224.openai", "open_clip_model.safetensors")))
clip.eval()
tok = open_clip.get_tokenizer("ViT-B-32")
with torch.no_grad():
    te = torch.cat([clip.encode_text(tok([f"a photo of a {n}." for n in names[i:i + 100]])) for i in range(0, 1000, 100)])
te = torch.nn.functional.normalize(te, dim=1)
clip_sim = te @ te.T

# The attacked model's own classifier weights
from perturbnet.model import load_efficientnet_v2_l  # noqa: E402

model = load_efficientnet_v2_l(torch.device("cpu"))
fc = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1].weight.detach()
fcn = torch.nn.functional.normalize(fc, dim=1)
fc_sim = fcn @ fcn.T

wn_cache: dict = {}


def wn_row(t: int, kind: str) -> torch.Tensor:
    if (t, kind) not in wn_cache:
        f = syn[t].wup_similarity if kind == "wup" else syn[t].path_similarity
        wn_cache[(t, kind)] = torch.tensor([float(f(s) or 0.0) for s in syn])
    return wn_cache[(t, kind)]


def row(measure: str, t: int) -> torch.Tensor:
    if measure in ("wup", "path"):
        return wn_row(t, measure)
    return (clip_sim if measure == "clip" else fc_sim)[t]


def rank_of(r: torch.Tensor, t: int, c: int) -> float:
    """1-based rank of c among the 999 non-true classes, most similar first; ties get the mean rank."""
    v = r.clone()
    v[t] = float("nan")
    others = torch.cat([v[:t], v[t + 1:]])
    higher = int((others > v[c]).sum())
    equal = int((others == v[c]).sum())  # includes c itself
    return higher + (equal + 1) / 2


data = json.loads((repo / "results" / "competitor-classes-dev.json").read_text())
rows = data["rows"]
MEASURES = ["wup", "path", "clip", "fc"]
KS = [1, 5, 10, 20, 50, 100]
out = {"n_images": 0, "best_class": {}, "within_image": {}, "examples": []}

ranks = {m: [] for m in MEASURES}
logit_ranks = []
for r in rows:
    t, c = r["true_index"], r["best_class"]
    if c is None or c == t:
        continue
    out["n_images"] += 1
    logit_ranks.append(r["best_rank"])
    for m in MEASURES:
        ranks[m].append(rank_of(row(m, t), t, c))
    if len(out["examples"]) < 12:
        out["examples"].append({"true": names[t], "best": names[c], "clean_rank": r["best_rank"],
                                **{m: ranks[m][-1] for m in MEASURES}})



def summarise(rs):
    return {"median_rank": st.median(rs), "mean_pct": round(100 * st.mean(x / 999 for x in rs), 1),
            **{f"hit@{k}": sum(x <= k for x in rs) for k in KS}}


out["best_class"]["clean_logit_rank"] = summarise(logit_ranks)  # best_rank: 1-based among non-true classes
for m in MEASURES:
    out["best_class"][m] = summarise(ranks[m])
out["best_class"]["random_expectation"] = {"median_rank": 500, "mean_pct": 50.0,
                                           **{f"hit@{k}": round(out["n_images"] * k / 999, 2) for k in KS}}


def spearman(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for p, i in enumerate(o):
            r[i] = p
        return r
    ra, rb = rk(a), rk(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


for m in MEASURES:
    out["best_class"][m]["spearman_vs_clean_rank"] = round(spearman(ranks[m], logit_ranks), 3)

# Within image: for classes reached by different competitor responses, does the more similar class score higher?
for m in MEASURES + ["clean_rank"]:
    agree = total = 0
    for r in rows:
        t = r["true_index"]
        best_per_class: dict[int, tuple[float, int]] = {}
        for a in r["advs"]:
            if a.get("class") is None or a["class"] == t or not a.get("official"):
                continue
            prev = best_per_class.get(a["class"])
            if prev is None or a["official"] > prev[0]:
                best_per_class[a["class"]] = (a["official"], a["clean_rank"])
        cls = list(best_per_class)
        for i in range(len(cls)):
            for j in range(i + 1, len(cls)):
                ci, cj = cls[i], cls[j]
                si, sj = best_per_class[ci][0], best_per_class[cj][0]
                if abs(si - sj) < 1e-6:
                    continue
                if m == "clean_rank":
                    vi, vj = -best_per_class[ci][1], -best_per_class[cj][1]
                else:
                    rw = row(m, t)
                    vi, vj = float(rw[ci]), float(rw[cj])
                if vi == vj:
                    continue
                total += 1
                agree += (vi > vj) == (si > sj)
    out["within_image"][m] = {"pairs": total, "concordance": round(agree / total, 3) if total else None}

print(json.dumps(out, indent=1))

# Complementarity: images the clean-logit top-20 misses, caught by a semantic top-20?
miss = [i for i, lr in enumerate(logit_ranks) if lr > 20]
comp = {m: sum(ranks[m][i] <= 20 for i in miss) for m in MEASURES}
union = {m: sum(lr <= 10 or ranks[m][i] <= 10 for i, lr in enumerate(logit_ranks)) for m in MEASURES}
print("COMPLEMENT", json.dumps({"logit_top20_misses": len(miss), "caught_by_semantic_top20": comp,
                                "logit_top20_alone": sum(lr <= 20 for lr in logit_ranks),
                                "logit_top10_or_semantic_top10": union}))
