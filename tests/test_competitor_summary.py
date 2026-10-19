"""tools.competitor_scores.summarize: per-variant aggregation of competitor responses."""
from tools.competitor_scores import summarize


def _response(file, scores, margins):
    variants = {n: {"score": s, "reason": "ok" if s > 0 else "not_adversarial", "margin": m}
                for n, s, m in zip(("default", "tf32_off", "tf32_on"), scores, margins)}
    return {"file": file, "official": min(scores), "min_margin": min(margins),
            "variant_disagreement": len({s > 0 for s in scores}) > 1, "variants": variants}


def test_summarize_counts_disagreements_and_thin_margins():
    rows = [
        {"key": "a", "best": 0.98, "scores": [
            _response("a0", (0.98, 0.98, 0.98), (12.0, 11.5, 12.1)),
            _response("a1", (0.95, 0.0, 0.95), (0.02, -0.01, 0.03)),
        ]},
        {"key": "b", "best": 0.97, "scores": [
            _response("b0", (0.97, 0.971, 0.97), (0.5, 0.6, 0.4)),
        ]},
    ]
    out = summarize(rows)
    assert out["images"] == 2 and out["adversarials"] == 3
    assert out["variant_disagreements"] == 1
    assert out["variant_disagreement_files"] == ["a1"]
    assert out["best_with_disagreement"] == 0
    assert out["best_min_margin"]["min"] == 0.4
    assert out["best_min_margin"]["max"] == 11.5
    assert out["passing_margin_below_1"] == 1  # a1 fails officially, so only b0 counts
    assert abs(out["max_variant_score_spread"] - 0.95) < 1e-12
    assert abs(out["mean_best_per_image"] - 0.975) < 1e-12


def test_summarize_empty():
    out = summarize([])
    assert out["images"] == 0 and out["best_min_margin"] == {} and out["max_variant_score_spread"] == 0.0
