from tools.lane_probe import classify


def _info(classes, switch):
    return {"lanes": [{"class": c, "est": e} for c, e in classes], "switch": switch}


def test_classify_outcomes_and_winner():
    info = _info([(5, 0.97), (7, 0.975), (5, 0.96), (None, 0.0)],
                 [{"lanes": [1, 2], "targets": [7, 9], "leader_class": 5},
                  {"lanes": [3], "targets": [11], "leader_class": 5}])
    rows, winner = classify(info)
    assert [(r["lane"], r["outcome"]) for r in rows] == [(1, "target"), (2, "leader"), (3, "none")]
    assert winner == 1


def test_classify_uses_last_retarget_and_other():
    info = _info([(5, 0.97), (8, 0.95)],
                 [{"lanes": [1], "targets": [7], "leader_class": 5},
                  {"lanes": [1], "targets": [9], "leader_class": 5}])
    rows, winner = classify(info)
    assert rows == [{"lane": 1, "round": 1, "target": 9, "leader_class": 5, "class": 8, "est": 0.95, "outcome": "other"}]
    assert winner == 0
