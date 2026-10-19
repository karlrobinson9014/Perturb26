from __future__ import annotations

from miner26.neuron import _next_boundary, _summarize_rounds


def test_next_boundary_is_strictly_after_now():
    assert _next_boundary(1000.0, 120) == 1080.0
    assert _next_boundary(1080.0, 120) == 1200.0
    assert _next_boundary(1199.9, 120) == 1200.0


def test_summarize_rounds_skips_missed_rounds():
    rounds = [
        {"official": 0.97, "done_since_boundary": 27.0},
        {"official": None},
        {"official": 0.0, "done_since_boundary": 29.5},
    ]
    s = _summarize_rounds(rounds)
    assert s["rounds"] == 3 and s["scored"] == 2 and s["success"] == 1
    assert abs(s["mean_official"] - 0.485) < 1e-12
    assert s["min_official"] == 0.0 and s["max_done_since_boundary"] == 29.5


def test_summarize_rounds_empty():
    s = _summarize_rounds([])
    assert s["scored"] == 0 and s["mean_official"] is None and s["max_done_since_boundary"] is None


def test_miner_logging_survives_bittensor_muting(capfd):
    import logging

    from miner26.neuron import _setup_miner_logging

    log = logging.getLogger("miner26")
    log.setLevel(logging.CRITICAL)  # what bittensor's default logging state does to non-bittensor loggers
    for h in list(log.handlers):
        log.removeHandler(h)
    _setup_miner_logging()
    _setup_miner_logging()  # idempotent: one handler, one line
    log.info("probe-line")
    assert capfd.readouterr().err.count("probe-line") == 1
