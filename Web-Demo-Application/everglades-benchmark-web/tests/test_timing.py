from everglades_core.timing import timed_forecast


def test_timed_forecast_returns_first_result_and_median():
    calls = []

    def fn():
        calls.append(1)
        return len(calls)

    result, t = timed_forecast(fn, repeats=5)
    assert result == 1 and len(calls) == 6
    assert t["repeats"] == 5 and len(t["forecast_seconds_samples"]) == 5
    assert t["forecast_seconds"] >= 0
