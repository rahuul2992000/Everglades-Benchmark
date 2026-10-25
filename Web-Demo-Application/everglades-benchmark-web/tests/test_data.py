from everglades_core import config, data


def test_split_matches_paper():
    Y = data.load_long()
    s = data.split(Y)
    assert Y["unique_id"].nunique() == 37
    assert (s.val_size, s.test_size) == (211, 211)
    assert s.Y_fit_df["ds"].nunique() == 1200
    assert str(s.test_start.date()) == "2024-01-29"
    assert str(s.test_end.date()) == "2024-08-26"


def test_model_registry():
    assert len(config.MODELS) == 12
    assert config.models_for("chronos") == ["Chronos-Bolt"]
    es = {m: s.early_stop for m, s in config.MODELS.items() if s.family == "nf"}
    assert {m for m, v in es.items() if v == 100} == {
        "TimeMixer", "TSMixer", "TSMixerx", "iTransformer"}
    assert all(v == 50 for m, v in es.items()
               if m not in {"TimeMixer", "TSMixer", "TSMixerx", "iTransformer"})
    inf = config.MODELS["Informer"].nf_kwargs(7)
    assert inf["hidden_size"] == 256 and inf["conv_hidden_size"] == 64
    assert config.MODELS["RMoK"].nf_kwargs(7)["n_series"] == 37
    assert "n_series" not in config.MODELS["NBEATS"].nf_kwargs(7)
