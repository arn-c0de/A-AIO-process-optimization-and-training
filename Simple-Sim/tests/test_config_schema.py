from simple_sim.config_schema import parse_config_typed


def test_parse_config_typed_minimal() -> None:
    cfg = {
        "run": {"run_id": "r1", "seed": 42, "schema_version": 2, "mode": ""},
        "roi": {"width_px": 128, "height_px": 128, "mm_per_px": 0.05},
        "train": {
            "model": "resnet18",
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.001,
            "optimizer": "adam",
            "weight_decay": 0.0,
        },
        "eval": {"batch_size": 4},
    }
    typed = parse_config_typed(cfg)
    assert typed.run.run_id == "r1"
    assert typed.train.model == "resnet18"
