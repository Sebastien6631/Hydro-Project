from __future__ import annotations

import yaml

from previ_r2d2.preprocessing.automate.config import load_physics_config, load_sync_config


def test_load_sync_config_reads_yaml(tmp_path):
    path = tmp_path / "automate_sync.yaml"
    path.write_text(yaml.dump({
        "bonneval_G2": {
            "source_host": "user@host",
            "source_base": "/tmp/user/bonneval/mes",
            "dest_base": "/mnt/nas/bonneval_clean",
            "variables": ["Pression"],
            "corrections": {"Pression": {"multiply": 10, "offset": 0}},
        }
    }))

    config = load_sync_config(path)

    assert config["bonneval_G2"]["source_host"] == "user@host"
    assert config["bonneval_G2"]["corrections"]["Pression"]["multiply"] == 10


def test_load_sync_config_missing_dossier_returns_none(tmp_path):
    path = tmp_path / "automate_sync.yaml"
    path.write_text(yaml.dump({"bonneval_G2": {"source_host": "user@host"}}))

    config = load_sync_config(path)

    assert config.get("dossier_absent") is None


def test_load_physics_config_reads_yaml_with_inf_tokens(tmp_path):
    path = tmp_path / "automate_physics.yaml"
    path.write_text(
        "melles:\n"
        "  Config_deversoir_2:\n"
        "    param_segments_1:\n"
        "      - {seuils: [-.inf, .inf], A: [0.05]}\n"
    )

    config = load_physics_config(path)

    seuils = config["melles"]["Config_deversoir_2"]["param_segments_1"][0]["seuils"]
    assert seuils == [float("-inf"), float("inf")]
