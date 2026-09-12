from pathlib import Path

import torch

from scripts.finish_project import checkpoint_epoch, package_results


def test_checkpoint_epoch_and_result_package(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "last_model.pt"
    torch.save({"epoch": 3, "model_state_dict": {}}, checkpoint)
    assert checkpoint_epoch(checkpoint) == 3
    assert checkpoint_epoch(tmp_path / "missing.pt") is None

    project = tmp_path / "project"
    (project / "models").mkdir(parents=True)
    (project / "outputs" / "evaluation").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "models" / "best_model.pt").write_bytes(b"model")
    (project / "outputs" / "evaluation" / "summary.json").write_text(
        '{"dice": 0.5}', encoding="utf-8"
    )
    config = project / "configs" / "kaggle_fast.yaml"
    config.write_text("test: true\n", encoding="utf-8")

    import scripts.finish_project as module

    monkeypatch.setattr(module, "ROOT", project)
    output = package_results(config, tmp_path / "results.zip")
    assert output.is_file()
    import zipfile

    with zipfile.ZipFile(output) as archive:
        assert "models/best_model.pt" in archive.namelist()
        assert "outputs/evaluation/summary.json" in archive.namelist()
        assert archive.read("configs/config.yaml").decode("utf-8").splitlines() == ["test: true"]
