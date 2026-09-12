"""Real CPU control-flow tests on small synthetic MRI, not accuracy evidence."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from src.data.dataset import BraTSSliceDataset
from src.data.discovery import build_metadata
from test_dataset_pipeline import _write_patient


def fixture_config(tmp_path):
    root = tmp_path / "mri"
    root.mkdir()
    for i in range(3):
        _write_patient(root, f"BraTS2021_{i:05d}", 2)
    metadata = tmp_path / "metadata.csv"
    build_metadata(root, metadata)
    splits = tmp_path / "splits"
    splits.mkdir()
    for i, name in enumerate(("train", "val", "test")):
        (splits / f"{name}.txt").write_text(f"brats2021_{i:05d}\n")
    config = yaml.safe_load(Path("configs/kaggle_fast.yaml").read_text())
    config["data"].update(metadata_csv=str(metadata), splits_dir=str(splits), target_size=32, index_cache_dir=str(tmp_path / "index"))
    config["model"]["base_channels"] = 4
    config["training"].update(epochs=2, batch_size=2, num_workers=0, mixed_precision=False, multi_gpu=False)
    config["training"]["early_stopping"]["enabled"] = False
    config["paths"].update(output_dir=str(tmp_path / "outputs"), model_dir=str(tmp_path / "models"), best_checkpoint=str(tmp_path / "models/best.pt"), last_checkpoint=str(tmp_path / "models/last.pt"))
    return config


def test_slice_index_cache_preserves_samples_and_invalidates(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    kwargs = dict(metadata_csv=config["data"]["metadata_csv"], split_file=Path(config["data"]["splits_dir"]) / "train.txt", modalities=["flair"], target_size=32, split_name="train")
    uncached = BraTSSliceDataset(**kwargs)
    cached = BraTSSliceDataset(**kwargs, index_cache_dir=tmp_path / "index")
    assert cached.samples == uncached.samples
    assert cached.positive_sample_keys == uncached.positive_sample_keys
    import src.data.dataset as module
    real_load = module.load_nifti
    def reject_load(*args):
        raise AssertionError("A cached index must not decompress MRI")
    monkeypatch.setattr(module, "load_nifti", reject_load)
    assert BraTSSliceDataset(**kwargs, index_cache_dir=tmp_path / "index").samples == uncached.samples
    next((tmp_path / "index").glob("*.json")).write_text("broken")
    with pytest.raises(AssertionError):
        BraTSSliceDataset(**kwargs, index_cache_dir=tmp_path / "index")
    monkeypatch.setattr(module, "load_nifti", real_load)
    BraTSSliceDataset(**kwargs, index_cache_dir=tmp_path / "index")
    source = Path(cached.metadata.iloc[0]["flair_path"])
    import os
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    monkeypatch.setattr(module, "load_nifti", reject_load)
    with pytest.raises(AssertionError):
        BraTSSliceDataset(**kwargs, index_cache_dir=tmp_path / "index")


def test_one_epoch_then_resume_completes_second_epoch(tmp_path):
    config = fixture_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    # Thread count is constrained only for this tiny CPU test.
    import os
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    command = [sys.executable, "-m", "src.training.train", "--config", str(config_path), "--epochs-this-run", "1"]
    subprocess.run(command, check=True, capture_output=True, text=True, env=env, timeout=90)
    last = config["paths"]["last_checkpoint"]
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert len(checkpoint["history"]) == 1
    subprocess.run([*command, "--resume", last], check=True, capture_output=True, text=True, env=env, timeout=90)
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert [row["epoch"] for row in checkpoint["history"]] == [1, 2]


def test_resume_pending_train_skips_training(tmp_path):
    config = fixture_config(tmp_path)
    config["paths"]["pending_checkpoint"] = str(tmp_path / "models/pending.pt")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    import os
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    command = [sys.executable, "-m", "src.training.train", "--config", str(path), "--epochs-this-run", "1"]
    subprocess.run(command, check=True, capture_output=True, text=True, env=env, timeout=90)
    pending = config["paths"]["pending_checkpoint"]
    state = torch.load(pending, map_location="cpu", weights_only=False)
    assert state["epoch"] == 0 and state["pending_train_metrics"] is not None
    result = subprocess.run([*command, "--resume", pending], check=True, capture_output=True, text=True, env=env, timeout=90)
    assert "restored completed TRAIN; continuing VALIDATION" in result.stderr
    assert "TRAIN starting" not in result.stderr
    assert torch.load(config["paths"]["last_checkpoint"], map_location="cpu", weights_only=False)["epoch"] == 1


def test_generated_notebook_cells_compile(tmp_path):
    from scripts.create_staged_notebook import build_notebook
    path = build_notebook(tmp_path / "staged.ipynb")
    document = json.loads(path.read_text(encoding="utf-8"))
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "<notebook>", "exec")


def test_stage_dispatch_prefers_pending_checkpoint_and_packages_backup(tmp_path, monkeypatch):
    from scripts import staged_pipeline as staged
    from scripts import finish_project
    config = fixture_config(tmp_path)
    config["paths"].update(best_checkpoint="models/best_model.pt", last_checkpoint="models/last_model.pt", pending_checkpoint="models/train_pending.pt")
    config_path = tmp_path / "configs/kaggle_staged.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config))
    (tmp_path / "models").mkdir()
    pending = tmp_path / "models/train_pending.pt"
    torch.save({"epoch": 0, "pending_train_metrics": {"loss": 0.8}}, pending)
    monkeypatch.setattr(staged, "ROOT", tmp_path)
    monkeypatch.setattr(staged, "CONFIG", config_path)
    monkeypatch.setattr(finish_project, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.chdir(tmp_path)
    calls = []
    def fake_run(*args):
        calls.append(args)
        for name in ("last_model.pt", "best_model.pt"):
            torch.save({"epoch": 1}, tmp_path / "models" / name)
    monkeypatch.setattr(staged, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["staged_pipeline.py", "train-next"])
    staged.main()
    assert calls[0][-2:] == ("--resume", str(pending))
    import zipfile
    with zipfile.ZipFile(tmp_path.parent / "brain-tumor-backup-epoch-001.zip") as archive:
        assert {"models/last_model.pt", "models/train_pending.pt", "configs/kaggle_staged.yaml"} <= set(archive.namelist())
        assert not any("mri/" in name for name in archive.namelist())
    monkeypatch.setattr(sys, "argv", ["staged_pipeline.py", "finish"])
    with pytest.raises(RuntimeError, match="Only 1/2"):
        staged.main()
