"""Run one resumable epoch at a time, with durable logs and downloadable backups."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import zipfile

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.finish_project import package_results

CONFIG = ROOT / "configs/kaggle_staged.yaml"


def completed(config: dict) -> tuple[int, bool]:
    path = ROOT / config["paths"].get("last_checkpoint", "models/last_model.pt")
    if not path.is_file():
        return 0, False
    state = torch.load(path, map_location="cpu", weights_only=False)
    early = config["training"].get("early_stopping", {})
    stopped = bool(early.get("enabled", True)) and int(state.get("early_stopping_bad_epochs", 0)) >= int(early.get("patience", 10))
    return int(state["epoch"]), stopped


def backup(config: dict) -> Path:
    epoch, _ = completed(config)
    destination = ROOT.parent / f"brain-tumor-backup-epoch-{epoch:03d}.zip"
    temporary = destination.with_suffix(".zip.tmp")
    package_results(CONFIG, temporary)
    with zipfile.ZipFile(temporary, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.write(CONFIG, "configs/kaggle_staged.yaml")
        pending = ROOT / config["paths"].get("pending_checkpoint", "models/train_pending.pt")
        if pending.is_file():
            archive.write(pending, pending.relative_to(ROOT).as_posix())
        for entry in (ROOT / config["data"]["metadata_csv"], *sorted((ROOT / config["data"]["splits_dir"]).glob("*.txt"))):
            if entry.is_file():
                archive.write(entry, entry.relative_to(ROOT).as_posix())
    temporary.replace(destination)
    print(f"BACKUP: {destination} -- download this ZIP before ending the session.", flush=True)
    return destination


def run(*args: str) -> None:
    """Persist stdout/stderr and report a heartbeat while the child is quiet."""
    log_path = ROOT / "outputs/logs/staged-console.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", *args]
    print(">>> " + " ".join(command), flush=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    lines: queue.Queue = queue.Queue()
    def read_output():
        try:
            for line in child.stdout:
                lines.put(line)
        finally:
            lines.put(None)
    threading.Thread(target=read_output, daemon=True).start()
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n{time.ctime()} START {command}\n")
            while True:
                try:
                    line = lines.get(timeout=30)
                except queue.Empty:
                    line = f"[heartbeat] PID={child.pid}, process alive={child.poll() is None}; waiting for output\n"
                    try:
                        import psutil
                        memory = psutil.virtual_memory()
                        line = line.rstrip() + f"; RAM used={memory.percent}%, available={memory.available / 1024**3:.1f} GiB\n"
                    except ImportError:
                        pass
                if line is None:
                    break
                print(line, end="", flush=True)
                log.write(line)
            result = child.wait()
            log.write(f"EXIT CODE: {result}\n")
        if result:
            raise RuntimeError(f"Stage exited with code {result}; inspect {log_path}. Do not repeatedly restart.")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        child.stdout.close()


def initialize() -> dict:
    if CONFIG.exists():
        return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    # Preserve the architecture and sampling of an existing full-training checkpoint.
    last = ROOT / "models/last_model.pt"
    if last.exists():
        config = torch.load(last, map_location="cpu", weights_only=False)["config"]
    else:
        config = yaml.safe_load((ROOT / "configs/kaggle_fast.yaml").read_text(encoding="utf-8"))
    config["training"].update(batch_size=32, num_workers=0, pin_memory=False)
    config["data"]["index_cache_dir"] = "outputs/slice-index"
    config["paths"]["pending_checkpoint"] = "models/train_pending.pt"
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train-next", "finish", "status"])
    args = parser.parse_args()
    os.chdir(ROOT)
    config = initialize()
    epoch, stopped = completed(config)
    target = int(config["training"]["epochs"])
    print(f"Completed epochs: {epoch}/{target}; early stopping: {stopped}", flush=True)
    if args.action == "status":
        if epoch or (ROOT / config["paths"].get("pending_checkpoint", "models/train_pending.pt")).exists():
            backup(config)
        return
    if args.action == "prepare":
        archives = sorted(Path("/kaggle/input").rglob("BraTS2021_Training_Data.tar"))
        metadata = ROOT / config["data"]["metadata_csv"]
        if not metadata.exists():
            if not archives:
                raise RuntimeError("Attach the BRaTS 2021 Task 1 Dataset first.")
            run("scripts/prepare_dataset.py", "--config", str(CONFIG), "--archive", str(archives[0]))
        else:
            # A metadata CSV can survive a restore while the actual MRI files do not.
            import pandas as pd
            frame = pd.read_csv(metadata)
            missing = any(not Path(value).is_file() for column in ["seg_path", *[f"{m}_path" for m in config["data"]["modalities"]]] for value in frame[column])
            if missing:
                if not archives:
                    raise RuntimeError("MRI files are missing; attach the BraTS dataset.")
                run("scripts/prepare_dataset.py", "--config", str(CONFIG), "--archive", str(archives[0]), "--force-extract")
        paths = [ROOT / config["data"]["splits_dir"] / f"{name}.txt" for name in ("train", "val", "test")]
        if not any(path.exists() for path in paths):
            run("scripts/create_splits.py", "--config", str(CONFIG))
        elif not all(path.exists() for path in paths):
            raise RuntimeError("Partial split files: restore all three original splits before training.")
        print("PREPARE COMPLETE. Run the train-next cell once.", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a compatible CUDA GPU before training/evaluation.")
    if args.action == "train-next":
        if epoch >= target or stopped:
            print("Training is finished. Run the FINISH cell.", flush=True)
            if epoch:
                backup(config)
            return
        command = ["-m", "src.training.train", "--config", str(CONFIG), "--epochs-this-run", "1"]
        resume_path = ROOT / config["paths"].get("last_checkpoint", "models/last_model.pt")
        pending_path = ROOT / config["paths"].get("pending_checkpoint", "models/train_pending.pt")
        if pending_path.exists():
            pending = torch.load(pending_path, map_location="cpu", weights_only=False)
            if int(pending["epoch"]) == epoch and pending.get("pending_train_metrics") is not None:
                resume_path = pending_path
            del pending
        if resume_path.exists():
            command.extend(["--resume", str(resume_path)])
        try:
            run(*command)
        finally:
            if (ROOT / config["paths"].get("last_checkpoint", "models/last_model.pt")).exists() or pending_path.exists():
                backup(config)
        now, stopped = completed(config)
        print(f"STAGE COMPLETE: epoch {now}/{target}. " +
              ("Run FINISH." if now >= target or stopped else "Download backup; run this cell again for the next epoch."), flush=True)
        return
    if epoch < target and not stopped:
        raise RuntimeError(f"Only {epoch}/{target} completed. Run train-next before final evaluation.")
    best = str(ROOT / config["paths"]["best_checkpoint"])
    run("scripts/find_best_threshold.py", "--config", str(CONFIG), "--checkpoint", best, "--update-config")
    run("-m", "src.evaluation.evaluate", "--config", str(CONFIG), "--checkpoint", best, "--no-save-predictions")
    summary = json.loads((ROOT / "outputs/evaluation/summary.json").read_text())
    import math
    if not isinstance(summary.get("dice"), (float, int)) or not math.isfinite(summary["dice"]):
        raise RuntimeError("Test evaluation did not produce a finite Dice score.")
    result = package_results(CONFIG, ROOT.parent / "brain-tumor-results.zip")
    print("PROJECT COMPLETE\n" + json.dumps(summary, indent=2) + f"\nArtifacts: {result}", flush=True)


if __name__ == "__main__":
    main()
