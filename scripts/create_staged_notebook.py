"""Build a self-contained Kaggle notebook; no source-dataset re-upload required."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build_notebook(destination: Path) -> Path:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "scripts", "configs", "tests"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".yaml"):
                    archive.write(path, path.relative_to(ROOT).as_posix())
        archive.write(ROOT / "pytest.ini", "pytest.ini")
    payload = base64.b64encode(buffer.getvalue()).decode()
    cells = []
    def markdown(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text))
    def code(text):
        cells.append(dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=textwrap.dedent(text).strip() + "\n"))
    markdown("# U-Net: chạy từng epoch và tải checkpoint\n\n"
             "Gắn dataset **BRaTS 2021 Task 1 Dataset**, chọn **T4 ×2**. Dừng phiên huấn luyện cũ trước khi chạy. "
             "Notebook này chứa sẵn source mới. **Không dùng Run All.**\n\n"
             "Thứ tự: 1 Thiết lập → 2 Chuẩn bị → 3 Chạy một epoch, tải ZIP; lặp ô 3 cho đến đủ epoch → 4 Đánh giá. "
             "Giữ nguyên tất cả bệnh nhân và train/val/test. 5 epoch là baseline; cần xem Dice/IoU và ảnh dự đoán để kết luận chất lượng.\n\n"
             "Nếu sang session mới, upload ZIP backup dưới dạng Input; ô 1 sẽ tự tìm và khôi phục checkpoint mới nhất. "
             "Checkpoint lưu sau train và sau validation. Ngắt giữa train thì phải chạy lại phần train của epoch; "
             "ngắt trong validation có thể tiếp tục từ đầu validation nếu train_pending.pt đã lưu.")
    code('''
        from pathlib import Path
        import os, sys, subprocess, shutil, zipfile, io, base64, importlib.util, signal

        # Tự tìm backup đã gắn trong Input. Nếu Kaggle giữ nguyên ZIP thì giải nén
        # vào working; nếu Kaggle đã giải nén thì dùng trực tiếp thư mục đó.
        RESTORE_FROM = ""
        backup_checkpoints = sorted(Path("/kaggle/input").rglob("models/last_model.pt"))
        if not backup_checkpoints:
            backup_archives = sorted(Path("/kaggle/input").rglob("brain-tumor-backup-epoch-*.zip"))
            if backup_archives:
                restore_unpack = Path("/kaggle/working/brain-tumor-restore")
                restore_unpack.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(backup_archives[-1]) as backup_archive:
                    restore_root = restore_unpack.resolve()
                    for member in backup_archive.infolist():
                        target = (restore_unpack / member.filename).resolve()
                        assert target == restore_root or restore_root in target.parents, "ZIP backup không hợp lệ"
                        backup_archive.extract(member, restore_unpack)
                backup_checkpoints = sorted(restore_unpack.rglob("models/last_model.pt"))
        if backup_checkpoints:
            RESTORE_FROM = str(backup_checkpoints[-1].parents[1])
            print("Khôi phục checkpoint từ:", RESTORE_FROM, flush=True)
        else:
            print("Không có backup trong Input; sẽ bắt đầu từ epoch 1.", flush=True)
        project = Path("/kaggle/working/brain-tumor-mri-unet")
        project.mkdir(parents=True, exist_ok=True)
        missing = [package for module, package in [("nibabel", "nibabel"), ("yaml", "PyYAML"), ("tqdm", "tqdm"), ("pytest", "pytest"), ("psutil", "psutil")]
                   if importlib.util.find_spec(module) is None]
        if missing:
            subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        import torch, psutil
        assert torch.cuda.is_available(), "Bật GPU T4 trước khi chạy"
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = process.info["cmdline"] or []
                if process.pid != os.getpid() and any(token == "src.training.train" or token.endswith("/staged_pipeline.py") or token.endswith("/finish_project.py") for token in command):
                    raise RuntimeError(f"Có tiến trình cũ PID {process.pid}; kiểm tra/dừng phiên cũ trước.")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        print("GPU:", torch.cuda.get_device_name(0), "Số GPU:", torch.cuda.device_count(), flush=True)
        print("CUDA calculation:", torch.ones(2, device="cuda").sum().item(), flush=True)
    ''')
    cells[-1]["source"] += f'\nSOURCE_ZIP = "{payload}"\n'
    cells[-1]["source"] += textwrap.dedent('''
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ZIP))) as archive:
            for member in archive.infolist():
                target = (project / member.filename).resolve()
                assert project.resolve() in target.parents
                if member.filename.startswith("configs/") and target.exists():
                    continue  # Preserve a tuned config or previous run.
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        if RESTORE_FROM:
            restore = Path(RESTORE_FROM)
            assert (restore / "models/last_model.pt").is_file() or (restore / "models/train_pending.pt").is_file()
            assert (restore / "configs/kaggle_staged.yaml").is_file()
            assert not any((project / "models" / name).exists() for name in ("last_model.pt", "train_pending.pt")), "Có checkpoint trong session; không ghi đè"
            for folder in ("models", "outputs", "data", "configs"):
                if (restore / folder).exists():
                    shutil.copytree(restore / folder, project / folder, dirs_exist_ok=True)
        os.chdir(project)
        os.environ["PYTHONPATH"] = str(project)
        def run_stage(action):
            child = subprocess.Popen([sys.executable, "-u", "scripts/staged_pipeline.py", action],
                                     cwd=project, start_new_session=True)
            try:
                result = child.wait()
            except KeyboardInterrupt:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
                raise
            if result:
                raise RuntimeError(f"Stage exit={result}. Xem outputs/logs/staged-console.log; không bấm chạy lại liên tục.")
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=project, check=True)
        print("SETUP COMPLETE")
    ''')
    markdown("## 2. Chuẩn bị (một lần mỗi session)\nCó dữ liệu và split hợp lệ thì giữ nguyên; thiếu MRI mới giải nén lại.")
    code('run_stage("prepare")')
    markdown("## 3. Chạy đúng MỘT epoch\nÔ này tự dừng sau train + validation và tạo `brain-tumor-backup-epoch-XXX.zip` "
             "trong Output `/kaggle/working`. **Tải ZIP về laptop sau mỗi chặng.** "
             "Thấy `STAGE COMPLETE` mới chạy lại ô này để học epoch kế tiếp. Không bấm lúc chỉ thấy `Train: 100%`. "
             "Đến `Run FINISH` thì chuyển ô 4. Nếu lỗi, lấy `outputs/logs/staged-console.log` để kiểm tra.")
    code('run_stage("train-next")')
    markdown("## 4. Tìm threshold trên validation và đánh giá test\nKhông huấn luyện lại. "
             "Sau `PROJECT COMPLETE`, tải `brain-tumor-results.zip`. Giải nén vào project local "
             "(sao lưu config cũ trước), chạy `streamlit run app/app.py`, thử ảnh MRI FLAIR và kiểm tra báo cáo test. "
             "Tải xong backup/kết quả mới kết thúc session.")
    code('run_stage("finish")')
    document = dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python", "version": "3.12.0"}}, nbformat=4, nbformat_minor=4)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(build_notebook(ROOT / "notebooks/03_staged_kaggle.ipynb"))
