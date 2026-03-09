# Reproducibility Guide (Windows)

This directory contains Windows-first scripts to launch training, evaluate models, and "freeze" results without relying on WSL or Bash.

## 1. Train Baseline
Run the launcher. It automatically enforces `num_workers > 0` for Windows and handles Hydra path separators.
```powershell
.\scripts\run_train_windows.ps1 -MaxEpochs 40
```
*Note: Make sure your data is in `../data/subject_89335547` (or pass `-DataDir <path>`).*

### Train BiGRU Variant
To train the new BiGRU-CTC model on the single-user split:
```powershell
python -m emg2qwerty.train user=single_user model=bigru_ctc trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=40
```
*(Or adapt the `run_train_windows.ps1` to accept a model argument if desired.)*

## 2. Freeze Run
After training completes, "freeze" the logs and metadata into `results/`. This creates small, committable artifacts instead of huge checkpoints.
```powershell
python scripts/freeze_run.py
```
This extracts validation/test metrics, links the best checkpoint, and updates `results/BASELINE_LATEST.md`.

## 3. Evaluate Checkpoint
If you want to explicitly run the test set on a specific checkpoint:
```powershell
.\scripts\eval_ckpt_windows.ps1 -CkptPath "path/to/model.ckpt"
```
The metrics will be logged and saved into `results/evals/`.

## Important Notes
- **Do not commit large artifacts.** The `.gitignore` has been updated to exclude `data/`, `logs/`, `outputs/`, `lightning_logs/`, and large files like `.ckpt`, `.hdf5`, `.h5`, `.zip`.
- Checkpoints and large logs must remain local to your machine. Always use the freeze script to summarize your run for report writing and teamwork.
