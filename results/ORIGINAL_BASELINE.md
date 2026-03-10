# Original Baseline (unmodified repo config)

Run with the same config as Calvin Pang's repo: `user=single_user`, `tds_conv_ctc`, `trainer.max_epochs=40`, etc. (no hyperparameter changes).

- **Val CER**: 23.0173
- **Test CER**: 24.6596
- **Val IER**: 7.19982 | **Val DER**: 1.77226 | **Val SER**: 14.0452
- **Test IER**: 6.50529 | **Test DER**: 1.92349 | **Test SER**: 16.2308

**Checkpoint**: `models/original_baseline.ckpt`  
*(Copy the best checkpoint from your terminal run into this path to keep it fixed.)*
