# Checkpoints

Trained model weights are not committed to this repository (large binary
files are better hosted via GitHub Releases, Git LFS, or a cloud bucket).

## What gets saved

`scripts/train.py` saves **one final checkpoint** at the end of training:

```
checkpoints/{dataset_name}_AttentionResidualUNet_final.pth
```

This matches the original notebook's behavior exactly: the notebook's
`__main__` block only calls `torch.save(model.state_dict(), final_path)`
once, after the full training loop completes. **No periodic or
best-validation checkpointing exists in the source notebook**, so none is
fabricated here either -- if you want best-checkpoint selection, you will
need to add it explicitly (this would be a genuine enhancement beyond the
original implementation, not a faithful port of it).

## Loading a checkpoint

```python
import torch
from src.model import AttentionResidualUNet

model = AttentionResidualUNet()
model.load_state_dict(torch.load("checkpoints/Noise_Multiply_Strong_AttentionResidualUNet_final.pth", map_location="cpu"))
model.eval()
```

## Distributing large weight files

If you want to publish trained weights publicly, prefer attaching them to a
GitHub Release rather than committing them to version control:

```bash
gh release create v1.0.0 checkpoints/Noise_Multiply_Strong_AttentionResidualUNet_final.pth \
    --title "ARU-Net v1.0.0 checkpoint" \
    --notes "Trained on the Noise_Multiply_Strong degradation split."
```
