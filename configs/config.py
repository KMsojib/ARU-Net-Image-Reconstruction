"""
Central configuration for the ARU-Net image reconstruction project.

Values here are taken directly from the executed `__main__` block of the
original notebook (`4_AttentionResidualUNet.ipynb`), NOT from Table 3.1 of
the thesis, wherever the two disagree. See docs/reproducibility.md and the
Research Implementation Audit in the top-level README for a full list of
discrepancies between the thesis text and the actual code.

All paths default to a repo-relative `data/` layout instead of the original
hard-coded Google Drive paths (e.g. `/content/drive/MyDrive/FrameReconstruction/...`).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    # Name of the degradation folder being trained/evaluated on, e.g.
    # "Noise_Multiply_Strong", "Gaussian_Blur", "Noise_Half_Split", etc.
    # This mirrors `dataset_name` in the notebook's __main__ block.
    dataset_name: str = "Noise_Multiply_Strong"

    # Root data directory (replaces the hard-coded Google Drive mount).
    data_root: Path = Path("data")

    @property
    def distorted_dir(self) -> Path:
        return self.data_root / "distorted" / self.dataset_name

    @property
    def clean_dir(self) -> Path:
        return self.data_root / "clean"

    @property
    def output_dir(self) -> Path:
        return Path("results") / "reconstructions" / self.dataset_name

    @property
    def checkpoint_dir(self) -> Path:
        return Path("checkpoints")

    @property
    def metrics_dir(self) -> Path:
        return Path("results") / "metrics"


@dataclass
class DatasetConfig:
    # Filename-suffix matching scheme used by ImagePairDataset in the
    # notebook: a distorted file's suffix is replaced with the clean
    # suffix to find its paired ground-truth image.
    dist_suffix: str = "_noise4_multiply_strong"
    clean_suffix: str = "_original"
    allowed_extensions: tuple = (".jpg", ".jpeg", ".png")

    # Split fractions actually used in the notebook's __main__ block
    # (80/10/10 train/val/test). NOTE: the thesis text (Sec. 3.3) only
    # describes an 80/20 train/validation split; the held-out test split
    # is a notebook-only detail, preserved here as the source of truth
    # for the *implementation*.
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    # test_fraction is implied: 1 - train_fraction - val_fraction

    split_seed: int = 42


@dataclass
class AugmentationConfig:
    # Matches `train_transform` in the notebook exactly. The thesis
    # (Sec. 3.3) only mentions "random flipping and rotation" -- the
    # random-resized-crop and color-jitter steps are notebook-only
    # additions, kept here as the implementation's source of truth.
    train_crop_size: int = 224
    random_rotation_degrees: float = 5.0
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1


@dataclass
class ModelConfig:
    in_channels: int = 3
    out_channels: int = 3
    # Encoder/decoder channel widths, matching Fig. 3.13/3.14 of the
    # thesis and the notebook's AttentionResidualUNet exactly.
    encoder_channels: tuple = (64, 128, 256, 512)
    bottleneck_channels: int = 1024


@dataclass
class LossConfig:
    """
    Weights for the composite loss actually implemented in the notebook's
    CombinedLoss class:

        L_total = 0.4 * L1
                + 0.25 * VGG16-perceptual (features[:16])
                + 0.20 * Laplacian edge loss
                + 0.15 * (1 - masked SSIM)

    The thesis (Sec. 3.4) only documents L_total = alpha*L1 + beta*(1-SSIM)
    with unspecified alpha/beta, and never mentions the perceptual or edge
    terms. These weights are taken from the notebook, which is the only
    source that specifies concrete numbers.
    """
    w_pixel_l1: float = 0.4
    w_perceptual: float = 0.25
    w_edge: float = 0.2
    w_ssim: float = 0.15
    use_vgg_perceptual: bool = True
    vgg_feature_layers: int = 16  # vgg16(pretrained=True).features[:16]


@dataclass
class TrainConfig:
    # The notebook's train() function default is epochs=70, but the
    # executed __main__ block overrides this to epochs=150. We use 150
    # here since that is what was actually run.
    epochs: int = 150
    learning_rate: float = 1e-4
    batch_size_train: int = 4
    batch_size_eval: int = 1
    optimizer: str = "adam"

    # StepLR scheduler as used in the notebook's __main__ block
    # (note: this differs from the train() function's own internal
    # default of step_size=10 -- __main__'s scheduler object, built with
    # step_size=15, is the one actually passed in and used).
    lr_scheduler_step_size: int = 15
    lr_scheduler_gamma: float = 0.5

    # Early stopping: the thesis (Sec. 3.3) states early stopping is
    # applied, and the notebook's __main__ block passes `patience=20` to
    # train(). However, the notebook's train() function does NOT define
    # a `patience` parameter or implement early-stopping logic --
    # calling it as shown in the notebook would raise a TypeError. This
    # repository implements the early-stopping behavior the thesis
    # describes (monitoring validation loss with the given patience),
    # since the original code cannot execute as written. This is a
    # software correction, not a change to the reported science.
    early_stopping_patience: int = 20

    use_mixed_precision: bool = True  # only active when device == "cuda"
    num_workers: int = 2
    random_seed: int = 42


@dataclass
class EvalConfig:
    # Minimum leaf-mask pixel count below which masked metrics fall back
    # to whole-image metrics (matches the notebook's `mask_i.sum() < 49`
    # i.e. a 7x7 pixel threshold).
    min_mask_pixels: int = 49
    min_crop_side: int = 7
    ssim_win_size: int = 7


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def get_config() -> Config:
    """Returns the default configuration. Override fields as needed, e.g.:

        cfg = get_config()
        cfg.paths.dataset_name = "Gaussian_Blur"
        cfg.train.epochs = 50
    """
    return Config()
