"""
Data Structures Module

Contains core data classes for SCI experiment configuration, metrics, and results
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class ReconFamily(Enum):
    """Reconstruction algorithm family"""
    CIAS_CORE = "CIAS-Core"
    CIAS_CORE_ELP = "CIAS-Core-ELP"
    GAP_NET = "GAP-Net"
    OTHER = "Other"


class UQScheme(Enum):
    """Uncertainty quantification scheme"""
    CONFORMAL = "Conformal"
    ENSEMBLE = "Ensemble"
    NONE = "None"


@dataclass
class ForwardConfig:
    """Forward model configuration"""
    compression_ratio: int
    mask_type: str
    sensor_noise: float
    resolution: Tuple[int, int]
    frame_rate: int = 30


@dataclass
class ReconParams:
    """Reconstruction network parameters"""
    num_stages: int
    num_features: int
    num_blocks: int
    learning_rate: float
    use_physics_prior: bool
    activation: str = "ReLU"


@dataclass
class TrainConfig:
    """Training configuration"""
    batch_size: int
    num_epochs: int
    optimizer: str
    scheduler: str
    early_stopping: bool
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0


@dataclass
class SCIConfiguration:
    """Complete SCI experiment configuration"""
    experiment_id: str
    forward_config: ForwardConfig
    recon_family: ReconFamily
    recon_params: ReconParams
    uq_scheme: UQScheme
    uq_params: Dict[str, Any]
    train_config: TrainConfig
    timestamp: str

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to API request format"""
        return {
            "experiment_id": self.experiment_id,
            "forward_model": {
                "compression_ratio": self.forward_config.compression_ratio,
                "mask_type": self.forward_config.mask_type,
                "sensor_noise": self.forward_config.sensor_noise,
                "resolution": list(self.forward_config.resolution),
                "frame_rate": self.forward_config.frame_rate
            },
            "reconstruction": {
                "family": self.recon_family.value,
                "num_stages": self.recon_params.num_stages,
                "num_features": self.recon_params.num_features,
                "num_blocks": self.recon_params.num_blocks,
                "learning_rate": self.recon_params.learning_rate,
                "use_physics_prior": self.recon_params.use_physics_prior,
                "activation": self.recon_params.activation
            },
            "training": asdict(self.train_config),
            "uncertainty_quantification": {
                "scheme": self.uq_scheme.value,
                "params": self.uq_params
            }
        }


@dataclass
class Metrics:
    """Experiment performance metrics"""
    psnr: float
    ssim: float
    coverage: float
    latency: float
    memory: float
    training_time: float
    convergence_epoch: int = 0


@dataclass
class Artifacts:
    """Experiment output files"""
    checkpoint_path: str
    training_log_path: str
    sample_reconstructions: List[str]
    figure_scripts: List[str]
    metrics_history: Dict[str, list]


@dataclass
class ExperimentResult:
    """Experiment result"""
    experiment_id: str
    config: SCIConfiguration
    metrics: Metrics
    artifacts: Artifacts
    status: str
    error_message: Optional[str] = None
    api_task_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class ReviewResult:
    """Result of plan review"""
    approved: bool
    approved_configs: List[SCIConfiguration]
    feedback: str
    critique: Dict[str, str]  # Map experiment_id to specific critique


import json
import hashlib

class ConfigHasher:
    """Utility class for computing configuration hashes"""

    @staticmethod
    def compute_hash(config: SCIConfiguration) -> str:
        """
        Compute a hash for a configuration based on its key parameters.
        Excludes experiment_id and timestamp as they are unique per run.

        Args:
            config: Experiment configuration

        Returns:
            SHA256 hash string
        """
        # Extract only the parameters that define the experiment
        hashable_dict = {
            "forward": {
                "compression_ratio": config.forward_config.compression_ratio,
                "mask_type": config.forward_config.mask_type,
                "sensor_noise": config.forward_config.sensor_noise,
                "resolution": list(config.forward_config.resolution),
                "frame_rate": config.forward_config.frame_rate,
            },
            "recon": {
                "family": config.recon_family.value if isinstance(config.recon_family, Enum) else config.recon_family,
                "num_stages": config.recon_params.num_stages,
                "num_features": config.recon_params.num_features,
                "num_blocks": config.recon_params.num_blocks,
                "learning_rate": config.recon_params.learning_rate,
                "use_physics_prior": config.recon_params.use_physics_prior,
                "activation": config.recon_params.activation,
            },
            "uq": {
                "scheme": config.uq_scheme.value if isinstance(config.uq_scheme, Enum) else config.uq_scheme,
                "params": config.uq_params,
            },
            "train": {
                "batch_size": config.train_config.batch_size,
                "num_epochs": config.train_config.num_epochs,
                "optimizer": config.train_config.optimizer,
                "scheduler": config.train_config.scheduler,
                "early_stopping": config.train_config.early_stopping,
            }
        }

        # Sort keys for consistent hashing
        json_str = json.dumps(hashable_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    @staticmethod
    def compute_hash_from_dict(config_dict: Dict[str, Any]) -> str:
        """
        Compute hash from a dictionary (e.g., from database)

        Args:
            config_dict: Configuration dictionary

        Returns:
            SHA256 hash string
        """
        hashable_dict = {
            "forward": config_dict.get("forward_config", {}),
            "recon": {
                "family": config_dict.get("recon_family", ""),
                **config_dict.get("recon_params", {})
            },
            "uq": {
                "scheme": config_dict.get("uq_scheme", ""),
                "params": config_dict.get("uq_params", {})
            },
            "train": config_dict.get("train_config", {})
        }
        json_str = json.dumps(hashable_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
