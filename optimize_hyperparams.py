#!/usr/bin/env python3
"""
Hyperparameter Optimization for Marvis TTS Fine-tuning

Uses Optuna to find optimal hyperparameters for Elise voice training.
Optimizes: learning_rate, decoder_fraction, decoder_loss_weight

Usage:
    python optimize_hyperparams.py --trials 20 --steps 1000

Recommended on RunPod:
    - RTX 4090: 20 trials × 1000 steps ≈ 3 hours, $1.32
    - Output: configs/elise_finetune_optimized.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

# Add marvis_tts to path
sys.path.insert(0, str(Path(__file__).parent))

from train import TrainingConfig, train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparamOptimizer:
    def __init__(
        self,
        base_config_path: str,
        max_steps: int = 1000,
        output_dir: str = "optuna_study",
    ):
        self.base_config_path = base_config_path
        self.max_steps = max_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load base config
        with open(base_config_path) as f:
            self.base_config_dict = json.load(f)

        logger.info(f"Loaded base config from: {base_config_path}")
        logger.info(f"Optimization max steps: {max_steps}")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Returns validation loss after training for max_steps.
        """
        # Sample hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        decoder_fraction = trial.suggest_float("decoder_fraction", 0.03125, 0.25)  # 1/32 to 1/4
        decoder_loss_weight = trial.suggest_float("decoder_loss_weight", 0.5, 2.0)

        # Optional: batch size (if GPU memory allows)
        # max_batch_size = trial.suggest_categorical("max_batch_size", [32, 48, 64])

        logger.info(f"\n{'='*70}")
        logger.info(f"Trial {trial.number}: lr={learning_rate:.6f}, "
                   f"decoder_frac={decoder_fraction:.4f}, "
                   f"decoder_weight={decoder_loss_weight:.3f}")
        logger.info(f"{'='*70}\n")

        # Create trial config
        trial_config_dict = self.base_config_dict.copy()
        trial_config_dict.update({
            "learning_rate": learning_rate,
            "decoder_fraction": decoder_fraction,
            "decoder_loss_weight": decoder_loss_weight,
            # "max_batch_size": max_batch_size,
        })

        # Save trial config
        trial_config_path = self.output_dir / f"trial_{trial.number}_config.json"
        with open(trial_config_path, 'w') as f:
            json.dump(trial_config_dict, f, indent=2)

        try:
            # Train for limited steps
            config = TrainingConfig(**trial_config_dict)

            # Override to limit training steps
            original_max_tokens = config.max_tokens
            config.max_tokens = self.max_steps * config.max_batch_size * 512  # Approximate

            # Train (this will use accelerate)
            final_loss = self._run_training(config, trial)

            # Report intermediate values for pruning
            trial.report(final_loss, self.max_steps)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            logger.info(f"Trial {trial.number} completed with loss: {final_loss:.4f}")
            return final_loss

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return high loss for failed trials
            return float('inf')

    def _run_training(self, config: TrainingConfig, trial: optuna.Trial) -> float:
        """
        Run training and return final loss.

        Note: This is simplified - you may need to modify train.py
        to return final loss instead of running indefinitely.
        """
        # Import here to avoid circular imports
        from marvis_tts.trainer import Trainer
        from marvis_tts.models import CSMModel
        from marvis_tts import flavors
        import datasets
        from accelerate import Accelerator

        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision='bf16' if config.precision == 'bf16' else 'no'
        )

        # Load dataset
        dataset_path = Path(config.dataset_repo_id)
        if dataset_path.exists():
            dataset = datasets.load_dataset(
                "webdataset",
                data_files=str(dataset_path / "*.tar"),
                split="train",
                streaming=True
            )
        else:
            raise ValueError(f"Dataset not found: {config.dataset_repo_id}")

        # Initialize model
        backbone_flavor = flavors.get_backbone_flavor(config.backbone_flavor)
        decoder_flavor = flavors.get_decoder_flavor(config.decoder_flavor)

        model_args = flavors.ModelArgs(
            backbone_flavor=backbone_flavor,
            decoder_flavor=decoder_flavor,
            tokenizer=config.tokenizer,
            audio_num_codebooks=config.audio_num_codebooks,
        )

        model = CSMModel(model_args).to(accelerator.device)

        # Load text tokenizer
        if config.tokenizer.startswith("smollm"):
            from transformers import AutoTokenizer
            text_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M").encode
        else:
            text_tokenizer = None

        # Initialize trainer
        trainer = Trainer(
            model=model,
            dataset=dataset,
            text_tokenizer=text_tokenizer,
            config=config,
            accelerator=accelerator
        )

        # Train for limited steps
        losses = []
        step = 0

        for batch in trainer.dataloader:
            if step >= self.max_steps:
                break

            # Forward pass
            loss_dict = trainer.model(batch['tokens'], batch['tokens_mask'],
                                     decoder_fraction=config.decoder_fraction)
            loss = loss_dict['loss']

            # Backward and optimize
            trainer.accelerator.backward(loss)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

            losses.append(loss.item())
            step += 1

            # Report to Optuna for pruning
            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                trial.report(avg_loss, step)

                if trial.should_prune():
                    raise optuna.TrialPruned()

        # Return average loss over last 100 steps
        final_loss = sum(losses[-100:]) / min(100, len(losses))
        return final_loss

    def optimize(self, n_trials: int = 20) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run

        Returns:
            Best hyperparameters found
        """
        logger.info(f"\n{'='*70}")
        logger.info("Starting Hyperparameter Optimization")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"Steps per trial: {self.max_steps}")
        logger.info(f"{'='*70}\n")

        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=200),
            study_name="marvis_elise_optimization"
        )

        # Optimize
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        # Print results
        logger.info(f"\n{'='*70}")
        logger.info("Optimization Complete!")
        logger.info(f"{'='*70}\n")

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best loss: {study.best_value:.4f}")
        logger.info(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")

        # Save best config
        best_config = self.base_config_dict.copy()
        best_config.update(study.best_params)

        output_config_path = Path("configs/elise_finetune_optimized.json")
        with open(output_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)

        logger.info(f"\n✓ Best config saved to: {output_config_path}")

        # Save study
        study_path = self.output_dir / "study.pkl"
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)

        logger.info(f"✓ Study saved to: {study_path}")

        return study.best_params


def main():
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for Marvis TTS fine-tuning"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/elise_finetune_gpu.json",
        help="Base configuration file"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of optimization trials"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Training steps per trial"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="optuna_study",
        help="Directory to save study results"
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available! This will be very slow.")
        logger.warning("    Consider running on RunPod with GPU.")

    # Run optimization
    optimizer = HyperparamOptimizer(
        base_config_path=args.config,
        max_steps=args.steps,
        output_dir=args.output_dir
    )

    best_params = optimizer.optimize(n_trials=args.trials)

    logger.info(f"\n{'='*70}")
    logger.info("🎉 Optimization complete!")
    logger.info(f"{'='*70}\n")
    logger.info("Next steps:")
    logger.info("  1. Review: configs/elise_finetune_optimized.json")
    logger.info("  2. Train with optimal hyperparameters:")
    logger.info("     accelerate launch train.py configs/elise_finetune_optimized.json")
    logger.info("")


if __name__ == "__main__":
    main()
