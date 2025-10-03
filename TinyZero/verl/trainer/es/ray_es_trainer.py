# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Ray-based trainer skeleton for Evolution Strategies (ES).

This mirrors the Ray PPO trainer’s structure but strips PPO-specific logic
and adds hooks where the ES flow (population rollout, fitness shaping,
parameter update) will sit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple, Type
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torch.utils.data import DataLoader
from verl.utils.tracking import Tracking

WorkerType = Type[Worker]


class Role(Enum):
    """Roles used by the ES trainer.

    We keep only the pieces currently needed (actor rollout and optional reward
    model), but reserving enum space makes future hybrid work easier.
    """

    ActorRollout = 0
    RewardModel = 1


@dataclass
class ResourcePoolManager:
    """Shallow wrapper for Ray resource pool creation/lookup."""

    resource_pool_spec: Dict[str, List[int]]
    mapping: Dict[Role, str]
    resource_pool_dict: Dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self) -> None:
        for name, processes in self.resource_pool_spec.items():
            pool = RayResourcePool(
                process_on_nodes=processes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=name,
            )
            self.resource_pool_dict[name] = pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]


@dataclass
class PopulationBatch:
    """Container for rollout results of a sampled population."""

    seeds: Sequence[int]
    rewards: torch.Tensor
    metadata: Dict[str, torch.Tensor]


def _broadcast_training_steps(config) -> None:
    """
    Inject total training steps into nested optimizer configs.

    Reuses the PPO helper to keep optimizer scheduling compatible until
    we wire a proper ES-specific dataloader.
    """

    total_training_steps = 0
    if isinstance(config.data.train_files, Iterable):
        total_training_steps = len(list(config.data.train_files))
    if config.trainer.total_generations is not None:
        total_training_steps = config.trainer.total_generations

    OmegaConf.set_struct(config, True)
    with open_dict(config):
        optim_cfg = getattr(config.actor_rollout_ref.actor, "optim", None)
        if optim_cfg is not None:
            optim_cfg.total_training_steps = total_training_steps


class RayESTrainer:
    """Ray-based single-controller Evolution Strategies trainer."""

    def __init__(self,
                config,
                tokenizer,
                role_worker_mapping: Dict[Role, WorkerType],
                resource_pool_manager: ResourcePoolManager,
                ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
                reward_fn: Callable[[DataProto], torch.Tensor] | None = None,
                val_reward_fn: Callable[[DataProto], torch.Tensor] | None = None):       
         
        self.config = config
        self.tokenizer = tokenizer
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.population_size = int(config.trainer.population_size)
        if config.trainer.algo != "es":
            raise ValueError(f"Expected ES config but got trainer.algo={config.trainer.algo!r}")

        self.population_size = int(config.trainer.population_size)
        self.noise_std = float(config.trainer.noise_std)
        self.antithetic = bool(config.trainer.get("antithetic", True))
        self.fitness_shaping = config.trainer.get("fitness_shaping", "rank")
        self.print_every = int(config.trainer.get("print_every", 0) or 0)

        if Role.ActorRollout not in role_worker_mapping:
            raise ValueError("ActorRollout role is required for RayESTrainer.")

        _broadcast_training_steps(self.config)

        self.actor_rollout_wg = None
        self.reward_model_wg = None
        self.global_generation = 0

        self._create_dataloader()
        self.train_batches_per_epoch = len(self.train_dataloader)

    # ------------------------------------------------------------------ #
    # Worker bootstrap                                                    #
    # ------------------------------------------------------------------ #
    def init_workers(self) -> None:
        """Create Ray worker groups for the configured roles."""

        self.resource_pool_manager.create_resource_pool()

        actor_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )

        colocated = {"actor_rollout": actor_cls}

        if Role.RewardModel in self.role_worker_mapping:
            rm_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            if rm_pool is actor_pool:
                colocated["reward_model"] = rm_cls
            else:
                rm_worker_cls = create_colocated_worker_cls({"reward_model": rm_cls})
                rm_wg = self.ray_worker_group_cls(
                    resource_pool=rm_pool, ray_cls_with_init=rm_worker_cls
                )
                spawned = rm_wg.spawn(prefix_set=["reward_model"])
                self.reward_model_wg = spawned["reward_model"]

        worker_dict_cls = create_colocated_worker_cls(colocated)
        wg = self.ray_worker_group_cls(
            resource_pool=actor_pool, ray_cls_with_init=worker_dict_cls
        )
        spawned = wg.spawn(prefix_set=colocated.keys())
        self.actor_rollout_wg = spawned["actor_rollout"]

        self.actor_rollout_wg.init_model()
        if self.reward_model_wg is None and "reward_model" in spawned:
            self.reward_model_wg = spawned["reward_model"]
            self.reward_model_wg.init_model()

    def _create_dataloader(self) -> None:
        tokenizer = self.tokenizer
        common_kwargs = dict(
            tokenizer=tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )

        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            **common_kwargs,
        )
        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            **common_kwargs,
        )

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_generations
        if self.config.trainer.get("total_training_steps") is not None:
            total_training_steps = self.config.trainer.total_training_steps

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            optim_cfg = getattr(self.config.actor_rollout_ref.actor, "optim", None)
            if optim_cfg is not None:
                optim_cfg.total_training_steps = total_training_steps

    def _next_training_batch(self) -> DataProto:
        try:
            batch_dict = next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_dataloader)
            batch_dict = next(self._train_iterator)
        return DataProto.from_single_dict(batch_dict)

    # ------------------------------------------------------------------ #
    # Core ES loop scaffolding                                           #
    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        """Entry point for the ES optimization loop."""

        if self.actor_rollout_wg is None:
            raise RuntimeError("Call init_workers() before fit().")

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self._log_trainer_banner()
        global_step = 0

        update_norms: List[float] = []
        best_worst_records: List[Dict[str, float]] = []
        for generation in range(self.config.trainer.total_generations):
            self.global_generation = generation
            generation_rewards: List[float] = []
            generation_start = time.time()

            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                prompts = DataProto.from_single_dict(batch_dict)
                seeds = self._sample_population()
                do_print = self._should_print_rollouts(generation, batch_idx)

                if do_print: print(f"********BATCH #{batch_idx}********")

                population_batch = self._evaluate_population(
                    seeds=seeds,
                    prompts=prompts,
                    do_print=do_print,
                )

                shaped_rewards = self._shape_fitness(population_batch.rewards)
                update = self._accumulate_update(seeds, shaped_rewards)
                update_norm = self._apply_update(update)
                update_norms.append(update_norm)

                # Collapse token-level rewards into a scalar per population member.
                member_rewards = population_batch.rewards.sum(dim=-1).float()
                generation_rewards.extend(member_rewards.tolist())
                best_value, best_idx = member_rewards.max(dim=0)
                worst_value, worst_idx = member_rewards.min(dim=0)
                best_worst_records.append(
                    {
                        "best_reward": float(best_value),
                        "best_seed": float(population_batch.seeds[int(best_idx)]),
                        "worst_reward": float(worst_value),
                        "worst_seed": float(population_batch.seeds[int(worst_idx)]),
                    }
                )
                global_step += 1

            elapsed = time.time() - generation_start
            metrics = self._summarize_generation(generation_rewards, update_norms, best_worst_records, elapsed)
            logger.log(metrics, step=generation)
            self._print_generation_summary(generation, metrics)

            if self._should_validate(generation):
                val_metrics = self._validate_population()
                if val_metrics:
                    logger.log(val_metrics, step=generation)

            if self._should_checkpoint(generation):
                self._save_checkpoint()

    # ------------------------------------------------------------------ #
    # Helper hooks – to be filled in subsequent changes                  #
    # ------------------------------------------------------------------ #
    def _sample_population(self) -> np.ndarray:
        if self.antithetic:
            half = self.population_size // 2
            seeds = np.random.randint(0, 2**31, size=half, dtype=np.int64)
            seeds = np.concatenate([seeds, -seeds])
        else:
            seeds = np.random.randint(0, 2**31, size=self.population_size, dtype=np.int64)
        return seeds

    def _evaluate_population(self,
                            seeds: Sequence[int],
                            prompts: DataProto,
                            do_print: bool = False,
                        ) -> PopulationBatch:
        """
        The actor worker must implement evaluate_population(prompts, seeds, noise_std).
        It should return a DataProto with one entry per population member.
        """

        prompts.meta_info.update(
            {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "population_size": len(seeds),
                "noise_std": self.noise_std,
            }
        )

        rollout_outputs: DataProto = self.actor_rollout_wg.evaluate_population(
            prompts=prompts,
            seeds=list(map(int, seeds)),
            noise_std=float(self.noise_std),
        )
        rollout_outputs.meta_info["do_print"] = do_print
        rollout_outputs.non_tensor_batch.setdefault("extra_info", {})["do_print"] = do_print

        if self.reward_fn is not None:
            rewards_tensor = self.reward_fn(rollout_outputs)
        elif "rewards" in rollout_outputs.batch:
            rewards_tensor = rollout_outputs.batch["rewards"]
        else:
            raise RuntimeError("Rollout worker did not supply rewards; provide reward_fn.")

        rewards_tensor = rewards_tensor.to(torch.float32)
        return PopulationBatch(seeds=seeds, rewards=rewards_tensor, metadata=rollout_outputs.batch)

    def _shape_fitness(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.fitness_shaping == "rank":
            ranks = torch.argsort(torch.argsort(rewards, dim=0), dim=0)
            centered = ranks.float() - ranks.float().mean(dim=0, keepdim=True)
            return centered / torch.clamp(torch.sqrt(torch.tensor(rewards.numel(), dtype=torch.float32)), min=1.0)
        if self.fitness_shaping == "centered":
            return rewards - rewards.mean(dim=0, keepdim=True)
        if self.fitness_shaping == "zscore":
            std = rewards.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            return (rewards - rewards.mean(dim=0, keepdim=True)) / std
        return rewards

    def _accumulate_update(self, seeds: Sequence[int], shaped_rewards: torch.Tensor):
        shaped_rewards = shaped_rewards.detach().cpu().tolist()
        update = self.actor_rollout_wg.accumulate_es_update(
            seeds=list(map(int, seeds)),
            rewards=shaped_rewards,
            noise_std=float(self.noise_std),
        )
        return update

    def _apply_update(self, update) -> float:
        lr = float(self.config.trainer.learning_rate)
        # compute L2 norm (pre learning-rate scaling)
        sq_sum = 0.0
        for tensor in update:
            sq_sum += torch.sum(tensor.float() ** 2).item()
        update_norm = float(sq_sum ** 0.5)
        self.actor_rollout_wg.apply_es_update(update, learning_rate=lr)
        return update_norm

    def _log_generation_stats(
        self, population_batch: PopulationBatch, shaped_rewards: torch.Tensor
    ) -> None:
        mean_reward = population_batch.rewards.mean().item()
        print(f"[ES] generation={self.global_generation} mean_reward={mean_reward:.4f}")

    def _should_validate(self, generation: int) -> bool:
        freq = self.config.trainer.get("test_freq", -1)
        return self.val_reward_fn is not None and freq > 0 and (generation + 1) % freq == 0

    def _should_checkpoint(self, generation: int) -> bool:
        freq = self.config.trainer.get("save_freq", -1)
        return freq > 0 and (generation + 1) % freq == 0

    def _validate_population(self) -> None:
        if self.val_reward_fn is None:
            return

        reward_tensors = []
        data_sources = []
        for batch_dict in self.val_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            gen_batch = batch.pop(["input_ids", "attention_mask", "position_ids"])
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }

            gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                gen_batch, self.actor_rollout_wg.world_size
            )
            outputs_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
            outputs = unpad_dataproto(outputs_padded, pad_size=pad_size)
            batch = batch.union(outputs)

            reward = self.val_reward_fn(batch)
            reward_tensors.append(reward)
            data_sources.append(batch.non_tensor_batch.get("data_source", ["unknown"] * reward.shape[0]))

        if not reward_tensors:
            return {}

        rewards = torch.cat(reward_tensors, dim=0).sum(-1).cpu()
        sources = np.concatenate(data_sources, axis=0)

        metrics: Dict[str, float] = {}
        for source in np.unique(sources):
            mask = sources == source
            metrics[f"val/reward_mean/{source}"] = float(rewards[mask].mean())
        metrics["val/reward_mean/all"] = float(rewards.mean())
        return metrics

    def _save_checkpoint(self) -> None:
        local_dir = os.path.join(
            self.config.trainer.default_local_dir,
            "actor",
            f"generation_{self.global_generation}",
        )
        remote_dir = None
        if self.config.trainer.default_hdfs_dir is not None:
            remote_dir = os.path.join(self.config.trainer.default_hdfs_dir, "actor")
        self.actor_rollout_wg.save_checkpoint(local_dir, remote_dir)

    def _log_trainer_banner(self) -> None:
        print("=== RayESTrainer (beta) ===")
        print(
            f"population_size={self.population_size} noise_std={self.noise_std} "
            f"antithetic={self.antithetic}"
        )
        print(f"fitness_shaping={self.fitness_shaping}")
    
    def _should_print_rollouts(self, generation: int, batch_idx: int) -> bool:
        if self.print_every <= 0 or self.train_batches_per_epoch == 0:
            return False
        global_idx = generation * self.train_batches_per_epoch + batch_idx
        return global_idx % self.print_every == 0

    def _summarize_generation(self, rewards: List[float], update_norms: List[float], best_worst: List[Dict[str, float]], elapsed: float,) -> Dict[str, float]:
        rewards_np = np.asarray(rewards, dtype=np.float32)
        p25, p75 = np.percentile(rewards_np, [25, 75])
        update_norms_np = np.asarray(update_norms, dtype=np.float32) if update_norms else np.zeros(1, np.float32)
        recent = best_worst[-1] if best_worst else {"best_reward": 0.0, "worst_reward": 0.0}
        hist_counts, hist_edges = np.histogram(rewards_np, bins=20)
        
        return {
            "generation/reward_mean": float(rewards_np.mean()),
            "generation/reward_max": float(rewards_np.max()),
            "generation/reward_min": float(rewards_np.min()),
            "generation/reward_std": float(rewards_np.std(ddof=0)),
            "generation/reward_p25": float(p25),
            "generation/reward_p75": float(p75),
            "generation/best_reward": float(recent["best_reward"]),
            "generation/worst_reward": float(recent["worst_reward"]),
            "generation/reward_hist_counts": hist_counts.tolist(),
            "generation/reward_hist_edges": hist_edges.tolist(),
            "generation/update_norm_mean": float(update_norms_np.mean()),
            "generation/update_norm_max": float(update_norms_np.max()),
            "generation/time_seconds": float(elapsed),
        }

    def _print_generation_summary(self, generation: int, metrics: Dict[str, float]) -> None:
        print(
            f"[ES] generation={generation} "
            f"mean={metrics['generation/reward_mean']:.4f} "
            f"max={metrics['generation/reward_max']:.4f} "
            f"min={metrics['generation/reward_min']:.4f} "
            f"time={metrics['generation/time_seconds']:.2f}s"
        )