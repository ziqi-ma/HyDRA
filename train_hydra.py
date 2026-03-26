"""
Expected batch format (you provide via your own DataLoader):
  batch = {
    "latents":        Float tensor [B, C_latent, T_latent, H_latent, W_latent]
                     Training freezes the context and predicts noise for the target.
    "context":        Float tensor [B, L, D_text] (text encoder embeddings).
    "context_camera": Float tensor [B, F, 12] (relative camera RT for context stream).
    "video_camera":   Float tensor [B, F, 12] (relative camera RT for target stream).
  }
"""

import argparse
from typing import Any, Dict, Optional

import lightning as pl
import torch
import torch.nn as nn

from diffsynth import HyDRAPipeline, ModelManager
from diffsynth.models.wan_video_dit import DynamicRetrievalAttention, MemoryTokenizer


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path: str,
        learning_rate: float = 1e-5,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        resume_ckpt_path: Optional[str] = None,
        hydra: bool = True,
        top_k: int = 10,
        num_frames: int = 40,
        frame_hw: int = 30 * 52,
        window_size: int = 5,
        train_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["resume_ckpt_path"])

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if "," in dit_path:
            model_manager.load_models([dit_path.split(",")])
        else:
            model_manager.load_models([dit_path])

        self.pipe = HyDRAPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(int(train_timesteps), training=True)

        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        num_heads = self.pipe.dit.blocks[0].self_attn.num_heads

        for block in self.pipe.dit.blocks:
            block.self_attn.tokenizer = MemoryTokenizer(dim)
            block.cam_encoder_con = nn.Linear(12, dim)
            block.cam_encoder_tgt = nn.Linear(12, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder_con.weight.data.zero_()
            block.cam_encoder_con.bias.data.zero_()
            block.cam_encoder_tgt.weight.data.zero_()
            block.cam_encoder_tgt.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))

            block.self_attn.hydra = bool(hydra)
            block.self_attn.change_sparse = bool(hydra)
            block.change_sparse = bool(hydra)
            if hydra:
                block.self_attn.attention_type = "sparse_frame"
                block.self_attn.attn = DynamicRetrievalAttention(
                    num_heads=num_heads,
                    num_frames=40,
                    frame_hw=30 * 52,
                    top_k=10,
                    frame_chunk_size=None,
                )

        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            self.pipe.dit.load_state_dict(state_dict, strict=True)

        self.learning_rate = float(learning_rate)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)

        self._freeze_and_mark_trainables()

    def _freeze_and_mark_trainables(self) -> None:
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

        for name, module in self.pipe.denoising_model().named_modules():
            if any(k in name for k in ("cam_encoder_con", "cam_encoder_tgt", "projector", "self_attn")):
                for p in module.parameters():
                    p.requires_grad = True

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        latents = batch["latents"].to(self.device, dtype=self.pipe.torch_dtype)
        context = batch["context"].to(self.device, dtype=self.pipe.torch_dtype)
        cam_emb_con = batch["context_camera"].to(self.device, dtype=self.pipe.torch_dtype)
        cam_emb_tgt = batch["video_camera"].to(self.device, dtype=self.pipe.torch_dtype)
        frame_ids = batch.get("frame_ids", None)

        self.pipe.device = self.device

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,), device=self.device)
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.device)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        tgt_latent_len = noisy_latents.shape[2] // 2
        noisy_latents[:, :, :tgt_latent_len, ...] = latents[:, :, :tgt_latent_len, ...]

        training_target = self.pipe.scheduler.training_target(
            latents[:, :, tgt_latent_len:, ...],
            noise[:, :, tgt_latent_len:, ...],
            timestep,
        )
        weight = self.pipe.scheduler.training_weight(timestep)

        noise_pred = self.pipe.denoising_model()(
            noisy_latents,
            timestep=timestep,
            cam_emb_tgt=cam_emb_tgt,
            cam_emb_con=cam_emb_con,
            context=context,
            frame_ids=frame_ids,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )

        loss = torch.nn.functional.mse_loss(
            noise_pred[:, :, tgt_latent_len:, ...].float(),
            training_target.float(),
        )
        loss = loss * weight
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.pipe.denoising_model().parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        state_dict = self.pipe.denoising_model().state_dict()
        checkpoint["state_dict"] = state_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HyDRA training skeleton (dataset not included).")
    p.add_argument("--dit_path", type=str, required=True)
    p.add_argument("--resume_ckpt_path", type=str, default=None)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--hydra", action="store_true", default=True)
    p.add_argument("--use_gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--use_gradient_checkpointing_offload", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    # This file intentionally does not provide a DataLoader.
    # Plug in your own Dataset/DataLoader and run with PyTorch Lightning Trainer.
    args = parse_args()
    _ = LightningModelForTrain(**vars(args))
    print(
        "Initialized LightningModelForTrain successfully.\n"
        "Provide your own DataLoader and run pl.Trainer(...).fit(model, dataloader)."
    )

