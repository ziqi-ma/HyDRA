"""
Run HyDRA inference on the wmagent vbench input set.

For every scene under
    /data/ziqi/data/wmagent/results_oldformat/baselines/input/vbench/<scene>/
    (with init_16x9.png, prompt.txt, trajectories/<traj>/trajectory.npz)
generate a video for each of 6 trajectories:
    right-up, s-w, up-right, w-left, down-left, w-right

Outputs to:
    /data/ziqi/data/wmagent/results_oldformat/baselines/hydra/<scene>/<traj>.mp4

The init image is treated as a static condition video (repeated 77 frames).
The condition camera is held static (identity); the target camera is the
provided trajectory, made relative to its first frame and subsampled to the
20 keyframes HyDRA's DiT consumes.
"""
import argparse
import os
from typing import List, Tuple

import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision.transforms import v2

from diffsynth import HyDRAPipeline, ModelManager, save_video
from diffsynth.models.wan_video_dit import DynamicRetrievalAttention, MemoryTokenizer


INPUT_ROOT = "/data/ziqi/data/wmagent/results_oldformat/baselines/input/vbench"
OUTPUT_ROOT = "/data/ziqi/data/wmagent/eval/baselines/hydra"
TRAJECTORIES = ["right-up", "s-w", "up-right", "w-left", "down-left", "w-right"]
# Mapping from input trajectory dir name -> output subdir name (matches other baselines)
TRAJ_OUT_DIR = {
    "right-up": "seen_right_up",
    "s-w": "seen_s_w",
    "up-right": "seen_up_right",
    "down-left": "unseen_down_left",
    "w-left": "unseen_w_left",
    "w-right": "unseen_w_right",
}
NUM_FRAMES = 77
HEIGHT = 480
WIDTH = 832
NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    image = torchvision.transforms.functional.resize(
        image,
        (round(src_h * scale), round(src_w * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    return torchvision.transforms.functional.center_crop(image, [height, width])


def image_to_static_video(image_path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """[C, T, H, W] in [-1, 1], same frame repeated num_frames times."""
    frame_process = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    img = crop_and_resize(img, height, width)
    frame = frame_process(img)  # [C, H, W]
    video = frame.unsqueeze(1).expand(-1, num_frames, -1, -1).contiguous()
    return video


def _apply_coord_transform_np(c2w: np.ndarray) -> np.ndarray:
    """Mirror of HyDRA's _apply_coordinate_transform (Unreal-cm -> OpenCV-m)."""
    c2w_t = c2w[..., :, [1, 2, 0, 3]].copy()
    c2w_t[..., :3, 1] *= -1.0
    c2w_t[..., :3, 3] /= 100.0
    return c2w_t


def cameras_for(traj_npz: str, translation_scale: float, apply_transform: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (cond_cam[20,12], tgt_cam[20,12]) for HyDRA.

    LingBot trajectories are already in OpenCV convention (x-right, y-down,
    z-forward, meters), which matches HyDRA's *post-transform* training
    convention. So we skip `_apply_coordinate_transform` (which converts
    Unreal-cm -> OpenCV-meters) and compute relative poses directly.

    - Target camera: 20 evenly-spaced c2w from the trajectory, made relative
      to frame 0, top-3 rows flattened to 12.
    - Condition camera: identity for all 20 keyframes (static image cond).
    """
    data = np.load(traj_npz)
    w2c = data["w2c"].astype(np.float32)  # [N, 4, 4]
    c2w_all = np.linalg.inv(w2c)
    if translation_scale != 1.0:
        c2w_all = c2w_all.copy()
        c2w_all[:, :3, 3] *= translation_scale

    n = c2w_all.shape[0]
    idx = np.round(np.linspace(0, n - 1, 20)).astype(int)
    c2w_20 = c2w_all[idx]  # [20, 4, 4]
    if apply_transform:
        c2w_20 = _apply_coord_transform_np(c2w_20)

    ref_w2c = np.linalg.inv(c2w_20[0])
    rel_tgt = ref_w2c[None] @ c2w_20  # [20, 4, 4]
    tgt = torch.from_numpy(rel_tgt[:, :3, :4].reshape(20, 12).astype(np.float32))

    if apply_transform:
        # Static condition: 20 copies of c2w_20[0] (post-transform), then
        # relative to itself = identity. The basis-change pre/post conjugates
        # an identity to identity, so static-cam ctx remains identity in 12-d.
        pass
    eye_3x4 = np.eye(4, dtype=np.float32)[:3, :4].reshape(12)
    ctx = torch.from_numpy(np.tile(eye_3x4, (20, 1)))
    return ctx, tgt


def build_pipeline(
    device: str,
    ckpt_path: str,
    base_dit_path: str,
    base_text_encoder_path: str,
    base_vae_path: str,
) -> HyDRAPipeline:
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([base_dit_path, base_text_encoder_path, base_vae_path])
    pipe = HyDRAPipeline.from_model_manager(model_manager, device=device)

    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    num_heads = pipe.dit.blocks[0].self_attn.num_heads

    for block in pipe.dit.blocks:
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

        block.self_attn.hydra = True
        block.self_attn.change_sparse = True
        block.self_attn.attention_type = "sparse_frame"
        block.self_attn.attn = DynamicRetrievalAttention(
            num_heads=num_heads,
            num_frames=40,
            frame_hw=(30 * 52),
            top_k=10,
            frame_chunk_size=None,
        )

    state = torch.load(ckpt_path, map_location="cpu")
    pipe.dit.load_state_dict(state, strict=True)
    pipe.to(device)
    pipe.to(dtype=torch.bfloat16)
    return pipe


def infer_one(
    pipe: HyDRAPipeline,
    image_path: str,
    traj_npz: str,
    prompt: str,
    out_path: str,
    cfg_scale: float,
    num_inference_steps: int,
    seed: int,
    fps: int,
    save_concat: bool,
    translation_scale: float,
    apply_transform: bool,
) -> None:
    cond_video = image_to_static_video(image_path, NUM_FRAMES, HEIGHT, WIDTH)
    ref_camera, target_camera = cameras_for(traj_npz, translation_scale=translation_scale, apply_transform=apply_transform)

    with torch.no_grad():
        pred_frames, _ = pipe(
            prompt=prompt,
            negative_prompt=NEG_PROMPT,
            source_video=cond_video.to(dtype=pipe.torch_dtype, device=pipe.device),
            target_camera=target_camera.to(dtype=pipe.torch_dtype, device=pipe.device),
            ref_camera=ref_camera.to(dtype=pipe.torch_dtype, device=pipe.device),
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            tiled=True,
        )

    if save_concat:
        cond_frames_list = pipe.tensor2video(cond_video)
        out_frames = list(cond_frames_list) + list(pred_frames)
    else:
        out_frames = list(pred_frames)
    save_video(out_frames, out_path, fps=fps, quality=5)


def list_scenes(input_root: str) -> List[str]:
    return sorted(d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", default=INPUT_ROOT)
    p.add_argument("--output_root", default=OUTPUT_ROOT)
    p.add_argument("--ckpt_path", default="./ckpts/HyDRA/hydra.ckpt")
    p.add_argument("--base_dit_path", default="./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    p.add_argument("--base_text_encoder_path", default="./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    p.add_argument("--base_vae_path", default="./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--scenes", nargs="*", default=None, help="If set, only run these scene names.")
    p.add_argument("--trajectories", nargs="*", default=TRAJECTORIES)
    p.add_argument("--shard", default="0/1",
                   help="Worker shard 'i/N': split (scene, traj) jobs across N workers, this is worker i (0-indexed).")
    p.add_argument("--save_concat", action="store_true",
                   help="Prepend the static condition frames to the saved mp4 (debug). Default off: only the 77 generated frames.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--translation_scale", type=float, default=1.0,
                   help="Multiply input c2w translations before HyDRA's built-in /100. "
                        "Use this to compensate if your input units differ from training.")
    p.add_argument("--apply_transform", action="store_true",
                   help="Apply HyDRA's _apply_coordinate_transform (Unreal-cm -> OpenCV-m) to input c2w.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pipe = build_pipeline(
        device=args.device,
        ckpt_path=args.ckpt_path,
        base_dit_path=args.base_dit_path,
        base_text_encoder_path=args.base_text_encoder_path,
        base_vae_path=args.base_vae_path,
    )

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    assert 0 <= shard_i < shard_n

    scenes = args.scenes if args.scenes else list_scenes(args.input_root)
    all_jobs = [(s, t) for s in scenes for t in args.trajectories]
    my_jobs = [j for k, j in enumerate(all_jobs) if k % shard_n == shard_i]
    print(f"[shard {shard_i}/{shard_n}] {len(my_jobs)}/{len(all_jobs)} jobs")

    by_scene: dict = {}
    for s, t in my_jobs:
        by_scene.setdefault(s, []).append(t)

    for scene, trajs in by_scene.items():
        scene_dir = os.path.join(args.input_root, scene)
        image_path = os.path.join(scene_dir, "init_16x9.png")
        prompt_path = os.path.join(scene_dir, "prompt.txt")
        if not (os.path.isfile(image_path) and os.path.isfile(prompt_path)):
            print(f"[skip] {scene}: missing image or prompt")
            continue

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        if not prompt:
            print(f"[skip] {scene}: empty prompt")
            continue

        out_scene_dir = os.path.join(args.output_root, scene)
        os.makedirs(out_scene_dir, exist_ok=True)

        for traj in trajs:
            traj_npz = os.path.join(scene_dir, "trajectories", traj, "trajectory.npz")
            if not os.path.isfile(traj_npz):
                print(f"[skip] {scene}/{traj}: trajectory missing")
                continue
            traj_out_dir = os.path.join(out_scene_dir, TRAJ_OUT_DIR[traj])
            os.makedirs(traj_out_dir, exist_ok=True)
            out_path = os.path.join(traj_out_dir, "gen.mp4")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"[skip] {scene}/{traj}: exists")
                continue

            print(f"[run]  {scene}/{traj} -> {out_path}")
            infer_one(
                pipe=pipe,
                image_path=image_path,
                traj_npz=traj_npz,
                prompt=prompt,
                out_path=out_path,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                fps=args.fps,
                save_concat=args.save_concat,
                translation_scale=args.translation_scale,
                apply_transform=args.apply_transform,
            )


if __name__ == "__main__":
    main()
