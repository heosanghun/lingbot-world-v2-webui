import gc
import os
import shutil
import time
from datetime import datetime
from PIL import Image
import numpy as np
import torch
import torchvision
import imageio

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.action_generator import generate_action_trajectory
from wan.utils.utils import save_video


class InteractiveEngine:
    _instance = None

    def __init__(self, device_id: int = 0, ckpt_dir: str = "lingbot-world-v2-1.3b-causal-fast", assets_dir: str = "lingbot-world-v2-14b-causal-fast"):
        self.device_id = device_id
        self.ckpt_dir = ckpt_dir
        self.assets_dir = assets_dir
        self.wan_i2v = None
        self.active_session = None

    @classmethod
    def get_instance(cls, device_id: int = 0):
        if cls._instance is None:
            cls._instance = cls(device_id=device_id)
        return cls._instance

    def load_model(self):
        if self.wan_i2v is not None:
            return self.wan_i2v

        print("[InteractiveEngine] Loading LingBot-World 1.3B Causal-Fast pipeline into VRAM...")
        t0 = time.time()
        cfg = WAN_CONFIGS["i2v-1.3B"]
        self.wan_i2v = wan.WanI2VCausal(
            config=cfg,
            checkpoint_dir=self.ckpt_dir,
            assets_dir=self.assets_dir,
            device_id=self.device_id,
            rank=0,
            t5_cpu=False,
            local_attn_size=18,
            sink_size=6,
            infer_mode="causal_fast",
        )
        print(f"[InteractiveEngine] Model successfully loaded in {time.time() - t0:.2f}s!")
        return self.wan_i2v

    def start_session(self, initial_image: Image.Image, prompt: str, seed: int = 42) -> dict:
        self.load_model()

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.abspath(os.path.join("outputs", "interactive", f"session_{session_id}"))
        clips_dir = os.path.join(session_dir, "clips")
        frames_dir = os.path.join(session_dir, "frames")
        os.makedirs(clips_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        # Standardize initial image (RGB)
        init_img = initial_image.convert("RGB")
        init_frame_path = os.path.join(frames_dir, "frame_0000.jpg")
        init_img.save(init_frame_path, quality=95)

        self.active_session = {
            "session_id": session_id,
            "session_dir": session_dir,
            "clips_dir": clips_dir,
            "frames_dir": frames_dir,
            "current_image": init_img,
            "current_prompt": prompt.strip() if prompt else "A continuous cinematic exploration in a photorealistic 3D world.",
            "seed": seed if seed >= 0 else 42,
            "step_count": 0,
            "action_history": ["START"],
            "clip_paths": [],
            "full_video_path": None,
            "cumulative_frames": [init_frame_path],
        }

        return {
            "status": "READY",
            "session_id": session_id,
            "step_count": 0,
            "current_image": init_img,
            "prompt": self.active_session["current_prompt"],
            "history": "🏁 [시작점]",
        }

    def step(self, action: str, prompt: str = None, step_size: float = 0.5, turn_angle: float = 12.0) -> dict:
        if self.active_session is None:
            raise RuntimeError("세션이 시작되지 않았습니다. 먼저 시작 이미지를 업로드하고 [월드 시작]을 눌러주세요.")

        session = self.active_session
        sess_dir = session["session_dir"]
        step_idx = session["step_count"] + 1

        # Allow dynamic prompt update (Director intervention)
        if prompt and prompt.strip():
            session["current_prompt"] = prompt.strip()

        action_name = action.upper().strip()
        print(f"[InteractiveEngine] Step {step_idx}: Executing action '{action_name}'...")
        t_start = time.time()

        # 1. Generate camera trajectory for action
        action_dir = os.path.join(sess_dir, "temp_action")
        os.makedirs(action_dir, exist_ok=True)
        poses, intrinsics = generate_action_trajectory(
            action=action_name,
            frame_num=17,
            step_size=step_size,
            turn_angle_deg=turn_angle,
            width=832,
            height=480,
        )
        np.save(os.path.join(action_dir, "poses.npy"), poses)
        np.save(os.path.join(action_dir, "intrinsics.npy"), intrinsics)

        # 2. Run inference with in-memory model
        current_img = session["current_image"]
        current_seed = session["seed"] + step_idx

        with torch.no_grad():
            video_tensor = self.wan_i2v.generate(
                input_prompt=session["current_prompt"],
                img=current_img,
                action_path=action_dir,
                chunk_size=4,
                max_area=MAX_AREA_CONFIGS["480*832"],
                frame_num=17,
                shift=5.0,
                seed=current_seed,
                offload_model=False,
            )

        t_infer = time.time() - t_start

        # 3. Save latest step video clip
        clip_filename = f"step_{step_idx:03d}_{action_name}.mp4"
        clip_path = os.path.join(session["clips_dir"], clip_filename)
        save_video(
            tensor=video_tensor[None],
            save_file=clip_path,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        session["clip_paths"].append(clip_path)

        # 4. Extract frames & update current_image to last frame
        # video_tensor is [C, F, H, W] in (-1, 1)
        frames_tensor = video_tensor.permute(1, 0, 2, 3) # [F, C, H, W]
        step_frames = []
        for f_idx, f_t in enumerate(frames_tensor):
            frame_np = ((f_t.clamp(-1, 1) + 1.0) / 2.0 * 255.0).permute(1, 2, 0).byte().cpu().numpy()
            frame_img = Image.fromarray(frame_np)
            global_frame_idx = (step_idx - 1) * 16 + f_idx
            frame_path = os.path.join(session["frames_dir"], f"frame_{global_frame_idx:04d}.jpg")
            frame_img.save(frame_path, quality=92)
            step_frames.append(frame_path)

        # The last frame becomes the anchor for the next action
        last_frame_img = Image.open(step_frames[-1])
        session["current_image"] = last_frame_img
        session["step_count"] = step_idx
        session["action_history"].append(action_name)

        # Exclude duplicated boundary frames when stitching
        if step_idx == 1:
            session["cumulative_frames"] = step_frames
        else:
            session["cumulative_frames"].extend(step_frames[1:])

        # 5. Render continuous full journey video
        full_video_path = os.path.join(sess_dir, f"full_journey_{session['session_id']}.mp4")
        self._stitch_frames(session["cumulative_frames"], full_video_path, fps=16)
        session["full_video_path"] = full_video_path

        # 6. Format action history readable text
        action_icons = {
            "W": "⬆️ 전진(W)",
            "S": "⬇️ 후진(S)",
            "A": "⬅️ 좌이동(A)",
            "D": "➡️ 우이동(D)",
            "Q": "⤹ 좌회전(Q)",
            "E": "⤸ 우회전(E)",
            "SPACE": "🔼 상승(Space)",
            "C": "🔽 하강(C)",
            "IDLE": "⏸️ 관찰(X)",
        }
        history_str = " ➔ ".join([action_icons.get(a, a) for a in session["action_history"]])

        print(f"[InteractiveEngine] Step {step_idx} done in {t_infer:.2f}s! Total frames: {len(session['cumulative_frames'])}")

        return {
            "status": "SUCCESS",
            "step_count": step_idx,
            "action": action_name,
            "latency_s": round(t_infer, 2),
            "latest_clip": clip_path,
            "full_video": full_video_path,
            "current_image": last_frame_img,
            "history": history_str,
            "prompt": session["current_prompt"],
        }

    def _stitch_frames(self, frame_paths: list[str], output_path: str, fps: int = 16):
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
        for p in frame_paths:
            img = imageio.imread(p)
            writer.append_data(img)
        writer.close()
