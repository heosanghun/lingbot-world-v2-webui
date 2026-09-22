#!/usr/bin/env python3
"""
LingBot-World-V2 Web UI (Gradio)
Local Server: http://localhost:7860
"""

import os
import sys
import glob
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import gradio as gr
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

def get_available_examples():
    if not EXAMPLES_DIR.exists():
        return []
    return sorted([d.name for d in EXAMPLES_DIR.iterdir() if d.is_dir()])

def load_example_data(example_name):
    ex_path = EXAMPLES_DIR / example_name
    img_path = ex_path / "image.jpg"
    prompt_path = ex_path / "prompt.txt"
    
    prompt_text = ""
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
            
    img = str(img_path) if img_path.exists() else None
    return str(ex_path), img, prompt_text

def run_inference(
    task,
    infer_mode,
    ckpt_dir,
    assets_dir,
    image,
    action_path,
    prompt,
    size,
    frame_num,
    seed,
    offload_model,
    t5_cpu,
    progress=gr.Progress()
):
    if not ckpt_dir or not os.path.exists(ckpt_dir):
        return None, f"❌ 체크포인트 디렉터리를 찾을 수 없습니다: '{ckpt_dir}'.\nREADME.md를 참고하여 먼저 모델 가중치를 다운로드해 주세요."
    
    if image is None:
        return None, "❌ 시작 이미지를 업로드하거나 예제를 선택해 주세요."
    
    if not action_path or not os.path.exists(action_path):
        return None, f"❌ 액션/카메라 경로를 찾을 수 없습니다: '{action_path}'.\n'examples/03' 같은 예제 폴더를 지정해 주세요."
    
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_img_path = output_dir / f"input_{timestamp}.jpg"
    if isinstance(image, str):
        temp_img_path = image
    else:
        Image.fromarray(image).save(temp_img_path)

    save_video_path = output_dir / f"lingbot_{timestamp}.mp4"
    
    cmd = [
        sys.executable,
        "generate.py",
        "--task", task,
        "--infer_mode", infer_mode,
        "--ckpt_dir", ckpt_dir,
        "--image", str(temp_img_path),
        "--action_path", action_path,
        "--prompt", prompt,
        "--size", size,
        "--frame_num", str(int(frame_num)),
        "--base_seed", str(int(seed)),
        "--save_file", str(save_video_path),
    ]
    
    if assets_dir and os.path.exists(assets_dir):
        cmd.extend(["--assets_dir", assets_dir])
        
    if offload_model:
        cmd.extend(["--offload_model", "True"])
    else:
        cmd.extend(["--offload_model", "False"])
        
    if t5_cpu:
        cmd.append("--t5_cpu")

    progress(0.1, desc="LingBot-World-V2 추론 시작 중...")
    logging.info(f"실행 명령: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    output_logs = []
    for line in process.stdout:
        output_logs.append(line)
        logging.info(line.strip())
        
    process.wait()
    log_text = "".join(output_logs)
    
    if process.returncode != 0:
        return None, f"❌ 영상 생성 실패 (에러 코드: {process.returncode}):\n{log_text}"
    
    if not save_video_path.exists():
        mp4_files = sorted(output_dir.glob("*.mp4"), key=os.path.getmtime, reverse=True)
        if mp4_files:
            save_video_path = mp4_files[0]
            
    if save_video_path.exists():
        return str(save_video_path), f"✅ 생성 완료! 파일 저장 경로: {save_video_path}"
    else:
        return None, f"⚠️ 비디오 파일을 찾을 수 없습니다.\n로그:\n{log_text}"

def create_ui():
    custom_css = """
    .container { max-width: 1200px; margin: auto; }
    .header-box { text-align: center; margin-bottom: 20px; }
    """
    
    available_examples = get_available_examples()
    default_ex = available_examples[0] if available_examples else "03"
    default_action, default_img, default_prompt = load_example_data(default_ex) if available_examples else ("", None, "")

    with gr.Blocks(title="LingBot-World 2.0 Web UI") as demo:
        with gr.Column(elem_classes=["container"]):
            gr.Markdown(
                """
                # 🌐 LingBot-World 2.0 (LingBot-World-Infinity)
                ### Causal Interactive World Simulator Web UI
                
                - **로컬 서버 주소**: `http://localhost:7860` 또는 `http://127.0.0.1:7860`
                - **참고**: 공식 모델 가중치(Checkpoint)가 지정된 디렉터리에 다운로드되어 있어야 실행할 수 있습니다.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 모델 설정")
                    task = gr.Dropdown(
                        label="모델 태스크 (Task)",
                        choices=["i2v-1.3B", "i2v-A14B"],
                        value="i2v-1.3B",
                        info="1.3B 모델은 단일 GPU에서 비교적 가볍게 실행 가능합니다."
                    )
                    infer_mode = gr.Radio(
                        label="추론 모드 (Infer Mode)",
                        choices=["causal_fast", "causal_pretrain"],
                        value="causal_fast",
                        info="causal_fast: 4단계 고속 생성 | causal_pretrain: 40단계 CFG"
                    )
                    ckpt_dir = gr.Textbox(
                        label="체크포인트 디렉터리 (Checkpoint Dir)",
                        value="lingbot-world-v2-1.3b-causal-fast",
                        placeholder="예: ./lingbot-world-v2-1.3b-causal-fast"
                    )
                    assets_dir = gr.Textbox(
                        label="공통 에셋 디렉터리 (Assets Dir - T5/VAE)",
                        value="lingbot-world-v2-14b-causal-fast",
                        placeholder="1.3B 사용 시 14B 공통 에셋 폴더 경로 입력 (옵션)",
                        info="1.3B는 T5/VAE를 14B 체크포인트와 공유합니다."
                    )
                    
                    gr.Markdown("### 📁 예제 프리셋 불러오기")
                    example_dropdown = gr.Dropdown(
                        label="예제 프리셋 (Examples)",
                        choices=available_examples,
                        value=default_ex if available_examples else None,
                        info="선택 시 해당 예제의 이미지, 프롬프트, 카메라 궤적이 자동으로 채워집니다."
                    )
                    
                    gr.Markdown("### 🎬 입력 데이터")
                    input_image = gr.Image(
                        label="시작 이미지 (First Frame Image)",
                        type="filepath",
                        value=default_img
                    )
                    action_path = gr.Textbox(
                        label="액션 / 카메라 경로 (Action/Camera Directory)",
                        value=default_action,
                        placeholder="poses.npy, intrinsics.npy 등이 포함된 폴더 경로"
                    )
                    prompt = gr.Textbox(
                        label="텍스트 프롬프트 (Prompt)",
                        lines=3,
                        value=default_prompt,
                        placeholder="생성할 장면의 묘사를 입력하세요..."
                    )
                    
                    with gr.Accordion("고급 생성 옵션 (Advanced Options)", open=False):
                        size = gr.Dropdown(
                            label="해상도 (Resolution)",
                            choices=["480*832", "832*480", "1280*720", "720*1280"],
                            value="480*832"
                        )
                        frame_num = gr.Slider(
                            label="생성 프레임 수 (Frame Num: 4n+1)",
                            minimum=17,
                            maximum=361,
                            step=4,
                            value=81,
                            info="81프레임(약 5초), 361프레임(약 22초)"
                        )
                        seed = gr.Number(label="시드 (Seed)", value=42, precision=0)
                        offload_model = gr.Checkbox(
                            label="CPU Offload 활성화 (단일 GPU 메모리 절약)",
                            value=True
                        )
                        t5_cpu = gr.Checkbox(
                            label="T5 인코더 CPU 로드 (VRAM 절약)",
                            value=True
                        )
                        
                    generate_btn = gr.Button("🚀 비디오 생성 시작", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 🎥 생성 결과")
                    output_video = gr.Video(label="생성된 비디오 (Generated Video)")
                    status_log = gr.Textbox(label="상태 및 로그 (Status & Logs)", lines=10, interactive=False)

            def update_preset(ex_name):
                if not ex_name:
                    return "", None, ""
                return load_example_data(ex_name)
            
            example_dropdown.change(
                fn=update_preset,
                inputs=[example_dropdown],
                outputs=[action_path, input_image, prompt]
            )

            generate_btn.click(
                fn=run_inference,
                inputs=[
                    task,
                    infer_mode,
                    ckpt_dir,
                    assets_dir,
                    input_image,
                    action_path,
                    prompt,
                    size,
                    frame_num,
                    seed,
                    offload_model,
                    t5_cpu,
                ],
                outputs=[output_video, status_log]
            )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LingBot-World-V2 Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port number (default: 7860)")
    parser.add_argument("--share", action="store_true", default=False, help="Create public Gradio link")
    args = parser.parse_args()

    print("=" * 60)
    print(f"🚀 LingBot-World 2.0 Web UI 서버 시작")
    print(f"🔗 로컬 서버 주소: http://localhost:{args.port} 또는 http://127.0.0.1:{args.port}")
    print(f"🔗 외부 접속 바인딩: http://{args.host}:{args.port}")
    print("=" * 60)

    demo = create_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
