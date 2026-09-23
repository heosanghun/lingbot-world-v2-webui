#!/usr/bin/env python3
"""
LingBot-World-V2 Web UI (Gradio)
Interactive World Simulator & Real-time Game Explorer
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

from interactive_engine import InteractiveEngine

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
            
    img = Image.open(img_path) if img_path.exists() else None
    return str(ex_path), img, prompt_text

# ==============================================================================
# Interactive World Engine Handlers
# ==============================================================================
def handle_start_world(image, prompt, seed):
    if image is None:
        return (
            None, None,
            "❌ 시작 이미지를 업로드하거나 예제 프리셋을 선택해 주세요.",
            "준비되지 않음"
        )
    
    try:
        engine = InteractiveEngine.get_instance()
        init_res = engine.start_session(
            initial_image=image,
            prompt=prompt,
            seed=int(seed) if seed is not None else 42,
        )
        
        # Automatically generate Step 1 (W: forward motion) so the 3D viewport is immediately moving!
        step1_res = engine.step("W", prompt=prompt)
        
        status_msg = (
            f"🎮 [월드 시뮬레이션 가동 중!]\n"
            f"• 스텝 1 (전진) 생성 완료: {step1_res['latency_s']}초 (1.6초 디노이징 + VAE)\n"
            f"• 뷰포트에서 실시간 3D 영상이 재생 중입니다!\n"
            f"• 키보드 [W, A, S, D, Q, E, Space, C]를 누르거나 화면의 버튼을 눌러 이동하세요!"
        )
        return (
            step1_res["latest_clip"],
            step1_res["full_video"],
            status_msg,
            step1_res["history"],
        )
    except Exception as e:
        logging.error(f"Failed to start world session: {e}", exc_info=True)
        return None, None, f"❌ 세션 시작 실패: {str(e)}", "에러"

def handle_step_action(action_name, prompt, step_size, turn_angle):
    engine = InteractiveEngine.get_instance()
    if engine.active_session is None:
        return (
            None, None,
            "⚠️ 먼저 [🚀 탐험 시작 / 월드 초기화] 버튼을 눌러 월드를 시작해 주세요.",
            "월드 미시작"
        )
    
    try:
        res = engine.step(
            action=action_name,
            prompt=prompt,
            step_size=float(step_size),
            turn_angle=float(turn_angle),
        )
        status_msg = (
            f"🎮 [스텝 {res['step_count']} 이동 완료: {res['action']}]\n"
            f"• 실시간 생성 시간: {res['latency_s']}초 (1.6초 디노이징 + VAE)\n"
            f"• 누적 프레임: {len(engine.active_session['cumulative_frames'])} 프레임\n"
            f"• 다음 이동 방향 키(W, A, S, D)를 누르세요!"
        )
        return (
            res["latest_clip"],
            res["full_video"],
            status_msg,
            res["history"],
        )
    except Exception as e:
        logging.error(f"Step action error: {e}", exc_info=True)
        return None, None, f"❌ 이동 생성 실패: {str(e)}", "에러 발생"

def make_step_handler(action_code):
    def _handler(prompt_val, step_size_val, turn_angle_val):
        return handle_step_action(action_code, prompt_val, step_size_val, turn_angle_val)
    return _handler

# ==============================================================================
# Classic Batch Generation Handler (Preserved)
# ==============================================================================
def run_classic_inference(
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
    elif isinstance(image, Image.Image):
        image.save(temp_img_path)
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

# ==============================================================================
# UI Construction
# ==============================================================================
def create_ui():
    keyboard_client_js = """
    () => {
        if (window._lingbot_installed) return;
        window._lingbot_installed = true;
        console.log("🎮 LingBot World Controller Active!");

        window.addEventListener("keydown", function(e) {
            const active = document.activeElement;
            if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {
                return;
            }
            const key = e.key.toLowerCase();
            const keyMap = {
                'w': 'btn_act_w', 'arrowup': 'btn_act_w',
                's': 'btn_act_s', 'arrowdown': 'btn_act_s',
                'a': 'btn_act_a', 'arrowleft': 'btn_act_a',
                'd': 'btn_act_d', 'arrowright': 'btn_act_d',
                'q': 'btn_act_q',
                'e': 'btn_act_e',
                ' ': 'btn_act_space',
                'c': 'btn_act_c',
                'x': 'btn_act_x'
            };
            const btnId = keyMap[key];
            if (btnId) {
                if (key === ' ') e.preventDefault();
                const el = document.getElementById(btnId);
                if (el) {
                    const btn = el.tagName === 'BUTTON' ? el : (el.querySelector('button') || el);
                    btn.classList.add('btn-pressed-anim');
                    setTimeout(() => btn.classList.remove('btn-pressed-anim'), 200);
                    btn.click();
                }
            }
        });
    }
    """

    custom_html = """
    <style>
    .pad-btn button {
        font-size: 1.15rem !important;
        font-weight: bold !important;
        padding: 14px 6px !important;
        border-radius: 8px !important;
        transition: all 0.1s ease !important;
    }
    .btn-main-act button {
        background: #e65100 !important;
        color: white !important;
    }
    .btn-pressed-anim button {
        transform: scale(0.92) !important;
        background: #ff9800 !important;
        color: white !important;
        box-shadow: 0 0 16px rgba(255, 152, 0, 0.8) !important;
    }
    #game_viewport video {
        width: 100% !important;
        max-height: 500px !important;
        object-fit: contain !important;
        border-radius: 12px !important;
        background: #000 !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.7) !important;
    }
    </style>
    <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" style="display:none;" onload="if(!window._lingbot_installed){window._lingbot_installed=true; window.addEventListener('keydown', function(e){ const a=document.activeElement; if(a&&(a.tagName==='INPUT'||a.tagName==='TEXTAREA'))return; const k=e.key.toLowerCase(); const m={'w':'btn_act_w','arrowup':'btn_act_w','s':'btn_act_s','arrowdown':'btn_act_s','a':'btn_act_a','arrowleft':'btn_act_a','d':'btn_act_d','arrowright':'btn_act_d','q':'btn_act_q','e':'btn_act_e',' ':'btn_act_space','c':'btn_act_c','x':'btn_act_x'}; const id=m[k]; if(id){if(k===' ')e.preventDefault(); const el=document.getElementById(id); if(el){const b=el.tagName==='BUTTON'?el:(el.querySelector('button')||el); b.click();}} }); console.log('🎮 Inline key listener ready!'); }">
    """

    examples = get_available_examples()

    with gr.Blocks(title="LingBot-World 2.0 Infinity") as demo:
        gr.HTML(custom_html)

        gr.Markdown(
            """
            # 🌐 LingBot-World 2.0 (LingBot-World-Infinity)
            ### Causal Interactive World Simulator & Real-time Explorer
            * **🎮 실시간 3D 월드 게임 탐험**: 이미지 1장을 넣고 키보드 **[W, A, S, D, Q, E, Space, C]** 또는 방향키(↑, ↓, ←, →)를 누르면 게임처럼 3D 공간을 실시간으로 걸어 다니며 탐험합니다!
            * **인메모리 GPU 초고속 렌더링**: 모델 상주형 엔진으로 1스텝당 **단 ~2.5초** 만에 실시간 생성됩니다.
            """
        )

        with gr.Tabs():
            # ==============================================================
            # TAB 1: INTERACTIVE WORLD EXPLORER (메인 실시간 탐험 게임 모드)
            # ==============================================================
            with gr.Tab("🎮 실시간 월드 탐험 (Interactive World Explorer)"):
                with gr.Row():
                    # 좌측 제어 패널 (Controls)
                    with gr.Column(scale=4):
                        gr.Markdown("### 1️⃣ 월드 설정 & 시작")
                        inter_preset = gr.Dropdown(
                            choices=examples,
                            label="예제 프리셋 (Examples)",
                            value="00" if "00" in examples else (examples[0] if examples else None),
                        )
                        inter_image = gr.Image(label="시작 이미지 (First Frame Image)", type="pil", height=200)
                        inter_prompt = gr.Textbox(
                            label="월드 환경 프롬프트 (Prompt / Director)",
                            lines=2,
                            placeholder="세계관 또는 현재 보고 있는 풍경을 묘사하세요...",
                            value="The video presents a soaring journey through a fantasy jungle towards an ancient gothic castle.",
                        )
                        with gr.Row():
                            inter_seed = gr.Number(label="시드 (Seed)", value=42, precision=0, scale=1)
                            btn_start_world = gr.Button("🚀 탐험 시작 / 월드 초기화", variant="primary", scale=2)

                        gr.Markdown("### 2️⃣ 실시간 WASD 조작 패드")
                        gr.Markdown(
                            """
                            *🕹️ **키보드 조작 안내**:*
                            * **W / ↑**: 전진 | **S / ↓**: 후진
                            * **A / ←**: 좌이동 | **D / →**: 우이동
                            * **Q / E**: 좌/우 회전 | **Space / C**: 상승/하강
                            """
                        )
                        
                        with gr.Group():
                            with gr.Row():
                                btn_q = gr.Button("⤹ Q: 좌회전", elem_id="btn_act_q", elem_classes=["pad-btn"])
                                btn_w = gr.Button("⬆️ W: 전진", elem_id="btn_act_w", variant="primary", elem_classes=["pad-btn", "btn-main-act"])
                                btn_e = gr.Button("⤸ E: 우회전", elem_id="btn_act_e", elem_classes=["pad-btn"])
                            with gr.Row():
                                btn_a = gr.Button("⬅️ A: 좌이동", elem_id="btn_act_a", elem_classes=["pad-btn"])
                                btn_x = gr.Button("⏸️ X: 정지", elem_id="btn_act_x", elem_classes=["pad-btn"])
                                btn_d = gr.Button("➡️ D: 우이동", elem_id="btn_act_d", elem_classes=["pad-btn"])
                            with gr.Row():
                                btn_space = gr.Button("🔼 Space: 상승", elem_id="btn_act_space", elem_classes=["pad-btn"])
                                btn_s = gr.Button("⬇️ S: 후진", elem_id="btn_act_s", elem_classes=["pad-btn"])
                                btn_c = gr.Button("🔽 C: 하강", elem_id="btn_act_c", elem_classes=["pad-btn"])

                        with gr.Accordion("⚙️ 이동 미세 설정", open=False):
                            step_size = gr.Slider(0.1, 1.5, value=0.5, step=0.1, label="스텝 이동 거리 (Step Size)")
                            turn_angle = gr.Slider(5.0, 30.0, value=12.0, step=1.0, label="회전 각도 (Turn Angle Deg)")

                    # 우측 메인 게임 뷰포트 (The Hero Game Screen)
                    with gr.Column(scale=8):
                        gr.Markdown("### 🎥 3D 인터랙티브 월드 뷰포트 (Interactive Viewport)")
                        
                        # THE SINGLE DOMINANT GAME SCREEN
                        inter_main_video = gr.Video(
                            label="3D 실시간 월드 화면 (Live Screen)",
                            autoplay=True,
                            loop=True,
                            height=480,
                            elem_id="game_viewport",
                            interactive=False,
                        )
                        
                        inter_history_txt = gr.Textbox(
                            label="📍 탐험 경로 타임라인 (Action Path)",
                            value="🏁 [대기 중] 좌측의 [🚀 탐험 시작]을 누르면 3D 시뮬레이션이 가동됩니다.",
                            interactive=False,
                        )
                        inter_status_log = gr.Textbox(
                            label="⚡ 실시간 엔진 상태 & 성능 로그",
                            lines=3,
                            interactive=False,
                            value="[준비] 이미지를 선택하고 [🚀 탐험 시작 / 월드 초기화] 버튼을 눌러주세요."
                        )

                        with gr.Accordion("🎬 누적 전체 여정 풀영상 다시보기 및 다운로드", open=False):
                            inter_full_video = gr.Video(label="처음부터 지금까지 탐험한 전체 연속 비디오", autoplay=False)

                # Event handlers for Tab 1
                def on_select_preset(preset_name):
                    if not preset_name:
                        return None, ""
                    _, img, pr = load_example_data(preset_name)
                    return img, pr

                inter_preset.change(
                    fn=on_select_preset,
                    inputs=[inter_preset],
                    outputs=[inter_image, inter_prompt]
                )

                btn_start_world.click(
                    fn=handle_start_world,
                    inputs=[inter_image, inter_prompt, inter_seed],
                    outputs=[
                        inter_main_video,
                        inter_full_video,
                        inter_status_log,
                        inter_history_txt,
                    ]
                )

                # Bind all 9 action buttons using safe factory function
                for btn, act_key in [
                    (btn_w, "W"), (btn_s, "S"), (btn_a, "A"), (btn_d, "D"),
                    (btn_q, "Q"), (btn_e, "E"), (btn_space, "SPACE"),
                    (btn_c, "C"), (btn_x, "IDLE"),
                ]:
                    btn.click(
                        fn=make_step_handler(act_key),
                        inputs=[inter_prompt, step_size, turn_angle],
                        outputs=[
                            inter_main_video,
                            inter_full_video,
                            inter_status_log,
                            inter_history_txt,
                        ]
                    )

            # ==============================================================
            # TAB 2: CLASSIC VIDEO GENERATOR (기존 일괄 생성 모드 유지)
            # ==============================================================
            with gr.Tab("🎬 클래식 비디오 생성 (Classic Generator)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 모델 설정")
                        task = gr.Dropdown(
                            choices=["i2v-1.3B", "i2v-A14B"],
                            value="i2v-1.3B",
                            label="모델 태스크 (Task)",
                            info="1.3B 모델은 단일 GPU에서 비교적 가볍게 실행 가능합니다."
                        )
                        infer_mode = gr.Radio(
                            choices=["causal_fast", "causal_pretrain"],
                            value="causal_fast",
                            label="추론 모드 (Infer Mode)",
                            info="causal_fast: 4단계 고속 생성 | causal_pretrain: 40단계 CFG"
                        )
                        ckpt_dir = gr.Textbox(
                            label="체크포인트 디렉터리 (Checkpoint Dir)",
                            value="lingbot-world-v2-1.3b-causal-fast"
                        )
                        assets_dir = gr.Textbox(
                            label="공동 에셋 디렉터리 (Assets Dir - T5/VAE)",
                            value="lingbot-world-v2-14b-causal-fast",
                            info="1.3B는 T5/VAE를 14B 체크포인트와 공유합니다."
                        )

                        gr.Markdown("### 📁 예제 프리셋 불러오기")
                        classic_preset = gr.Dropdown(
                            choices=examples,
                            label="예제 프리셋 (Examples)",
                            value="00" if "00" in examples else (examples[0] if examples else None),
                        )

                        gr.Markdown("### 🎬 입력 데이터")
                        classic_image = gr.Image(label="시작 이미지 (First Frame Image)", type="pil")
                        action_path = gr.Textbox(
                            label="액션 / 카메라 경로 (Action/Camera Directory)",
                            placeholder="/path/to/examples/00",
                            value=str(EXAMPLES_DIR / "00") if (EXAMPLES_DIR / "00").exists() else ""
                        )
                        classic_prompt = gr.Textbox(
                            label="텍스트 프롬프트 (Prompt)",
                            lines=3,
                            placeholder="생성할 영상의 장면을 상세히 설명하세요..."
                        )

                        with gr.Accordion("고급 생성 옵션 (Advanced Options)", open=False):
                            size = gr.Dropdown(
                                choices=["480*832", "720*1280", "1280*720"],
                                value="480*832",
                                label="해상도 (Resolution)"
                            )
                            frame_num = gr.Slider(
                                minimum=17,
                                maximum=361,
                                step=4,
                                value=81,
                                label="생성 프레임 수 (Frame Num - 4n+1)",
                                info="기본값 81 (약 5초 영상, 16 fps)"
                            )
                            classic_seed = gr.Number(label="랜덤 시드 (Seed)", value=42, precision=0)
                            offload_model = gr.Checkbox(label="모델 CPU 오프로드 (Offload to save VRAM)", value=True)
                            t5_cpu = gr.Checkbox(label="T5 CPU 실행 (T5 on CPU)", value=True)

                        btn_generate = gr.Button("🚀 비디오 생성 시작", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🎥 생성 결과")
                        output_video = gr.Video(label="생성된 비디오 (Generated Video)")
                        classic_status_log = gr.Textbox(label="상태 및 로그 (Status & Logs)", lines=10, interactive=False)

                classic_preset.change(
                    fn=lambda name: load_example_data(name),
                    inputs=[classic_preset],
                    outputs=[action_path, classic_image, classic_prompt]
                )

                btn_generate.click(
                    fn=run_classic_inference,
                    inputs=[
                        task, infer_mode, ckpt_dir, assets_dir,
                        classic_image, action_path, classic_prompt,
                        size, frame_num, classic_seed, offload_model, t5_cpu
                    ],
                    outputs=[output_video, classic_status_log]
                )

        # Register client keyboard listener via demo.load
        demo.load(None, None, None, js=keyboard_client_js)

    return demo

def main():
    parser = argparse.ArgumentParser(description="LingBot-World-V2 Interactive Gradio UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 LingBot-World 2.0 Infinity Interactive Web UI 서버 시작")
    print(f"🔗 로컬 서버 주소: http://localhost:{args.port} 또는 http://127.0.0.1:{args.port}")
    print(f"🔗 외부 접속 바인딩: http://{args.host}:{args.port}")
    print("🎮 기능: 실시간 WASD 3D 월드 탐험 (Game Viewport) + 클래식 비디오 생성기")
    print("=" * 60)

    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )

if __name__ == "__main__":
    main()
