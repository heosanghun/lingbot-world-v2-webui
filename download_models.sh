#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

HF_CLI="/home/sims/anaconda3/envs/trellis2/bin/huggingface-cli"

echo "============================================================"
echo "🚀 LingBot-World-V2 전체 모델 다운로드 스크립트"
echo "============================================================"

# 1.3B 모델 확인 및 다운로드 (단일 GPU 경량 모델)
echo "📦 [1/2] 1.3B 모델 확인 및 다운로드..."
$HF_CLI download robbyant/lingbot-world-v2-1.3b-causal-fast --local-dir ./lingbot-world-v2-1.3b-causal-fast/transformers

# 14B 메인 플래그십 모델 및 공통 T5/VAE 에셋 (재시도 루프 포함)
echo "📦 [2/2] 14B 메인 모델 및 공통 에셋 다운로드 (~86GB)..."
MAX_RETRIES=5
COUNT=0
until $HF_CLI download robbyant/lingbot-world-v2-14b-causal-fast --local-dir ./lingbot-world-v2-14b-causal-fast; do
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ 다운로드 재시도 횟수 초과 ($MAX_RETRIES 회)."
        exit 1
    fi
    echo "⚠️ 네트워크 일시 지연 발생. 5초 후 자동으로 이어서 다운로드를 재시도합니다 ($COUNT/$MAX_RETRIES)..."
    sleep 5
done

echo "============================================================"
echo "🎉 모든 필수 모델(1.3B 및 14B) 다운로드가 완료되었습니다!"
echo "웹 UI(http://localhost:7860)에서 바로 고품질 월드 시뮬레이터를 실행하실 수 있습니다."
echo "============================================================"

