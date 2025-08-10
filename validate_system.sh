#!/bin/bash
cd /workspace01/team02/mosu
echo "🧪 시스템 검증 시작"
echo "=================="

echo "1. GPU 정보 확인"
python3 -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}'); print(f'GPU 수: {torch.cuda.device_count()}')"

echo -e "\n2. 모듈 Import 테스트"
python3 -c "
import sys, os
sys.path.append('.')
try:
    from advanced_config import AdvancedTrainingConfig
    print('✅ AdvancedTrainingConfig import 성공')
    config = AdvancedTrainingConfig()
    print(f'✅ 기본 multi_gpu 설정: {config.multi_gpu}')
    print(f'✅ 총 학습 단계: {len(config.multi_stage.stages)}')
    
    # 라벨 스무딩 확인
    label_smoothing_stages = []
    for i, stage in enumerate(config.multi_stage.stages):
        if stage.label_smoothing > 0:
            label_smoothing_stages.append(f'Stage {i+1}: {stage.label_smoothing}')
    print(f'✅ 라벨 스무딩 단계: {label_smoothing_stages}')
except Exception as e:
    print(f'❌ 설정 테스트 실패: {e}')
"

echo -e "\n3. 디바이스 관리자 테스트"
python3 -c "
import sys, os
sys.path.append('.')
try:
    from device_utils import DeviceManager
    device = DeviceManager.detect_best_device('auto', multi_gpu=True)
    device_info = DeviceManager.get_device_info(device)
    print(f'✅ 디바이스 감지: {device_info[\"name\"]}')
    print(f'✅ 멀티 GPU 가능: {DeviceManager.is_multi_gpu_available()}')
except Exception as e:
    print(f'❌ 디바이스 테스트 실패: {e}')
"

echo -e "\n4. 라벨 스무딩 기능 테스트"
python3 -c "
import sys, os
sys.path.append('.')
try:
    import torch
    from sign_language_model import LabelSmoothingCrossEntropy
    loss_fn = LabelSmoothingCrossEntropy(num_classes=100, smoothing=0.1)
    pred = torch.randn(2, 100)
    target = torch.randint(0, 100, (2,))
    loss = loss_fn(pred, target)
    print(f'✅ 라벨 스무딩 손실 계산: {loss.item():.4f}')
except Exception as e:
    print(f'❌ 라벨 스무딩 테스트 실패: {e}')
"

echo -e "\n=================="
echo "🎉 시스템 검증 완료"
