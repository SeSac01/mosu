#!/usr/bin/env python3
"""
저장된 모델들을 분석하고 비교하는 스크립트
"""

import torch
import json
from pathlib import Path
from typing import Dict, List

def analyze_saved_models(checkpoint_dir: str = "./advanced_checkpoints") -> Dict:
    """저장된 모델들 분석"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ 체크포인트 디렉토리가 존재하지 않습니다: {checkpoint_dir}")
        return {}
    
    results = {}
    
    # best_model 파일들 찾기
    best_models = list(checkpoint_path.glob("best_model_stage_*.pt"))
    
    print(f"📊 저장된 모델 분석 ({len(best_models)}개 모델 발견)")
    print("="*80)
    
    for model_file in sorted(best_models):
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            
            stage_name = checkpoint.get('stage', 'unknown')
            val_loss = checkpoint.get('val_loss', float('inf'))
            val_accuracy = checkpoint.get('val_accuracy', 0.0)
            epoch = checkpoint.get('epoch', 0)
            
            print(f"🔹 {model_file.name}")
            print(f"   Stage: {stage_name}")
            print(f"   Best Epoch: {epoch}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val Accuracy: {val_accuracy:.3f}")
            
            # 모델 파라미터 수 계산
            if 'model_state_dict' in checkpoint:
                total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
                print(f"   Parameters: {total_params:,}")
            
            print()
            
            results[model_file.stem] = {
                'stage': stage_name,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'epoch': epoch,
                'file_path': str(model_file)
            }
            
        except Exception as e:
            print(f"❌ {model_file.name} 로드 실패: {e}")
    
    return results

def compare_models(results: Dict):
    """모델들 성능 비교"""
    if not results:
        return
    
    print("\n📈 모델 성능 비교")
    print("="*80)
    
    # Val Loss 기준 정렬
    sorted_by_loss = sorted(results.items(), key=lambda x: x[1]['val_loss'])
    
    print("🏆 Val Loss 기준 순위:")
    for i, (model_name, info) in enumerate(sorted_by_loss, 1):
        print(f"  {i}위: {info['stage']} - Loss: {info['val_loss']:.4f}, Acc: {info['val_accuracy']:.3f}")
    
    # Val Accuracy 기준 정렬
    sorted_by_acc = sorted(results.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)
    
    print("\n🎯 Val Accuracy 기준 순위:")
    for i, (model_name, info) in enumerate(sorted_by_acc, 1):
        print(f"  {i}위: {info['stage']} - Acc: {info['val_accuracy']:.3f}, Loss: {info['val_loss']:.4f}")
    
    # 단계별 성능 변화
    print("\n📊 단계별 성능 변화:")
    stage_order = ['baseline', 'augmentation', 'regularization', 'fine_tuning', 'polishing']
    
    for stage in stage_order:
        for model_name, info in results.items():
            if info['stage'] == stage:
                print(f"  {stage}: Loss {info['val_loss']:.4f}, Acc {info['val_accuracy']:.3f}")
                break

def create_model_usage_guide():
    """모델 사용 가이드 생성"""
    guide = """
🔧 저장된 모델 사용 가이드

1. 최고 성능 모델 선택:
   - Val Loss가 가장 낮은 모델을 선택
   - 또는 Val Accuracy가 가장 높은 모델을 선택

2. 추가 학습 방법:
   python3 continue_training.py --list  # 사용 가능한 모델 확인
   python3 continue_training.py --model <모델명> --type fine_tune  # 미세 조정
   python3 continue_training.py --model <모델명> --type regularization  # 정규화 강화
   python3 continue_training.py --model <모델명> --type exploration  # 탐색적 학습

3. 모델 로드 방법:
   import torch
   checkpoint = torch.load('advanced_checkpoints/best_model_stage_X.pt')
   model.load_state_dict(checkpoint['model_state_dict'])

4. 추론에 사용:
   model.eval()
   with torch.no_grad():
       outputs = model(pose_features, frame_masks=frame_masks)
"""
    
    print(guide)

def main():
    print("🔍 저장된 모델 분석 시작")
    
    # 모델 분석
    results = analyze_saved_models()
    
    if results:
        # 성능 비교
        compare_models(results)
        
        # 결과 JSON으로 저장
        with open('model_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n💾 분석 결과가 model_analysis.json에 저장되었습니다.")
        
        # 사용 가이드
        create_model_usage_guide()
    
    else:
        print("❌ 분석할 모델이 없습니다.")

if __name__ == "__main__":
    main()
