#!/usr/bin/env python3
"""
저장된 모델을 기반으로 추가 학습을 수행하는 유틸리티
"""

import argparse
import logging
import torch
from pathlib import Path
from typing import Dict, Optional

from advanced_config import AdvancedTrainingConfig, TrainingStageConfig
from advanced_trainer import AdvancedSignLanguageTrainer
from sign_language_model import SequenceToSequenceSignModel

class ModelContinuationTrainer:
    """저장된 모델을 기반으로 추가 학습을 수행하는 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def list_available_models(self, checkpoint_dir: str = "./advanced_checkpoints") -> Dict:
        """사용 가능한 저장된 모델들을 나열"""
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            self.logger.warning(f"체크포인트 디렉토리가 존재하지 않습니다: {checkpoint_dir}")
            return {}
        
        models = {}
        
        # 단계별 모델들 찾기 (stage_*.pt)
        stage_models = list(checkpoint_path.glob("stage_*.pt"))
        for model_file in sorted(stage_models):
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                models[model_file.stem] = {
                    'path': model_file,
                    'stage_idx': checkpoint.get('stage_idx', 0),
                    'stage_name': checkpoint.get('stage_name', 'unknown'),
                    'description': checkpoint.get('stage_description', ''),
                    'val_loss': checkpoint.get('val_loss', float('inf')),
                    'val_accuracy': checkpoint.get('val_accuracy', 0.0),
                    'training_time': checkpoint.get('training_time', 0.0)
                }
            except Exception as e:
                self.logger.warning(f"모델 로드 실패: {model_file} - {e}")
        
        # best_model 파일들도 찾기 (stage_*.pt가 없는 경우 대안으로)
        if not models:
            best_models = list(checkpoint_path.glob("best_model_stage_*.pt"))
            for model_file in sorted(best_models):
                try:
                    checkpoint = torch.load(model_file, map_location='cpu')
                    stage_num = model_file.stem.split('_')[-1]  # stage_X에서 X 추출
                    models[model_file.stem] = {
                        'path': model_file,
                        'stage_idx': int(stage_num) if stage_num.isdigit() else 0,
                        'stage_name': checkpoint.get('stage', 'unknown'),
                        'description': f"Best model from stage {stage_num}",
                        'val_loss': checkpoint.get('val_loss', float('inf')),
                        'val_accuracy': checkpoint.get('val_accuracy', 0.0),
                        'training_time': 0.0
                    }
                except Exception as e:
                    self.logger.warning(f"모델 로드 실패: {model_file} - {e}")
        
        return models
    
    def load_model_for_continuation(self, model_path: str, new_config: AdvancedTrainingConfig) -> SequenceToSequenceSignModel:
        """저장된 모델을 로드하여 추가 학습 준비"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 모델 아키텍처 정보 가져오기
        stage_config = checkpoint.get('stage_config', {})
        
        # 새로운 모델 생성 (기본 설정 사용)
        # 실제로는 데이터셋에서 vocab_size를 가져와야 함
        model = SequenceToSequenceSignModel(
            vocab_size=442,  # 기본값, 실제로는 데이터셋에서 가져와야 함
            embed_dim=256,
            num_encoder_layers=6,
            num_decoder_layers=4,
            num_heads=8,
            dim_feedforward=1024,
            dropout=stage_config.get('dropout_rate', 0.1),
            max_seq_len=500,
            label_smoothing=stage_config.get('label_smoothing', 0.0)
        )
        
        # 저장된 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"✅ 모델 로드 완료: {model_path}")
        self.logger.info(f"  Stage: {checkpoint.get('stage_name', 'unknown')}")
        self.logger.info(f"  Val Loss: {checkpoint.get('val_loss', 0):.4f}")
        self.logger.info(f"  Val Acc: {checkpoint.get('val_accuracy', 0):.3f}")
        
        return model
    
    def create_continuation_config(self, base_model_info: Dict, continuation_type: str = "fine_tune") -> AdvancedTrainingConfig:
        """추가 학습을 위한 설정 생성"""
        config = AdvancedTrainingConfig()
        
        # 실험명 설정
        config.experiment_name = f"continuation_{base_model_info['stage_name']}_{continuation_type}"
        
        # 추가 학습 유형별 설정
        if continuation_type == "fine_tune":
            # 미세 조정
            stages = [
                TrainingStageConfig(
                    name="fine_tune_continuation",
                    description=f"Stage {base_model_info['stage_idx']} 모델 미세 조정",
                    num_epochs=10,
                    batch_size=16,
                    learning_rate=5e-6,  # 매우 작은 학습률
                    enable_augmentation=True,
                    augmentation_strength=0.3,
                    dropout_rate=0.05,
                    label_smoothing=0.05
                )
            ]
        elif continuation_type == "regularization":
            # 정규화 강화
            stages = [
                TrainingStageConfig(
                    name="regularization_continuation",
                    description=f"Stage {base_model_info['stage_idx']} 모델 정규화 강화",
                    num_epochs=15,
                    batch_size=24,
                    learning_rate=2e-5,
                    enable_augmentation=True,
                    augmentation_strength=1.2,
                    dropout_rate=0.25,
                    label_smoothing=0.15
                )
            ]
        elif continuation_type == "exploration":
            # 탐색적 학습
            stages = [
                TrainingStageConfig(
                    name="exploration_continuation",
                    description=f"Stage {base_model_info['stage_idx']} 모델 탐색적 학습",
                    num_epochs=20,
                    batch_size=32,
                    learning_rate=1e-4,
                    enable_augmentation=True,
                    augmentation_strength=1.0,
                    dropout_rate=0.15,
                    label_smoothing=0.1
                )
            ]
        else:
            raise ValueError(f"지원하지 않는 continuation_type: {continuation_type}")
        
        config.multi_stage.stages = stages
        config.multi_stage.improvement_threshold = -1.0  # 모든 단계 진행
        
        return config
    
    def continue_training(self, model_name: str, continuation_type: str = "fine_tune", 
                         checkpoint_dir: str = "./advanced_checkpoints") -> Dict:
        """저장된 모델로부터 추가 학습 수행"""
        # 사용 가능한 모델 확인
        available_models = self.list_available_models(checkpoint_dir)
        
        if model_name not in available_models:
            raise ValueError(f"모델을 찾을 수 없습니다: {model_name}")
        
        model_info = available_models[model_name]
        
        # 모델 로드
        config = self.create_continuation_config(model_info, continuation_type)
        model = self.load_model_for_continuation(str(model_info['path']), config)
        
        # 트레이너 생성 및 학습
        trainer = AdvancedSignLanguageTrainer(config)
        
        # 모델을 트레이너에 설정 (추가 학습을 위해)
        trainer.model = model
        
        # 추가 학습 실행
        results = trainer.train_multi_stage()
        
        return results

def main():
    parser = argparse.ArgumentParser(description="저장된 모델 기반 추가 학습")
    parser.add_argument("--list", action="store_true", help="사용 가능한 모델 목록 출력")
    parser.add_argument("--model", type=str, help="추가 학습할 모델 이름")
    parser.add_argument("--type", choices=["fine_tune", "regularization", "exploration"],
                       default="fine_tune", help="추가 학습 유형")
    parser.add_argument("--checkpoint-dir", type=str, default="./advanced_checkpoints",
                       help="체크포인트 디렉토리")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    trainer = ModelContinuationTrainer()
    
    if args.list:
        # 사용 가능한 모델 목록 출력
        models = trainer.list_available_models(args.checkpoint_dir)
        
        if not models:
            print("❌ 사용 가능한 모델이 없습니다.")
            return
        
        print("📋 사용 가능한 모델들:")
        print("="*80)
        for model_name, info in models.items():
            print(f"🔹 {model_name}")
            print(f"   Stage: {info['stage_name']} ({info['description']})")
            print(f"   Performance: Val Loss {info['val_loss']:.4f}, Val Acc {info['val_accuracy']:.3f}")
            print(f"   Training Time: {info['training_time']:.1f}초")
            print()
    
    elif args.model:
        # 추가 학습 수행
        try:
            results = trainer.continue_training(args.model, args.type, args.checkpoint_dir)
            print(f"✅ '{args.model}' 모델의 추가 학습 완료!")
        except Exception as e:
            print(f"❌ 추가 학습 실패: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
