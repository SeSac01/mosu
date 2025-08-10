#!/usr/bin/env python3
"""
모델 비교 및 추가 학습 유틸리티
"""

import torch
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_config import AdvancedTrainingConfig, TrainingStageConfig
from advanced_trainer import AdvancedSignLanguageTrainer
from sign_language_model import SequenceToSequenceSignModel

logger = logging.getLogger(__name__)

class ModelComparator:
    """모델 성능 비교 및 분석"""
    
    def __init__(self, checkpoint_dir: str = "./advanced_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        
    def load_model_info(self, model_path: Path) -> Dict:
        """모델 정보 로드"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            return {
                'path': str(model_path),
                'stage': checkpoint.get('stage', 'unknown'),
                'epoch': checkpoint.get('epoch', -1),
                'val_loss': checkpoint.get('val_loss', float('inf')),
                'val_accuracy': checkpoint.get('val_accuracy', 0.0),
                'config': checkpoint.get('config', {}),
                'model_size': sum(p.numel() for p in checkpoint['model_state_dict'].values())
            }
        except Exception as e:
            logger.error(f"모델 로드 실패 {model_path}: {e}")
            return None
    
    def compare_models(self, experiment_name: Optional[str] = None) -> pd.DataFrame:
        """저장된 모델들 비교"""
        if experiment_name:
            model_pattern = f"*{experiment_name}*/best_model_stage_*.pt"
        else:
            model_pattern = "best_model_stage_*.pt"
        
        model_files = list(self.checkpoint_dir.glob(model_pattern))
        if not model_files:
            model_files = list(self.checkpoint_dir.rglob("best_model_stage_*.pt"))
        
        logger.info(f"발견된 모델: {len(model_files)}개")
        
        model_infos = []
        for model_path in sorted(model_files):
            info = self.load_model_info(model_path)
            if info:
                model_infos.append(info)
        
        if not model_infos:
            logger.warning("비교할 모델이 없습니다.")
            return pd.DataFrame()
        
        # DataFrame 생성
        df = pd.DataFrame(model_infos)
        
        # 정렬 (검증 정확도 기준)
        df = df.sort_values('val_accuracy', ascending=False)
        
        return df
    
    def print_comparison(self, df: pd.DataFrame):
        """모델 비교 결과 출력"""
        if df.empty:
            print("비교할 모델이 없습니다.")
            return
        
        print("\n" + "="*80)
        print("📊 모델 성능 비교")
        print("="*80)
        
        for idx, row in df.iterrows():
            print(f"\n🏆 순위 {idx+1}: {Path(row['path']).name}")
            print(f"  단계: {row['stage']}")
            print(f"  검증 손실: {row['val_loss']:.4f}")
            print(f"  검증 정확도: {row['val_accuracy']:.3f}")
            print(f"  모델 크기: {row['model_size']:,} 파라미터")
            if isinstance(row['config'], dict) and 'label_smoothing' in row['config']:
                print(f"  라벨 스무딩: {row['config']['label_smoothing']}")
                print(f"  드롭아웃: {row['config'].get('dropout_rate', 'N/A')}")
                print(f"  증강 활성화: {row['config'].get('enable_augmentation', False)}")
        
        print("\n" + "="*80)
        print(f"🥇 최고 성능: {df.iloc[0]['stage']} (정확도: {df.iloc[0]['val_accuracy']:.3f})")
        print("="*80)
    
    def plot_comparison(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """모델 성능 시각화"""
        if df.empty or len(df) < 2:
            logger.warning("시각화할 데이터가 부족합니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('모델 성능 비교', fontsize=16)
        
        # 1. 검증 정확도 비교
        axes[0,0].bar(range(len(df)), df['val_accuracy'])
        axes[0,0].set_title('검증 정확도')
        axes[0,0].set_xlabel('모델 순위')
        axes[0,0].set_ylabel('정확도')
        axes[0,0].set_xticks(range(len(df)))
        axes[0,0].set_xticklabels([f"Stage {i+1}" for i in range(len(df))])
        
        # 2. 검증 손실 비교
        axes[0,1].bar(range(len(df)), df['val_loss'])
        axes[0,1].set_title('검증 손실')
        axes[0,1].set_xlabel('모델 순위')
        axes[0,1].set_ylabel('손실')
        axes[0,1].set_xticks(range(len(df)))
        axes[0,1].set_xticklabels([f"Stage {i+1}" for i in range(len(df))])
        
        # 3. 정확도 vs 손실
        axes[1,0].scatter(df['val_loss'], df['val_accuracy'])
        axes[1,0].set_title('정확도 vs 손실')
        axes[1,0].set_xlabel('검증 손실')
        axes[1,0].set_ylabel('검증 정확도')
        
        # 4. 모델 크기 비교
        axes[1,1].bar(range(len(df)), df['model_size'] / 1e6)  # 백만 파라미터 단위
        axes[1,1].set_title('모델 크기 (M 파라미터)')
        axes[1,1].set_xlabel('모델 순위')
        axes[1,1].set_ylabel('파라미터 수 (백만)')
        axes[1,1].set_xticks(range(len(df)))
        axes[1,1].set_xticklabels([f"Stage {i+1}" for i in range(len(df))])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"차트 저장: {save_path}")
        
        plt.show()

class ContinualLearningManager:
    """지속적 학습 관리자"""
    
    def __init__(self, base_config: AdvancedTrainingConfig):
        self.base_config = base_config
        
    def create_transfer_config(self, 
                             source_model_path: str,
                             target_stages: List[TrainingStageConfig],
                             experiment_suffix: str = "transfer") -> AdvancedTrainingConfig:
        """전이 학습 설정 생성"""
        config = AdvancedTrainingConfig()
        
        # 기본 설정 복사
        config.random_seed = self.base_config.random_seed
        config.data_split = self.base_config.data_split
        config.annotation_path = self.base_config.annotation_path
        config.pose_data_dir = self.base_config.pose_data_dir
        
        # 새 실험 이름
        config.experiment_name = f"{self.base_config.experiment_name}_{experiment_suffix}"
        
        # 체크포인트 및 로그 디렉토리
        config.checkpoint_dir = f"./advanced_checkpoints/{config.experiment_name}"
        config.log_dir = f"./advanced_logs/{config.experiment_name}"
        
        # 전이 학습 단계 설정
        config.multi_stage.stages = target_stages
        config.multi_stage.improvement_threshold = -1.0  # 모든 단계 실행
        
        # 소스 모델 정보 저장 (메타데이터로)
        config.source_model_path = source_model_path
        
        return config
    
    def create_fine_tuning_stages(self, 
                                base_lr: float = 1e-5,
                                num_epochs: int = 5) -> List[TrainingStageConfig]:
        """미세 조정 단계 생성"""
        return [
            # 단계 1: 전체 모델 미세 조정 (낮은 학습률)
            TrainingStageConfig(
                name="full_fine_tuning",
                description="전체 모델 미세 조정",
                num_epochs=num_epochs,
                batch_size=16,
                learning_rate=base_lr,
                enable_augmentation=True,
                augmentation_strength=0.5,
                dropout_rate=0.1,
                label_smoothing=0.05
            ),
            
            # 단계 2: 분류기만 미세 조정 (높은 학습률)
            TrainingStageConfig(
                name="classifier_fine_tuning",
                description="분류기 레이어만 미세 조정",
                num_epochs=num_epochs,
                batch_size=16,
                learning_rate=base_lr * 5,
                enable_augmentation=True,
                augmentation_strength=0.7,
                dropout_rate=0.15,
                label_smoothing=0.1,
                freeze_encoder=True  # 인코더 동결
            )
        ]
    
    def create_domain_adaptation_stages(self, 
                                      base_lr: float = 5e-6) -> List[TrainingStageConfig]:
        """도메인 적응 단계 생성"""
        return [
            # 단계 1: 점진적 해제 (Gradual Unfreezing)
            TrainingStageConfig(
                name="gradual_unfreezing",
                description="점진적 레이어 해제",
                num_epochs=3,
                batch_size=12,
                learning_rate=base_lr,
                enable_augmentation=True,
                augmentation_strength=1.2,
                dropout_rate=0.2,
                label_smoothing=0.15
            ),
            
            # 단계 2: 전체 모델 적응
            TrainingStageConfig(
                name="full_adaptation", 
                description="전체 모델 도메인 적응",
                num_epochs=5,
                batch_size=8,
                learning_rate=base_lr * 2,
                enable_augmentation=True,
                augmentation_strength=0.8,
                dropout_rate=0.1,
                label_smoothing=0.05
            )
        ]
    
    def run_transfer_learning(self, 
                            source_model_path: str,
                            transfer_type: str = "fine_tuning",
                            **kwargs) -> Dict:
        """전이 학습 실행"""
        logger.info(f"🔄 전이 학습 시작: {transfer_type}")
        logger.info(f"소스 모델: {source_model_path}")
        
        # 전이 학습 단계 생성
        if transfer_type == "fine_tuning":
            stages = self.create_fine_tuning_stages(**kwargs)
        elif transfer_type == "domain_adaptation":
            stages = self.create_domain_adaptation_stages(**kwargs)
        else:
            raise ValueError(f"지원하지 않는 전이 학습 타입: {transfer_type}")
        
        # 설정 생성
        config = self.create_transfer_config(
            source_model_path=source_model_path,
            target_stages=stages,
            experiment_suffix=transfer_type
        )
        
        # 트레이너 생성
        trainer = AdvancedSignLanguageTrainer(config)
        
        # 소스 모델 로드 (필요시)
        # TODO: 모델 가중치 사전 로드 기능 구현
        
        # 전이 학습 실행
        results = trainer.train_multi_stage()
        
        logger.info("✅ 전이 학습 완료")
        return results

def main():
    """메인 함수 - 모델 비교 데모"""
    logging.basicConfig(level=logging.INFO)
    
    # 모델 비교
    comparator = ModelComparator()
    df = comparator.compare_models()
    comparator.print_comparison(df)
    
    if len(df) > 1:
        comparator.plot_comparison(df, "model_comparison.png")
    
    # 지속적 학습 예제 (주석 처리)
    # base_config = AdvancedTrainingConfig()
    # manager = ContinualLearningManager(base_config)
    
    # if len(df) > 0:
    #     best_model_path = df.iloc[0]['path']
    #     results = manager.run_transfer_learning(
    #         source_model_path=best_model_path,
    #         transfer_type="fine_tuning",
    #         base_lr=1e-5,
    #         num_epochs=2
    #     )
    #     print("전이 학습 결과:", results)

if __name__ == "__main__":
    main()
