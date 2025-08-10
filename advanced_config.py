"""
고급 학습 설정 - 다단계 학습을 위한 설정
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
import torch
import os

@dataclass
class RandomSeedConfig:
    """랜덤시드 고정 설정"""
    seed: int = 42
    
    def fix_all_seeds(self):
        """모든 랜덤시드 고정"""
        # Python random
        random.seed(self.seed)
        
        # NumPy random
        np.random.seed(self.seed)
        
        # PyTorch random
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # XPU random (if available)
        try:
            torch.xpu.manual_seed(self.seed)
            torch.xpu.manual_seed_all(self.seed)
        except:
            pass
        
        # 확정적 동작 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 해시 시드 고정
        import os
        os.environ['PYTHONHASHSEED'] = str(self.seed)

@dataclass
class DataSplitConfig:
    """개선된 데이터 분할 설정"""
    train_ratio: float = 0.8
    val_ratio: float = 0.15  
    test_ratio: float = 0.05
    
    # Stratified split (단어별 균등 분할)
    stratified_split: bool = True
    
    # 최소 샘플 수 보장
    min_samples_per_word_train: int = 10
    min_samples_per_word_val: int = 3
    min_samples_per_word_test: int = 2
    
    def __post_init__(self):
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6

@dataclass  
class EarlyStoppingConfig:
    """얼리스탑 설정"""
    patience: int = 15
    min_delta: float = 1e-4
    monitor: str = "val_loss"  # val_loss, val_accuracy, val_word_accuracy
    mode: str = "min"  # min, max
    restore_best_weights: bool = True
    
@dataclass
class TrainingStageConfig:
    """학습 단계별 설정"""
    name: str
    description: str
    
    # 학습 파라미터
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    
    # 증강 설정
    enable_augmentation: bool = False
    augmentation_strength: float = 1.0  # 증강 강도 조절
    
    # 학습률 스케줄링
    use_warmup: bool = True
    warmup_steps: int = 500
    scheduler_type: str = "reduce_on_plateau"  # reduce_on_plateau, cosine, linear
    
    # 정규화 기법
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    
    # 모델 아키텍처 수정
    freeze_encoder: bool = False
    fine_tune_layers: Optional[List[str]] = None

@dataclass
class MultiStageTrainingConfig:
    """다단계 학습 설정"""
    stages: List[TrainingStageConfig] = field(default_factory=list)
    
    # 단계 간 모델 전이
    transfer_best_weights: bool = True
    transfer_optimizer_state: bool = False
    
    # 성능 향상 임계값
    improvement_threshold: float = 0.01  # 1% 이상 향상 시 다음 단계 진행
    
    # 최대 단계 수
    max_stages: int = 5
    
    def __post_init__(self):
        if not self.stages:
            self.stages = self._create_default_stages()
    
    def _create_default_stages(self) -> List[TrainingStageConfig]:
        """기본 학습 단계 생성"""
        return [
            # Stage 1: 기본 학습
            TrainingStageConfig(
                name="baseline",
                description="기본 학습 (증강 없음)",
                num_epochs=30,
                batch_size=48,
                learning_rate=1e-4,
                enable_augmentation=False,
                dropout_rate=0.1
            ),
            
            # Stage 2: 데이터 증강 적용
            TrainingStageConfig(
                name="augmentation",
                description="데이터 증강 적용",
                num_epochs=25,
                batch_size=32,
                learning_rate=5e-5,
                enable_augmentation=True,
                augmentation_strength=1.0,
                dropout_rate=0.15
            ),
            
            # Stage 3: 강한 정규화
            TrainingStageConfig(
                name="regularization",
                description="강한 정규화 (Label Smoothing + Dropout)",
                num_epochs=20,
                batch_size=32,
                learning_rate=2e-5,
                enable_augmentation=True,
                augmentation_strength=0.8,
                dropout_rate=0.2,
                label_smoothing=0.1
            ),
            
            # Stage 4: 미세 조정
            TrainingStageConfig(
                name="fine_tuning",
                description="미세 조정 (낮은 학습률)",
                num_epochs=15,
                batch_size=24,
                learning_rate=1e-5,
                enable_augmentation=True,
                augmentation_strength=0.5,
                dropout_rate=0.1,
                freeze_encoder=False
            ),
            
            # Stage 5: 최종 폴리싱
            TrainingStageConfig(
                name="polishing",
                description="최종 폴리싱 (매우 낮은 학습률)",
                num_epochs=10,
                batch_size=16,
                learning_rate=5e-6,
                enable_augmentation=True,
                augmentation_strength=0.3,
                dropout_rate=0.05
            )
        ]

@dataclass
class AdvancedTrainingConfig:
    """고급 학습 설정 통합"""
    # 기본 설정들
    random_seed: RandomSeedConfig = field(default_factory=RandomSeedConfig)
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    multi_stage: MultiStageTrainingConfig = field(default_factory=MultiStageTrainingConfig)
    
    # 데이터 경로
    annotation_path: str = "./data/sign_language_dataset_only_sen_lzf.h5"
    pose_data_dir: str = "./data"
    
    # 출력 디렉토리
    checkpoint_dir: str = "./advanced_checkpoints"
    log_dir: str = "./advanced_logs"
    
    # 디바이스 설정
    device: str = "auto"
    
    # 멀티 GPU 설정 (임시로 False로 설정하여 안정성 확보)
    multi_gpu: bool = False  # 멀티 GPU 비활성화 (디바이스 동기화 오류 회피)
    use_data_parallel: bool = False  # DataParallel 비활성화
    auto_adjust_batch_size: bool = True  # 멀티 GPU에 맞게 배치 크기 자동 조정
    
    # 실험 관리
    experiment_name: str = "multi_stage_training"
    save_intermediate_results: bool = True
    
    # 평가 설정
    evaluate_on_test: bool = True
    test_every_n_stages: int = 1

# 테스트용 빠른 설정 생성 함수
def create_quick_test_config(epochs_per_stage: int = 1) -> AdvancedTrainingConfig:
    """테스트용 빠른 설정 생성"""
    config = AdvancedTrainingConfig()
    
    # 모든 단계의 에포크를 줄임
    for stage in config.multi_stage.stages:
        stage.num_epochs = epochs_per_stage
        stage.batch_size = min(16, stage.batch_size)  # 배치 크기도 줄임
    
    # 얼리스탑 patience 줄임
    config.early_stopping.patience = 2
    
    # 실험명 변경
    config.experiment_name = f"quick_test_epoch_{epochs_per_stage}"
    
    # 개선 임계값을 매우 작게 설정 (모든 단계 진행)
    config.multi_stage.improvement_threshold = -1.0  # 항상 개선으로 간주
    
    return config

if __name__ == "__main__":
    # 설정 테스트
    config = AdvancedTrainingConfig()
    print("🧪 고급 학습 설정 테스트")
    print(f"총 학습 단계: {len(config.multi_stage.stages)}")
    
    for i, stage in enumerate(config.multi_stage.stages):
        print(f"Stage {i+1}: {stage.name} - {stage.description}")