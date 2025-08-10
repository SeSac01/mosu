"""
고급 다단계 학습 실행 스크립트
"""
import argparse
import logging
import sys
from pathlib import Path

from advanced_config import (
    AdvancedTrainingConfig, TrainingStageConfig, 
    DataSplitConfig, EarlyStoppingConfig, MultiStageTrainingConfig
)
from advanced_trainer import AdvancedSignLanguageTrainer

def setup_logging(log_level: str = "INFO"):
    """로깅 설정 - 훈련 진행상황을 모니터링하기 위해 INFO 레벨 유지"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('advanced_training.log', encoding='utf-8')
        ]
    )

def create_custom_config(args) -> AdvancedTrainingConfig:
    """커스텀 설정 생성"""
    config = AdvancedTrainingConfig()
    
    # 명령행 인수로 설정 오버라이드
    if args.seed:
        config.random_seed.seed = args.seed
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
        
    if args.data_dir:
        config.pose_data_dir = args.data_dir
        
    if args.annotation_path:
        config.annotation_path = args.annotation_path
    
    # 디바이스 설정
    if args.device:
        config.device = args.device
    
    # 멀티 GPU 설정 - 기본값 유지하되, 명령행에서 비활성화 가능
    if hasattr(args, 'multi_gpu') and args.multi_gpu is not None:
        config.multi_gpu = args.multi_gpu
    # 기본값인 True가 그대로 사용됨 (자동 감지)
        
    if hasattr(args, 'data_parallel') and args.data_parallel is not None:
        config.use_data_parallel = args.data_parallel
    
    # 데이터 분할 비율 설정
    if args.train_ratio:
        config.data_split.train_ratio = args.train_ratio
        config.data_split.val_ratio = (1.0 - args.train_ratio) * 0.75
        config.data_split.test_ratio = (1.0 - args.train_ratio) * 0.25
    
    # 얼리스탑 설정
    if args.patience:
        config.early_stopping.patience = args.patience
    
    # 빠른 테스트 모드
    if args.quick_test:
        # 단계 수 줄이기
        config.multi_stage.stages = config.multi_stage.stages[:2]
        for stage in config.multi_stage.stages:
            stage.num_epochs = min(3, stage.num_epochs)
            stage.batch_size = min(16, stage.batch_size)
        config.early_stopping.patience = 3
        config.experiment_name = "quick_test"
    
    # 커스텀 단계 설정
    if args.stages_config:
        config = create_stages_from_config(config, args.stages_config)
    
    return config

def create_stages_from_config(config: AdvancedTrainingConfig, stages_config: str) -> AdvancedTrainingConfig:
    """설정 파일에서 단계 구성 로드"""
    if stages_config == "conservative":
        # 보수적인 학습 (작은 학습률, 긴 학습)
        config.multi_stage.stages = [
            TrainingStageConfig(
                name="conservative_baseline",
                description="보수적 기본 학습",
                num_epochs=50,
                batch_size=32,
                learning_rate=5e-5,
                enable_augmentation=False
            ),
            TrainingStageConfig(
                name="conservative_augmentation",
                description="보수적 증강 학습",
                num_epochs=40,
                batch_size=24,
                learning_rate=2e-5,
                enable_augmentation=True,
                augmentation_strength=0.7
            )
        ]
    elif stages_config == "aggressive":
        # 공격적인 학습 (큰 학습률, 강한 증강)
        config.multi_stage.stages = [
            TrainingStageConfig(
                name="aggressive_baseline",
                description="공격적 기본 학습",
                num_epochs=20,
                batch_size=48,
                learning_rate=2e-4,
                enable_augmentation=False
            ),
            TrainingStageConfig(
                name="aggressive_augmentation",
                description="공격적 증강 학습",
                num_epochs=25,
                batch_size=32,
                learning_rate=1e-4,
                enable_augmentation=True,
                augmentation_strength=1.3
            ),
            TrainingStageConfig(
                name="aggressive_regularization",
                description="강한 정규화",
                num_epochs=15,
                batch_size=24,
                learning_rate=5e-5,
                enable_augmentation=True,
                dropout_rate=0.3,
                label_smoothing=0.15
            )
        ]
    elif stages_config == "minimal":
        # 최소한의 단계 (빠른 실험용)
        config.multi_stage.stages = [
            TrainingStageConfig(
                name="minimal_training",
                description="최소한의 학습",
                num_epochs=15,
                batch_size=32,
                learning_rate=1e-4,
                enable_augmentation=True,
                augmentation_strength=0.8
            )
        ]
    
    return config

def main():
    parser = argparse.ArgumentParser(description="고급 다단계 수화 인식 모델 학습")
    
    # 기본 설정
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--device", choices=["auto", "cuda", "xpu", "cpu"], 
                       default="auto", help="사용할 디바이스")
    parser.add_argument("--multi-gpu", action="store_true",
                       help="멀티 GPU 사용 (CUDA만 지원)")
    parser.add_argument("--no-data-parallel", action="store_false", dest="data_parallel",
                       help="DataParallel 사용 안함 (멀티 GPU 시)")
    parser.add_argument("--experiment-name", type=str, 
                       default="multi_stage_training", help="실험 이름")
    
    # 데이터 설정
    parser.add_argument("--data-dir", type=str, default="./data", 
                       help="포즈 데이터 디렉토리")
    parser.add_argument("--annotation-path", type=str, 
                       default="./data/sign_language_dataset_only_sen_lzf.h5",
                       help="어노테이션 파일 경로")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="훈련 데이터 비율")
    
    # 학습 설정
    parser.add_argument("--patience", type=int, default=15,
                       help="얼리스탑 patience")
    parser.add_argument("--stages-config", 
                       choices=["default", "conservative", "aggressive", "minimal"],
                       default="default", help="학습 단계 구성")
    
    # 실행 옵션
    parser.add_argument("--quick-test", action="store_true",
                       help="빠른 테스트 모드 (작은 에포크 수)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="로그 레벨")
    parser.add_argument("--resume", type=str, help="재개할 체크포인트 경로")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 고급 다단계 수화 인식 모델 학습 시작")
    logger.info("="*80)
    
    try:
        # 설정 생성
        config = create_custom_config(args)
        
        logger.info(f"⚙️ 실험 설정:")
        logger.info(f"  실험명: {config.experiment_name}")
        logger.info(f"  랜덤 시드: {config.random_seed.seed}")
        logger.info(f"  디바이스: {config.device}")
        logger.info(f"  데이터 분할: 훈련 {config.data_split.train_ratio:.1%}, "
                   f"검증 {config.data_split.val_ratio:.1%}, 테스트 {config.data_split.test_ratio:.1%}")
        logger.info(f"  학습 단계 수: {len(config.multi_stage.stages)}")
        logger.info(f"  얼리스탑 patience: {config.early_stopping.patience}")
        
        # 단계별 설정 출력
        logger.info(f"\n📋 학습 단계 계획:")
        for i, stage in enumerate(config.multi_stage.stages):
            logger.info(f"  Stage {i+1}: {stage.name}")
            logger.info(f"    - 설명: {stage.description}")
            logger.info(f"    - 에포크: {stage.num_epochs}, 배치: {stage.batch_size}")
            logger.info(f"    - 학습률: {stage.learning_rate}, 드롭아웃: {stage.dropout_rate}")
            logger.info(f"    - 증강: {'✅' if stage.enable_augmentation else '❌'}")
        
        # 트레이너 생성 및 학습 실행
        trainer = AdvancedSignLanguageTrainer(config)
        results = trainer.train_multi_stage()
        
        logger.info("\n🎉 모든 학습이 성공적으로 완료되었습니다!")
        logger.info(f"실험 결과가 저장되었습니다: {config.checkpoint_dir}")
        
        # 최종 성능 요약
        if 'final_performance' in results:
            final_perf = results['final_performance']
            logger.info(f"\n🏆 최종 성능:")
            logger.info(f"  테스트 정확도: {final_perf['test_accuracy']:.3f}")
            logger.info(f"  테스트 손실: {final_perf['test_loss']:.4f}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 사용자에 의해 학습이 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()