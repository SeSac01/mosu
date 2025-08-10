"""
고급 다단계 학습 트레이너
"""
import os
import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import numpy as np

from sign_language_model import SequenceToSequenceSignModel
from sign_language_trainer import SignLanguageTrainer
from advanced_config import AdvancedTrainingConfig, TrainingStageConfig
from advanced_data_utils import StratifiedDataSplitter, EarlyStopping
from unified_pose_dataloader import UnifiedSignLanguageDataset
from device_utils import DeviceManager

logger = logging.getLogger(__name__)

class AdvancedSignLanguageTrainer:
    """고급 다단계 학습 트레이너"""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        
        # 랜덤시드 고정
        self.config.random_seed.fix_all_seeds()
        logger.info(f"🎲 랜덤시드 고정: {self.config.random_seed.seed}")
        
        # 디렉토리 생성
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.log_dir = Path(self.config.log_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # 실험 결과 저장
        self.experiment_results = {
            'config': config.__dict__,
            'stages': [],
            'final_performance': {}
        }
        
        # 현재 최고 성능
        self.best_overall_performance = {
            'stage': None,
            'epoch': None,
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'test_performance': {}
        }
        
        # 디바이스 설정 (개선된 방식)
        self.device = DeviceManager.detect_best_device(self.config.device, self.config.multi_gpu)
        device_info = DeviceManager.get_device_info(self.device)
        logger.info(f"🚀 디바이스 설정 완료: {device_info['name']}")
        
        # 멀티 GPU 정보
        self.multi_gpu_available = DeviceManager.is_multi_gpu_available()
        if self.config.multi_gpu and self.multi_gpu_available:
            logger.info(f"🚀 멀티 GPU 모드 활성화: {torch.cuda.device_count()}개 GPU")
        elif self.config.multi_gpu and not self.multi_gpu_available:
            logger.warning("⚠️ 멀티 GPU 요청되었으나 사용 불가 - 단일 GPU/CPU 사용")
            self.config.multi_gpu = False
        
        # 디바이스별 최적화
        DeviceManager.optimize_for_device(self.device, self.config.multi_gpu)
    
    def load_and_prepare_data(self) -> Tuple[UnifiedSignLanguageDataset, StratifiedDataSplitter]:
        """데이터 로드 및 분할 준비"""
        logger.info("📊 데이터 로드 및 분할 준비...")
        
        # 기본 데이터셋 로드
        base_dataset = UnifiedSignLanguageDataset(
            annotation_path=self.config.annotation_path,
            pose_data_dir=self.config.pose_data_dir,
            sequence_length=200,
            min_segment_length=20,
            max_segment_length=300,
            enable_augmentation=False  # 기본 데이터셋은 증강 없음
        )
        
        if len(base_dataset) == 0:
            raise ValueError("유효한 데이터가 없습니다!")
        
        logger.info(f"✅ 데이터셋 로드 완료: {len(base_dataset)}개 세그먼트")
        logger.info(f"   어휘 크기: {base_dataset.vocab_size}")
        
        # 데이터 분할기 생성
        splitter = StratifiedDataSplitter(
            config=self.config.data_split,
            random_seed=self.config.random_seed.seed
        )
        
        return base_dataset, splitter
    
    def create_model(self, vocab_size: int, stage_config: TrainingStageConfig) -> SequenceToSequenceSignModel:
        """단계별 모델 생성"""
        model = SequenceToSequenceSignModel(
            vocab_size=vocab_size,
            embed_dim=384,
            num_encoder_layers=6,
            num_decoder_layers=4,
            num_heads=8,
            dim_feedforward=1024,
            max_seq_len=200,
            dropout=stage_config.dropout_rate,
            label_smoothing=stage_config.label_smoothing  # 라벨 스무딩 추가
        )
        
        # 먼저 모델을 메인 디바이스로 이동
        model = model.to(self.device)
        logger.info(f"📱 모델을 {self.device}로 이동 완료")
        
        # 멀티 GPU 설정은 나중에 별도로 수행
        return model
    
    def setup_stage_training(self, 
                           model: SequenceToSequenceSignModel,
                           stage_config: TrainingStageConfig,
                           train_dataloader,
                           val_dataloader) -> Tuple[SignLanguageTrainer, EarlyStopping]:
        """단계별 학습 설정"""
        
        # 트레이너 생성
        trainer = SignLanguageTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=str(self.device),  # 개선된 디바이스 사용
            checkpoint_dir=str(self.checkpoint_dir / stage_config.name),
            log_dir=str(self.log_dir / stage_config.name)
        )
        
        # 학습 파라미터 설정
        trainer.setup_training(
            learning_rate=stage_config.learning_rate,
            weight_decay=stage_config.weight_decay,
            gradient_clip_val=1.0,
            word_loss_weight=1.0,
            boundary_loss_weight=0.5,
            confidence_loss_weight=0.3
        )
        
        # 얼리스탑 설정
        early_stopping = EarlyStopping(self.config.early_stopping)
        
        return trainer, early_stopping
    
    def train_single_stage(self,
                          stage_idx: int,
                          stage_config: TrainingStageConfig,
                          model: SequenceToSequenceSignModel,
                          base_dataset: UnifiedSignLanguageDataset,
                          splitter: StratifiedDataSplitter) -> Dict:
        """단일 단계 학습"""
        
        logger.info("="*80)
        logger.info(f"🚀 Stage {stage_idx+1}: {stage_config.name}")
        logger.info(f"📝 설명: {stage_config.description}")
        logger.info(f"⚙️ 설정: LR={stage_config.learning_rate}, BS={stage_config.batch_size}, "
                   f"Aug={'✅' if stage_config.enable_augmentation else '❌'}")
        logger.info("="*80)
        
        # 증강 설정
        augmentation_config = None
        if stage_config.enable_augmentation:
            augmentation_config = {
                'enable_horizontal_flip': True,
                'horizontal_flip_prob': 0.5 * stage_config.augmentation_strength,
                'enable_rotation': True,
                'rotation_range': 15.0 * stage_config.augmentation_strength,
                'enable_scaling': True,
                'scaling_range': (1.0 - 0.1 * stage_config.augmentation_strength, 
                                 1.0 + 0.1 * stage_config.augmentation_strength),
                'enable_noise': True,
                'noise_std': 0.005 * stage_config.augmentation_strength
            }
        
        # 데이터로더 생성 (멀티 GPU에 맞는 배치 크기 조정)
        effective_batch_size = stage_config.batch_size
        if self.config.multi_gpu and self.config.auto_adjust_batch_size:
            effective_batch_size = DeviceManager.get_effective_batch_size(
                stage_config.batch_size, self.device
            )
        
        train_dataloader, val_dataloader, test_dataloader = splitter.create_dataloaders(
            dataset=base_dataset,
            batch_size=effective_batch_size,
            enable_train_augmentation=stage_config.enable_augmentation,
            augmentation_config=augmentation_config
        )
        
        # 학습 설정
        trainer, early_stopping = self.setup_stage_training(
            model, stage_config, train_dataloader, val_dataloader
        )
        
        # 학습 실행
        stage_start_time = time.time()
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_epoch = -1
        
        stage_results = {
            'stage_name': stage_config.name,
            'description': stage_config.description,
            'config': stage_config.__dict__,
            'epochs_trained': 0,
            'best_epoch': -1,
            'best_val_loss': float('inf'),
            'best_val_accuracy': 0.0,
            'training_time': 0.0,
            'early_stopped': False,
            'improvement_from_previous': 0.0
        }
        
        try:
            for epoch in range(stage_config.num_epochs):
                epoch_start_time = time.time()
                
                # 훈련 (에포크 단위 저장으로 개선)
                train_loss, train_metrics = trainer.train_epoch(
                    epoch=epoch,
                    log_every_n_steps=100
                )
                
                # 검증
                val_loss, val_metrics = trainer.validate_epoch(epoch)
                
                # 스케줄러 업데이트
                trainer.scheduler.step(val_loss)
                
                # 성능 기록
                val_accuracy = val_metrics.get('word_accuracy', 0.0)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch
                
                # 로깅
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch:3d}/{stage_config.num_epochs-1} ({epoch_time:.1f}s)")
                logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                logger.info(f"  Train Acc: {train_metrics.get('word_accuracy', 0):.3f} | "
                           f"Val Acc: {val_accuracy:.3f}")
                
                # 얼리스탑 체크
                if early_stopping(val_loss, epoch, model.state_dict()):
                    logger.info(f"⏹️ 얼리스탑 발동 - Stage {stage_idx+1} 종료")
                    stage_results['early_stopped'] = True
                    
                    # 최고 성능 가중치 복원
                    if early_stopping.best_weights:
                        model.load_state_dict(early_stopping.best_weights)
                        logger.info("✅ 최고 성능 가중치 복원 완료")
                    break
                
                stage_results['epochs_trained'] = epoch + 1
            
            # 단계 결과 업데이트
            stage_results['best_epoch'] = best_epoch
            stage_results['best_val_loss'] = best_val_loss
            stage_results['best_val_accuracy'] = best_val_accuracy
            stage_results['training_time'] = time.time() - stage_start_time
            
            # 전체 최고 성능 대비 개선도 계산
            if self.best_overall_performance['val_loss'] < float('inf'):
                improvement = (self.best_overall_performance['val_loss'] - best_val_loss) / self.best_overall_performance['val_loss']
                stage_results['improvement_from_previous'] = improvement
            
            # 전체 최고 성능 업데이트
            if best_val_loss < self.best_overall_performance['val_loss']:
                self.best_overall_performance.update({
                    'stage': stage_config.name,
                    'epoch': best_epoch,
                    'val_loss': best_val_loss,
                    'val_accuracy': best_val_accuracy
                })
                
                # 최고 성능 모델 저장
                best_model_path = self.checkpoint_dir / f"best_model_stage_{stage_idx+1}.pt"
                torch.save({
                    'stage': stage_config.name,
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': best_val_loss,
                    'val_accuracy': best_val_accuracy,
                    'config': stage_config.__dict__
                }, best_model_path)
                logger.info(f"💾 최고 성능 모델 저장: {best_model_path}")
            
            # 각 단계별 모델 저장 (성능과 상관없이)
            stage_model_path = self.checkpoint_dir / f"stage_{stage_idx+1}_{stage_config.name}.pt"
            torch.save({
                'stage_idx': stage_idx + 1,
                'stage_name': stage_config.name,
                'stage_description': stage_config.description,
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_accuracy,
                'stage_config': stage_config.__dict__,
                'training_time': stage_results['training_time']
            }, stage_model_path)
            logger.info(f"🗃️ Stage {stage_idx+1} 모델 저장: {stage_model_path}")
            
            # 테스트 평가 (선택적)
            if self.config.evaluate_on_test and (stage_idx + 1) % self.config.test_every_n_stages == 0:
                logger.info("📊 테스트 세트 평가 중...")
                test_loss, test_metrics = self._evaluate_on_test(model, test_dataloader, trainer)
                stage_results['test_performance'] = {
                    'test_loss': test_loss,
                    'test_accuracy': test_metrics.get('word_accuracy', 0.0)
                }
                logger.info(f"  테스트 Loss: {test_loss:.4f}, 테스트 Acc: {test_metrics.get('word_accuracy', 0.0):.3f}")
            
            logger.info(f"✅ Stage {stage_idx+1} 완료!")
            logger.info(f"  최고 성능: Val Loss {best_val_loss:.4f}, Val Acc {best_val_accuracy:.3f} (Epoch {best_epoch})")
            logger.info(f"  학습 시간: {stage_results['training_time']:.1f}초")
            
        except Exception as e:
            logger.error(f"❌ Stage {stage_idx+1} 학습 중 오류: {e}")
            stage_results['error'] = str(e)
            raise
        
        return stage_results
    
    @torch.no_grad()
    def _evaluate_on_test(self, model, test_dataloader, trainer) -> Tuple[float, Dict]:
        """테스트 세트 평가"""
        model.eval()
        total_loss = 0
        total_metrics = {'word_accuracy': 0, 'boundary_accuracy': 0}
        
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="테스트 평가")):
            # GPU로 이동
            pose_features = batch['pose_features'].to(trainer.device)
            vocab_ids = batch['vocab_ids'].to(trainer.device)
            frame_masks = batch['frame_masks'].to(trainer.device)
            vocab_masks = batch['vocab_masks'].to(trainer.device)
            
            # 디버깅을 위한 배치 정보 로깅 (첫 번째 배치만)
            if batch_idx == 0:
                logger.debug(f"🔍 Test batch debug info:")
                logger.debug(f"  pose_features shape: {pose_features.shape}")
                logger.debug(f"  vocab_ids shape: {vocab_ids.shape}")
                logger.debug(f"  frame_masks shape: {frame_masks.shape}")
                logger.debug(f"  vocab_masks shape: {vocab_masks.shape}")
            
            try:
                # 추론 - 테스트에서도 Teacher Forcing 모드 사용 (차원 일치를 위해)
                outputs = model(
                    pose_features=pose_features,
                    vocab_ids=vocab_ids,
                    frame_masks=frame_masks,
                    vocab_masks=vocab_masks,
                    force_training_mode=True  # 차원 일치를 위해 강제로 teacher forcing 모드
                )
                
                # 출력 차원 로깅 (첫 번째 배치만)
                if batch_idx == 0:
                    logger.debug(f"🔍 Model outputs debug info:")
                    logger.debug(f"  word_logits shape: {outputs['word_logits'].shape}")
                    logger.debug(f"  boundary_logits shape: {outputs['boundary_logits'].shape}")
            
            except Exception as e:
                logger.error(f"❌ Model forward pass error at batch {batch_idx}: {e}")
                logger.error(f"  pose_features shape: {pose_features.shape}")
                logger.error(f"  vocab_ids shape: {vocab_ids.shape}")
                raise
            
            try:
                # 손실 계산
                boundary_labels = trainer.create_boundary_labels(vocab_ids, vocab_masks)
                targets = {'vocab_ids': vocab_ids, 'boundary_labels': boundary_labels}
                losses = model.compute_loss(outputs, targets, vocab_masks)
                
            except Exception as e:
                logger.error(f"❌ Loss computation error at batch {batch_idx}: {e}")
                logger.error(f"  vocab_ids shape: {vocab_ids.shape}")
                logger.error(f"  outputs word_logits shape: {outputs['word_logits'].shape}")
                logger.error(f"  boundary_labels shape: {boundary_labels.shape if boundary_labels is not None else 'None'}")
                raise
            
            total_loss += losses['total_loss'].item()
            
            # 메트릭 계산
            metrics = trainer.compute_metrics(outputs, targets, {'vocab_masks': vocab_masks})
            for key, value in metrics.items():
                if key in total_metrics:
                    total_metrics[key] += value
        
        # 평균 계산
        avg_loss = total_loss / len(test_dataloader)
        avg_metrics = {key: value / len(test_dataloader) for key, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def should_continue_training(self, stage_results: List[Dict]) -> bool:
        """다음 단계 진행 여부 결정"""
        if len(stage_results) < 2:
            return True
        
        latest_result = stage_results[-1]
        
        # 에러가 발생했으면 중단
        if 'error' in latest_result:
            return False
        
        # 개선 임계값이 음수면 항상 계속 (테스트 모드 - 모든 단계 진행)
        if self.config.multi_stage.improvement_threshold < 0:
            logger.info("🔄 테스트 모드: 모든 단계 진행")
            return True
        
        # 개선도가 임계값보다 낮으면 중단
        improvement = latest_result.get('improvement_from_previous', 0)
        if improvement < self.config.multi_stage.improvement_threshold:
            logger.info(f"⏹️ 개선도가 임계값보다 낮음 ({improvement:.4f} < {self.config.multi_stage.improvement_threshold:.4f})")
            return False
        
        return True
    
    def train_multi_stage(self) -> Dict:
        """다단계 학습 실행"""
        logger.info("🚀 고급 다단계 학습 시작!")
        logger.info(f"실험명: {self.config.experiment_name}")
        
        # 데이터 준비
        base_dataset, splitter = self.load_and_prepare_data()
        
        # 모델 생성 (첫 번째 단계 설정으로)
        first_stage = self.config.multi_stage.stages[0]
        model = self.create_model(base_dataset.vocab_size, first_stage)
        
        # 멀티 GPU 설정 (모델이 이미 메인 디바이스에 있음)
        if self.config.multi_gpu and DeviceManager.is_multi_gpu_available():
            if self.config.use_data_parallel and torch.cuda.device_count() > 1:
                # 모든 파라미터가 같은 디바이스에 있는지 확인
                param_devices = {param.device for param in model.parameters()}
                if len(param_devices) > 1:
                    logger.warning(f"⚠️ 모델 파라미터가 여러 디바이스에 분산되어 있음: {param_devices}")
                    model = model.to(self.device)  # 강제로 메인 디바이스로 이동
                
                model = DeviceManager.setup_multi_gpu(model, self.device, self.config.use_data_parallel)
                logger.info(f"🚀 DataParallel 활성화: {torch.cuda.device_count()}개 GPU 사용")
            else:
                logger.warning("⚠️ 멀티 GPU 요청되었지만 DataParallel 비활성화됨")
        else:
            # 단일 GPU/CPU 사용
            model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 모델 정보: {total_params:,} 파라미터")
        
        # 각 단계별 학습
        all_stage_results = []
        
        for stage_idx, stage_config in enumerate(self.config.multi_stage.stages):
            # 최대 단계 수 확인
            if stage_idx >= self.config.multi_stage.max_stages:
                logger.info(f"⏹️ 최대 단계 수 ({self.config.multi_stage.max_stages}) 도달")
                break
            
            # 이전 단계에서 개선이 없으면 중단
            if stage_idx > 0 and not self.should_continue_training(all_stage_results):
                logger.info("⏹️ 개선이 없어 다단계 학습 중단")
                break
            
            # 모델 동적 수정 (필요한 경우)
            if stage_config.dropout_rate != 0.1:  # 기본값과 다르면
                self._update_model_dropout(model, stage_config.dropout_rate)
            
            # 단계별 학습 실행
            stage_results = self.train_single_stage(
                stage_idx, stage_config, model, base_dataset, splitter
            )
            
            all_stage_results.append(stage_results)
            self.experiment_results['stages'] = all_stage_results
            
            # 중간 결과 저장
            if self.config.save_intermediate_results:
                self._save_experiment_results()
        
        # 최종 테스트 평가
        if self.config.evaluate_on_test:
            logger.info("🎯 최종 테스트 평가 수행...")
            _, _, test_dataloader = splitter.create_dataloaders(
                base_dataset, batch_size=32, enable_train_augmentation=False
            )
            
            # 더미 트레이너 생성 (평가용)
            dummy_trainer = SignLanguageTrainer(model, None, None, device=str(self.device))
            
            final_test_loss, final_test_metrics = self._evaluate_on_test(
                model, test_dataloader, dummy_trainer
            )
            
            self.experiment_results['final_performance'] = {
                'test_loss': final_test_loss,
                'test_accuracy': final_test_metrics.get('word_accuracy', 0.0),
                'test_metrics': final_test_metrics
            }
            
            logger.info(f"🏆 최종 테스트 성능:")
            logger.info(f"  테스트 Loss: {final_test_loss:.4f}")
            logger.info(f"  테스트 Accuracy: {final_test_metrics.get('word_accuracy', 0.0):.3f}")
        
        # 최종 결과 저장
        self._save_experiment_results()
        
        # 결과 요약
        self._print_final_summary(all_stage_results)
        
        return self.experiment_results
    
    def _update_model_dropout(self, model: nn.Module, new_dropout_rate: float):
        """모델의 드롭아웃 비율 동적 변경"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_dropout_rate
        logger.info(f"🔧 모델 드롭아웃 비율 변경: {new_dropout_rate}")
    
    def _save_experiment_results(self):
        """실험 결과 저장"""
        results_path = self.checkpoint_dir / f"{self.config.experiment_name}_results.json"
        
        # NumPy 배열과 Tensor를 JSON 직렬화 가능하도록 변환
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.float32):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                # 설정 객체들을 딕셔너리로 변환 (private 속성 제외)
                try:
                    result = {}
                    for k, v in obj.__dict__.items():
                        if not k.startswith('_'):  # private 속성 제외
                            result[k] = convert_for_json(v)
                    return result
                except Exception as e:
                    return str(obj)
            elif hasattr(obj, '_asdict'):  # namedtuple인 경우
                return convert_for_json(obj._asdict())
            elif callable(obj):
                return str(obj)
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                try:
                    json.dumps(obj)  # JSON 직렬화 가능한지 테스트
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        serializable_results = convert_for_json(self.experiment_results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 실험 결과 저장: {results_path}")
    
    def _print_final_summary(self, stage_results: List[Dict]):
        """최종 결과 요약 출력"""
        logger.info("\n" + "="*80)
        logger.info("🎉 다단계 학습 완료! 최종 결과 요약")
        logger.info("="*80)
        
        total_training_time = sum(stage['training_time'] for stage in stage_results)
        logger.info(f"총 학습 시간: {total_training_time:.1f}초 ({total_training_time/60:.1f}분)")
        logger.info(f"수행된 단계 수: {len(stage_results)}")
        
        logger.info("\n📊 단계별 성능:")
        for i, stage in enumerate(stage_results):
            logger.info(f"  Stage {i+1} ({stage['stage_name']}):")
            logger.info(f"    Val Loss: {stage['best_val_loss']:.4f}")
            logger.info(f"    Val Acc: {stage['best_val_accuracy']:.3f}")
            logger.info(f"    에포크: {stage['epochs_trained']}")
            if 'improvement_from_previous' in stage:
                logger.info(f"    개선도: {stage['improvement_from_previous']:.4f}")
        
        logger.info(f"\n🏆 최고 성능: {self.best_overall_performance['stage']} 단계")
        logger.info(f"  Val Loss: {self.best_overall_performance['val_loss']:.4f}")
        logger.info(f"  Val Acc: {self.best_overall_performance['val_accuracy']:.3f}")
        
        if 'final_performance' in self.experiment_results:
            final_perf = self.experiment_results['final_performance']
            logger.info(f"\n🎯 최종 테스트 성능:")
            logger.info(f"  Test Loss: {final_perf['test_loss']:.4f}")
            logger.info(f"  Test Acc: {final_perf['test_accuracy']:.3f}")
        
        logger.info("="*80)

if __name__ == "__main__":
    # 간단한 테스트
    from advanced_config import AdvancedTrainingConfig
    
    logging.basicConfig(level=logging.INFO)
    
    config = AdvancedTrainingConfig()
    config.multi_stage.stages = config.multi_stage.stages[:2]  # 처음 2단계만 테스트
    
    trainer = AdvancedSignLanguageTrainer(config)
    results = trainer.train_multi_stage()
    
    print("✅ 다단계 학습 테스트 완료!")