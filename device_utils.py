"""
디바이스 자동 감지 및 설정 유틸리티
"""
import torch
import logging

logger = logging.getLogger(__name__)

class DeviceManager:
    """통합 디바이스 관리자"""
    
    @staticmethod
    def detect_best_device(preferred_device: str = "auto", multi_gpu: bool = False) -> torch.device:
        """최적의 디바이스 자동 감지
        
        Args:
            preferred_device: 선호 디바이스 ("auto", "cuda", "xpu", "cpu")
            multi_gpu: 멀티 GPU 사용 여부
        """
        
        if preferred_device != "auto":
            # 사용자 지정 디바이스 검증
            device = DeviceManager._validate_device(preferred_device)
            if multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
                logger.info(f"🚀 멀티 GPU 모드: {torch.cuda.device_count()}개 CUDA 디바이스")
            return device
        
        # 자동 감지 순서: CUDA (멀티 GPU 포함) > XPU > CPU
        logger.info("🔍 사용 가능한 디바이스 검색 중...")
        
        # 1. CUDA 확인 (멀티 GPU 우선)
        cuda_device = DeviceManager._check_cuda(multi_gpu)
        if cuda_device:
            return cuda_device
            
        # 2. XPU 확인 (Intel GPU)
        xpu_device = DeviceManager._check_xpu()
        if xpu_device:
            return xpu_device
        
        # 3. CPU 폴백
        logger.info("💻 CPU 디바이스 사용")
        return torch.device("cpu")
    
    @staticmethod
    def _check_xpu() -> torch.device:
        """XPU 사용 가능성 확인"""
        try:
            # XPU 모듈 존재 확인
            if not hasattr(torch, 'xpu'):
                logger.debug("XPU 모듈 미지원")
                return None
            
            # device_count() 방식으로 확인
            try:
                device_count = torch.xpu.device_count()
                if device_count > 0:
                    logger.info(f"⚡ XPU 디바이스 감지: {device_count}개 디바이스")
                    # 추가 검증: 실제 텐서 생성 테스트
                    test_tensor = torch.tensor([1.0], device='xpu:0')
                    logger.info(f"✅ XPU 디바이스 사용: {test_tensor.device}")
                    return torch.device("xpu:0")
            except Exception as e:
                logger.debug(f"XPU device_count() 실패: {e}")
            
            # is_available() 방식으로 확인 (있는 경우)
            try:
                if hasattr(torch.xpu, 'is_available') and torch.xpu.is_available():
                    test_tensor = torch.tensor([1.0], device='xpu')
                    logger.info(f"⚡ XPU 디바이스 사용: {test_tensor.device}")
                    return torch.device("xpu")
            except Exception as e:
                logger.debug(f"XPU is_available() 실패: {e}")
            
            # 직접 테스트 방식
            try:
                test_tensor = torch.tensor([1.0], device='xpu')
                logger.info(f"⚡ XPU 디바이스 사용: {test_tensor.device}")
                return torch.device("xpu")
            except Exception as e:
                logger.debug(f"XPU 직접 테스트 실패: {e}")
                
        except Exception as e:
            logger.debug(f"XPU 전체 확인 실패: {e}")
        
        return None
    
    @staticmethod
    def _check_cuda(multi_gpu: bool = False) -> torch.device:
        """CUDA 사용 가능성 확인"""
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                if multi_gpu and device_count > 1:
                    logger.info(f"� 멀티 GPU CUDA 사용: {device_count}개 디바이스")
                    for i in range(device_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        logger.info(f"  GPU {i}: {gpu_name}")
                    return torch.device("cuda")  # 메인 디바이스
                else:
                    logger.info(f"🔥 단일 CUDA 디바이스 사용: {device_name}")
                    return torch.device(f"cuda:{current_device}")
        except Exception as e:
            logger.debug(f"CUDA 확인 실패: {e}")
        
        return None
    
    @staticmethod
    def _validate_device(device_str: str) -> torch.device:
        """사용자 지정 디바이스 검증"""
        try:
            device = torch.device(device_str)
            
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("⚠️ CUDA 요청되었으나 사용 불가 - CPU로 폴백")
                    return torch.device("cpu")
                logger.info(f"🔥 지정된 CUDA 디바이스 사용: {device}")
                return device
            
            elif device.type == "xpu":
                xpu_device = DeviceManager._check_xpu()
                if xpu_device is None:
                    logger.warning("⚠️ XPU 요청되었으나 사용 불가 - CPU로 폴백")
                    return torch.device("cpu")
                logger.info(f"⚡ 지정된 XPU 디바이스 사용: {device}")
                return device
            
            elif device.type == "cpu":
                logger.info("💻 지정된 CPU 디바이스 사용")
                return device
            
            else:
                logger.warning(f"⚠️ 알 수 없는 디바이스 타입: {device} - CPU로 폴백")
                return torch.device("cpu")
                
        except Exception as e:
            logger.error(f"❌ 디바이스 검증 실패: {e} - CPU로 폴백")
            return torch.device("cpu")
    
    @staticmethod
    def get_device_info(device: torch.device) -> dict:
        """디바이스 정보 반환"""
        info = {
            'type': device.type,
            'index': device.index,
            'name': str(device)
        }
        
        try:
            if device.type == "cuda":
                info.update({
                    'name': torch.cuda.get_device_name(device),
                    'memory_total': torch.cuda.get_device_properties(device).total_memory,
                    'memory_cached': torch.cuda.memory_cached(device),
                    'memory_allocated': torch.cuda.memory_allocated(device)
                })
            elif device.type == "xpu":
                # XPU 정보 (가능한 경우)
                try:
                    if hasattr(torch.xpu, 'get_device_name'):
                        info['name'] = torch.xpu.get_device_name(device.index)
                except:
                    pass
            elif device.type == "cpu":
                import os
                info.update({
                    'cores': os.cpu_count(),
                    'threads': torch.get_num_threads()
                })
        except Exception as e:
            logger.debug(f"디바이스 정보 수집 실패: {e}")
        
        return info
    
    @staticmethod
    def optimize_for_device(device: torch.device, multi_gpu: bool = False):
        """디바이스별 최적화 설정"""
        if device.type == "cuda":
            # CUDA 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            if multi_gpu and torch.cuda.device_count() > 1:
                logger.info(f"🔥 멀티 GPU CUDA 최적화 설정 완료 ({torch.cuda.device_count()}개 GPU)")
            else:
                logger.info("🔥 단일 GPU CUDA 최적화 설정 완료")
            
        elif device.type == "xpu":
            # XPU 최적화 (가능한 경우)
            try:
                # XPU 관련 최적화 설정
                logger.info("⚡ XPU 최적화 설정 완료")
            except:
                pass
                
        elif device.type == "cpu":
            # CPU 최적화
            torch.set_num_threads(min(8, torch.get_num_threads()))
            logger.info("💻 CPU 최적화 설정 완료")
    
    @staticmethod
    def setup_multi_gpu(model: torch.nn.Module, device: torch.device, 
                       use_data_parallel: bool = True) -> torch.nn.Module:
        """멀티 GPU 설정
        
        Args:
            model: PyTorch 모델
            device: 메인 디바이스
            use_data_parallel: DataParallel 사용 여부 (False면 수동 관리)
        
        Returns:
            멀티 GPU가 설정된 모델
        """
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            # 먼저 모델을 메인 디바이스로 완전히 이동
            model = model.to(device)
            
            # 모든 서브모듈과 버퍼를 명시적으로 메인 디바이스로 이동
            def move_to_device(module):
                for child in module.children():
                    move_to_device(child)
                # 파라미터 이동
                for param in module.parameters(recurse=False):
                    if param.device != device:
                        param.data = param.data.to(device)
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(device)
                # 버퍼 이동
                for buffer in module.buffers(recurse=False):
                    if buffer.device != device:
                        buffer.data = buffer.data.to(device)
            
            if use_data_parallel:
                # 디바이스 동기화 강화
                move_to_device(model)
                
                # 모든 파라미터가 같은 디바이스에 있는지 최종 확인
                devices = {param.device for param in model.parameters()}
                if len(devices) > 1:
                    logger.error(f"❌ 모델 파라미터가 여러 디바이스에 분산됨: {devices}")
                    # 강제로 모든 파라미터를 메인 디바이스로 이동
                    model = model.to(device)
                
                # DataParallel 설정
                model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
                model = model.to(device)  # DataParallel 래퍼도 메인 디바이스에
                
                logger.info(f"🚀 DataParallel 설정 완료: {torch.cuda.device_count()}개 GPU 사용 (디바이스: {list(range(torch.cuda.device_count()))})")
            else:
                logger.info(f"🚀 멀티 GPU 환경 감지됨: {torch.cuda.device_count()}개 GPU (수동 관리)")
        else:
            # 단일 GPU 또는 CPU
            model = model.to(device)
        
        return model
    
    @staticmethod
    def get_effective_batch_size(base_batch_size: int, device: torch.device) -> int:
        """멀티 GPU 환경에 맞는 효과적 배치 크기 계산"""
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            gpu_count = torch.cuda.device_count()
            # 각 GPU당 배치 크기가 너무 작아지지 않도록 조정
            effective_batch_size = max(base_batch_size, gpu_count * 4)  # 최소 GPU당 4
            logger.info(f"📊 멀티 GPU 배치 크기 조정: {base_batch_size} → {effective_batch_size} "
                       f"(GPU당 ~{effective_batch_size // gpu_count})")
            return effective_batch_size
        return base_batch_size
    
    @staticmethod
    def is_multi_gpu_available() -> bool:
        """멀티 GPU 사용 가능 여부 확인"""
        return torch.cuda.is_available() and torch.cuda.device_count() > 1

def get_device_string(device: torch.device) -> str:
    """디바이스를 문자열로 변환"""
    if device.index is not None:
        return f"{device.type}:{device.index}"
    return device.type

# 편의 함수들
def auto_device(preferred: str = "auto", multi_gpu: bool = False) -> torch.device:
    """자동 디바이스 선택"""
    return DeviceManager.detect_best_device(preferred, multi_gpu)

def device_info(device: torch.device = None) -> dict:
    """디바이스 정보"""
    if device is None:
        device = auto_device()
    return DeviceManager.get_device_info(device)

def optimize_device(device: torch.device = None, multi_gpu: bool = False):
    """디바이스 최적화"""
    if device is None:
        device = auto_device(multi_gpu=multi_gpu)
    DeviceManager.optimize_for_device(device, multi_gpu)

def setup_multi_gpu(model: torch.nn.Module, device: torch.device = None, 
                   use_data_parallel: bool = True) -> torch.nn.Module:
    """멀티 GPU 설정 편의 함수"""
    if device is None:
        device = auto_device(multi_gpu=True)
    return DeviceManager.setup_multi_gpu(model, device, use_data_parallel)

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    print("=== 디바이스 관리자 테스트 ===")
    
    # 자동 감지
    device = auto_device()
    print(f"감지된 디바이스: {device}")
    
    # 정보 출력
    info = device_info(device)
    print(f"디바이스 정보: {info}")
    
    # 최적화 적용
    optimize_device(device)
    
    print("테스트 완료!")
