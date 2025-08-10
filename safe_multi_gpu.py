#!/usr/bin/env python3
"""
안전한 멀티 GPU 구현
DataParallel 대신 수동 배치 분할을 사용합니다.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SafeMultiGPUWrapper(nn.Module):
    """안전한 멀티 GPU 래퍼 - DataParallel 대신 수동 배치 분할"""
    
    def __init__(self, model: nn.Module, device_ids: List[int]):
        super().__init__()
        self.model = model
        self.device_ids = device_ids
        self.main_device = device_ids[0]
        
        # 각 GPU에 모델 복사본 생성
        self.replicas = []
        for device_id in device_ids:
            device = torch.device(f'cuda:{device_id}')
            replica = self._create_replica(model, device)
            self.replicas.append(replica)
        
        logger.info(f"🚀 SafeMultiGPUWrapper 초기화 완료: {len(device_ids)}개 GPU")
    
    def _create_replica(self, model: nn.Module, device: torch.device) -> nn.Module:
        """모델 복사본 생성"""
        # 모델을 해당 디바이스로 이동
        replica = model.to(device)
        
        # 모든 파라미터가 올바른 디바이스에 있는지 확인
        for param in replica.parameters():
            if param.device != device:
                param.data = param.data.to(device)
        
        return replica
    
    def forward(self, *inputs, **kwargs):
        """순전파 - 배치를 GPU별로 분할"""
        if len(inputs) == 0:
            return None
        
        # 첫 번째 입력을 기준으로 배치 크기 확인
        batch_size = inputs[0].size(0)
        
        if batch_size < len(self.device_ids):
            # 배치 크기가 GPU 수보다 작으면 단일 GPU 사용
            main_device = torch.device(f'cuda:{self.main_device}')
            inputs_on_main = tuple(inp.to(main_device) if torch.is_tensor(inp) else inp for inp in inputs)
            kwargs_on_main = {k: v.to(main_device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
            return self.replicas[0](*inputs_on_main, **kwargs_on_main)
        
        # 배치를 GPU별로 분할
        chunk_size = batch_size // len(self.device_ids)
        remainder = batch_size % len(self.device_ids)
        
        input_chunks = []
        kwargs_chunks = []
        
        start_idx = 0
        for i, device_id in enumerate(self.device_ids):
            # 나머지가 있으면 첫 번째 GPU들에 1개씩 더 할당
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            device = torch.device(f'cuda:{device_id}')
            
            # 입력을 청크로 분할하고 해당 디바이스로 이동
            input_chunk = tuple(
                inp[start_idx:end_idx].to(device) if torch.is_tensor(inp) else inp 
                for inp in inputs
            )
            
            kwargs_chunk = {}
            for k, v in kwargs.items():
                if torch.is_tensor(v) and v.size(0) == batch_size:
                    kwargs_chunk[k] = v[start_idx:end_idx].to(device)
                else:
                    kwargs_chunk[k] = v
            
            input_chunks.append(input_chunk)
            kwargs_chunks.append(kwargs_chunk)
            start_idx = end_idx
        
        # 각 GPU에서 순전파 수행
        outputs = []
        for i, (replica, input_chunk, kwargs_chunk) in enumerate(zip(self.replicas, input_chunks, kwargs_chunks)):
            try:
                output = replica(*input_chunk, **kwargs_chunk)
                outputs.append(output)
            except Exception as e:
                logger.error(f"❌ GPU {self.device_ids[i]}에서 순전파 실패: {e}")
                raise
        
        # 출력 결합
        return self._combine_outputs(outputs)
    
    def _combine_outputs(self, outputs: List[Any]) -> Any:
        """GPU별 출력을 결합"""
        if not outputs:
            return None
        
        # 첫 번째 출력의 타입에 따라 처리
        first_output = outputs[0]
        
        if isinstance(first_output, torch.Tensor):
            # 텐서인 경우 concatenate
            main_device = torch.device(f'cuda:{self.main_device}')
            outputs_on_main = [out.to(main_device) for out in outputs]
            return torch.cat(outputs_on_main, dim=0)
        
        elif isinstance(first_output, dict):
            # 딕셔너리인 경우 키별로 concatenate
            combined = {}
            main_device = torch.device(f'cuda:{self.main_device}')
            
            for key in first_output.keys():
                if torch.is_tensor(first_output[key]):
                    tensors = [out[key].to(main_device) for out in outputs]
                    combined[key] = torch.cat(tensors, dim=0)
                else:
                    # 텐서가 아닌 경우 첫 번째 값 사용
                    combined[key] = first_output[key]
            
            return combined
        
        elif isinstance(first_output, (list, tuple)):
            # 리스트/튜플인 경우 element별로 처리
            combined = []
            main_device = torch.device(f'cuda:{self.main_device}')
            
            for i in range(len(first_output)):
                if torch.is_tensor(first_output[i]):
                    tensors = [out[i].to(main_device) for out in outputs]
                    combined.append(torch.cat(tensors, dim=0))
                else:
                    combined.append(first_output[i])
            
            return type(first_output)(combined)
        
        else:
            # 기타 타입은 첫 번째 값 반환
            return first_output

def create_safe_multi_gpu_model(model: nn.Module, device_ids: List[int] = None) -> nn.Module:
    """안전한 멀티 GPU 모델 생성"""
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    if len(device_ids) <= 1:
        # 단일 GPU
        device = torch.device(f'cuda:{device_ids[0]}' if device_ids else 'cuda:0')
        return model.to(device)
    
    # 멀티 GPU
    return SafeMultiGPUWrapper(model, device_ids)
