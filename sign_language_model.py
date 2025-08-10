#!/usr/bin/env python3
"""
실시간 수화 인식 모델 - Sequence-to-Sequence Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SpatialEncoder(nn.Module):
    """공간 특징 추출기 (133개 키포인트 → 임베딩 차원)"""
    
    def __init__(self, input_dim: int = 133 * 3, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # 공간 특징 압축
        self.spatial_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # 키포인트 중요도 어텐션 (선택적)
        self.attention = nn.MultiheadAttention(embed_dim=3, num_heads=1, batch_first=True)
        self.use_attention = False  # 성능 테스트 후 결정
        
    def forward(self, x):
        # x: [batch, seq_len, 133, 3]
        batch_size, seq_len, n_keypoints, coords = x.shape
        
        if self.use_attention:
            # 키포인트 어텐션 적용 (실험적)
            x_reshaped = x.view(batch_size * seq_len, n_keypoints, coords)
            attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            x = attn_out.view(batch_size, seq_len, n_keypoints, coords)
        
        # 공간 차원 평탄화 및 임베딩
        x = x.view(batch_size, seq_len, -1)  # [batch, seq_len, 133*3]
        x = self.spatial_layers(x)  # [batch, seq_len, embed_dim]
        
        return x

class SequenceToSequenceSignModel(nn.Module):
    """Sequence-to-Sequence 수화 인식 모델"""
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 256,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 4,
                 num_heads: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_len: int = 500,
                 label_smoothing: float = 0.0):  # Label smoothing 추가
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.label_smoothing = label_smoothing  # Label smoothing 저장
        
        # 1. 공간 인코더 (133개 키포인트 → 임베딩)
        self.spatial_encoder = SpatialEncoder(
            input_dim=133 * 3,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # 2. 위치 인코딩
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # 3. Transformer Encoder (시간 모델링)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 4. Vocabulary 임베딩 (디코더 입력용)
        self.vocab_embedding = nn.Embedding(vocab_size, embed_dim)
        self.vocab_pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # 5. Transformer Decoder (시퀀스 생성)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # 6. 출력 헤드들
        self.word_classifier = nn.Linear(embed_dim, vocab_size)
        self.boundary_detector = nn.Linear(embed_dim, 3)  # START/CONTINUE/END
        self.confidence_head = nn.Linear(embed_dim, 1)
        
        # 7. 손실 함수 가중치
        self.register_buffer('word_loss_weight', torch.tensor(1.0))
        self.register_buffer('boundary_loss_weight', torch.tensor(0.5))
        self.register_buffer('confidence_loss_weight', torch.tensor(0.3))
        
        # 8. Label Smoothing 손실 함수 초기화
        if self.label_smoothing > 0:
            self.word_loss_fn = LabelSmoothingCrossEntropy(
                num_classes=vocab_size, 
                smoothing=self.label_smoothing,
                ignore_index=0
            )
        else:
            self.word_loss_fn = None
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, 0, 0.1)
    
    def create_padding_mask(self, lengths, max_len):
        """패딩 마스크 생성"""
        batch_size = len(lengths)
        mask = torch.arange(max_len).expand(batch_size, max_len).to(lengths.device)
        mask = mask >= lengths.unsqueeze(1)
        return mask
    
    def create_causal_mask(self, seq_len):
        """인과적 마스크 생성 (디코더용)"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(self, pose_features, vocab_ids=None, frame_masks=None, vocab_masks=None, force_training_mode=False):
        """
        Args:
            pose_features: [batch, seq_len, 133, 3]
            vocab_ids: [batch, vocab_len] (훈련 시에만)
            frame_masks: [batch, seq_len]
            vocab_masks: [batch, vocab_len]
            force_training_mode: 테스트에서도 teacher forcing 모드 사용
        """
        batch_size, seq_len = pose_features.shape[:2]
        
        # 1. 공간 특징 추출
        encoder_input = self.spatial_encoder(pose_features)  # [batch, seq_len, embed_dim]
        
        # 2. 위치 인코딩 추가
        encoder_input = encoder_input.transpose(0, 1)  # [seq_len, batch, embed_dim]
        encoder_input = self.pos_encoding(encoder_input)
        encoder_input = encoder_input.transpose(0, 1)  # [batch, seq_len, embed_dim]
        
        # 3. Encoder 마스크 생성
        if frame_masks is not None:
            encoder_key_padding_mask = ~frame_masks  # True = 패딩
        else:
            encoder_key_padding_mask = None
        
        # 4. Transformer Encoder
        encoder_output = self.transformer_encoder(
            encoder_input, 
            src_key_padding_mask=encoder_key_padding_mask
        )
        
        # 5. 모드 결정: 훈련 중이거나 강제 teacher forcing 모드인 경우
        if (self.training or force_training_mode) and vocab_ids is not None:
            # 훈련 모드: Teacher Forcing
            return self._forward_training(encoder_output, vocab_ids, vocab_masks)
        else:
            # 추론 모드: 프레임별 분류
            return self._forward_inference(encoder_output)
    
    def _forward_training(self, encoder_output, vocab_ids, vocab_masks):
        """훈련 모드 포워드 (Teacher Forcing)"""
        batch_size, vocab_len = vocab_ids.shape
        
        # Decoder 입력 준비 (시작 토큰 추가)
        decoder_input_ids = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.long, device=vocab_ids.device),
            vocab_ids[:, :-1]
        ], dim=1)
        
        # Vocabulary 임베딩
        decoder_input = self.vocab_embedding(decoder_input_ids)  # [batch, vocab_len, embed_dim]
        
        # 위치 인코딩 추가
        decoder_input = decoder_input.transpose(0, 1)
        decoder_input = self.vocab_pos_encoding(decoder_input)
        decoder_input = decoder_input.transpose(0, 1)
        
        # 마스크 생성
        tgt_mask = self.create_causal_mask(vocab_len).to(vocab_ids.device)
        tgt_key_padding_mask = ~vocab_masks if vocab_masks is not None else None
        
        # Transformer Decoder
        decoder_output = self.transformer_decoder(
            decoder_input,
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 출력 헤드들
        word_logits = self.word_classifier(decoder_output)
        boundary_logits = self.boundary_detector(decoder_output)  
        confidence_scores = self.confidence_head(decoder_output).squeeze(-1)
        
        return {
            'word_logits': word_logits,  # [batch, vocab_len, vocab_size]
            'boundary_logits': boundary_logits,  # [batch, vocab_len, 3]
            'confidence_scores': confidence_scores  # [batch, vocab_len]
        }
    
    def _forward_inference(self, encoder_output):
        """추론 모드 포워드 (프레임별 분류)"""
        batch_size, seq_len, embed_dim = encoder_output.shape
        
        # 각 프레임에 대해 직접 분류
        word_logits = self.word_classifier(encoder_output)  # [batch, seq_len, vocab_size]
        boundary_logits = self.boundary_detector(encoder_output)  # [batch, seq_len, 3]
        confidence_scores = self.confidence_head(encoder_output).squeeze(-1)  # [batch, seq_len]
        
        return {
            'word_logits': word_logits,
            'boundary_logits': boundary_logits,
            'confidence_scores': confidence_scores
        }
    
    def compute_loss(self, outputs, targets, vocab_masks=None):
        """손실 함수 계산"""
        word_logits = outputs['word_logits']
        boundary_logits = outputs['boundary_logits']
        confidence_scores = outputs['confidence_scores']
        
        vocab_ids = targets['vocab_ids']
        boundary_labels = targets.get('boundary_labels', None)
        confidence_targets = targets.get('confidence_targets', None)
        
        # 디버깅을 위한 차원 로깅
        logger.debug(f"🔍 Loss computation debug:")
        logger.debug(f"  word_logits shape: {word_logits.shape}")
        logger.debug(f"  vocab_ids shape: {vocab_ids.shape}")
        logger.debug(f"  boundary_logits shape: {boundary_logits.shape}")
        
        # 차원 일치성 검증
        batch_size = word_logits.shape[0]
        seq_len = word_logits.shape[1]
        
        # vocab_ids의 배치 크기가 word_logits와 일치하는지 확인
        if vocab_ids.shape[0] != batch_size:
            logger.error(f"❌ Batch size mismatch: word_logits={batch_size}, vocab_ids={vocab_ids.shape[0]}")
            raise ValueError(f"Batch size mismatch: word_logits batch={batch_size}, vocab_ids batch={vocab_ids.shape[0]}")
        
        # 시퀀스 길이 일치성 검증 (vocab_ids가 2D인 경우)
        if len(vocab_ids.shape) > 1 and vocab_ids.shape[1] != seq_len:
            logger.warning(f"⚠️ Sequence length mismatch: word_logits={seq_len}, vocab_ids={vocab_ids.shape[1]}")
            # 짧은 쪽에 맞춰 자르기
            min_len = min(seq_len, vocab_ids.shape[1])
            word_logits = word_logits[:, :min_len, :]
            vocab_ids = vocab_ids[:, :min_len]
            if boundary_logits is not None:
                boundary_logits = boundary_logits[:, :min_len, :]
        
        losses = {}
        
        # 1. 단어 분류 손실 (Label Smoothing 적용)
        if self.label_smoothing > 0 and self.word_loss_fn is not None:
            # Label Smoothing 손실 사용
            word_loss = self.word_loss_fn(
                word_logits.reshape(-1, self.vocab_size),
                vocab_ids.reshape(-1)
            )
        else:
            # 일반 CrossEntropy 손실 사용
            word_loss = F.cross_entropy(
                word_logits.reshape(-1, self.vocab_size),
                vocab_ids.reshape(-1),
                ignore_index=0  # 패딩 무시
            )
        losses['word_loss'] = word_loss
        
        # 2. 경계 탐지 손실 (있는 경우)
        if boundary_labels is not None and boundary_logits is not None:
            # 차원 확인 및 조정
            if len(boundary_labels.shape) > 1:
                boundary_labels = boundary_labels.reshape(-1)
            
            boundary_loss = F.cross_entropy(
                boundary_logits.reshape(-1, 3),  # view 대신 reshape 사용
                boundary_labels,
                ignore_index=-1  # 무시할 라벨
            )
            losses['boundary_loss'] = boundary_loss
        else:
            losses['boundary_loss'] = torch.tensor(0.0, device=word_loss.device)
        
        # 3. 신뢰도 손실 (있는 경우)
        if confidence_targets is not None and confidence_scores is not None:
            # 차원 확인 및 조정
            if len(confidence_targets.shape) > 1:
                confidence_targets = confidence_targets.reshape(-1)
            
            confidence_loss = F.mse_loss(
                confidence_scores.reshape(-1),  # view 대신 reshape 사용
                confidence_targets
            )
            losses['confidence_loss'] = confidence_loss
        else:
            losses['confidence_loss'] = torch.tensor(0.0, device=word_loss.device)
        
        # 4. 총 손실
        total_loss = (self.word_loss_weight * losses['word_loss'] +
                     self.boundary_loss_weight * losses['boundary_loss'] +
                     self.confidence_loss_weight * losses['confidence_loss'])
        
        losses['total_loss'] = total_loss
        
        return losses

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [N, num_classes] - 예측 logits
            target: [N] - 타겟 클래스 인덱스
        """
        pred = pred.log_softmax(dim=-1)
        
        # One-hot 인코딩 생성
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # ignore_index 처리
        mask = (target != self.ignore_index).float()
        true_dist = true_dist * mask.unsqueeze(1)
        
        # KL Divergence 계산
        loss = torch.sum(-true_dist * pred, dim=-1)
        loss = loss * mask
        
        return loss.sum() / mask.sum().clamp(min=1)

class RealtimeDecoder:
    """실시간 디코딩 클래스"""
    
    def __init__(self, 
                 vocab_size: int,
                 confidence_threshold: float = 0.7,
                 boundary_threshold: float = 0.8):
        self.vocab_size = vocab_size
        self.confidence_threshold = confidence_threshold
        self.boundary_threshold = boundary_threshold
        
        self.state = "WAITING"  # WAITING/IN_WORD/COOLDOWN
        self.current_word_buffer = []
        self.confidence_buffer = []
        self.cooldown_frames = 0
        self.cooldown_duration = 10  # 10프레임 대기
    
    def process_frame_output(self, word_logits, boundary_logits, confidence):
        """프레임별 출력 처리"""
        # 경계 예측
        boundary_probs = F.softmax(boundary_logits, dim=-1)
        boundary_pred = torch.argmax(boundary_probs)
        boundary_conf = torch.max(boundary_probs)
        
        # 단어 예측
        word_probs = F.softmax(word_logits, dim=-1)
        word_pred = torch.argmax(word_probs)
        word_conf = torch.max(word_probs)
        
        # 상태 머신
        if self.state == "WAITING":
            if (boundary_pred == 0 and  # START
                boundary_conf > self.boundary_threshold):
                self.state = "IN_WORD"
                self.current_word_buffer = [word_logits]
                self.confidence_buffer = [confidence]
                
        elif self.state == "IN_WORD":
            self.current_word_buffer.append(word_logits)
            self.confidence_buffer.append(confidence)
            
            if (boundary_pred == 2 and  # END
                boundary_conf > self.boundary_threshold and
                len(self.current_word_buffer) > 3):  # 최소 길이
                predicted_word = self.emit_word()
                self.state = "COOLDOWN"
                self.cooldown_frames = self.cooldown_duration
                return predicted_word
                
        elif self.state == "COOLDOWN":
            self.cooldown_frames -= 1
            if self.cooldown_frames <= 0:
                self.state = "WAITING"
        
        return None
    
    def emit_word(self):
        """단어 방출"""
        if not self.current_word_buffer:
            return None
        
        # 버퍼 내 예측들의 앙상블
        avg_logits = torch.stack(self.current_word_buffer).mean(dim=0)
        predicted_word = torch.argmax(avg_logits)
        
        # 평균 신뢰도
        avg_confidence = torch.stack(self.confidence_buffer).mean()
        
        self.reset_state()
        
        if avg_confidence > self.confidence_threshold:
            return predicted_word.item()
        else:
            return None
    
    def reset_state(self):
        """상태 초기화"""
        self.current_word_buffer = []
        self.confidence_buffer = []

# 테스트 코드
if __name__ == "__main__":
    print("🤖 수화 인식 모델 테스트")
    print("=" * 40)
    
    # 모델 생성
    vocab_size = 442  # 실제 vocab 크기에 맞춰 조정
    model = SequenceToSequenceSignModel(
        vocab_size=vocab_size,
        embed_dim=256,
        num_encoder_layers=4,
        num_decoder_layers=3,
        num_heads=8
    )
    
    # 더미 데이터 생성
    batch_size, seq_len, vocab_len = 2, 100, 5
    pose_features = torch.randn(batch_size, seq_len, 133, 3)
    vocab_ids = torch.randint(1, vocab_size, (batch_size, vocab_len))
    frame_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    vocab_masks = torch.ones(batch_size, vocab_len, dtype=torch.bool)
    
    print(f"✅ 모델 생성 완료:")
    print(f"   - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - 훈련 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 훈련 모드 테스트
    model.train()
    outputs = model(pose_features, vocab_ids, frame_masks, vocab_masks)
    print(f"\n🏋️ 훈련 모드 출력:")
    print(f"   - Word logits: {outputs['word_logits'].shape}")
    print(f"   - Boundary logits: {outputs['boundary_logits'].shape}")
    print(f"   - Confidence scores: {outputs['confidence_scores'].shape}")
    
    # 추론 모드 테스트
    model.eval()
    with torch.no_grad():
        outputs = model(pose_features, frame_masks=frame_masks)
    print(f"\n🔍 추론 모드 출력:")
    print(f"   - Word logits: {outputs['word_logits'].shape}")
    print(f"   - Boundary logits: {outputs['boundary_logits'].shape}")
    print(f"   - Confidence scores: {outputs['confidence_scores'].shape}")
    
    # 실시간 디코더 테스트
    decoder = RealtimeDecoder(vocab_size)
    print(f"\n⚡ 실시간 디코더 테스트:")
    for i in range(20):
        word_logits = outputs['word_logits'][0, i]
        boundary_logits = outputs['boundary_logits'][0, i]
        confidence = outputs['confidence_scores'][0, i]
        
        result = decoder.process_frame_output(word_logits, boundary_logits, confidence)
        if result is not None:
            print(f"   프레임 {i}: 단어 {result} 검출!")
    
    print("\n🎉 모든 테스트 완료!")
