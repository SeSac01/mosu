import cv2
import numpy as np
import requests
import time
import json
import argparse
from typing import List, Optional, Tuple
from pathlib import Path
import logging
from collections import deque

def bbox_xyxy2cs(bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """bbox(x1,y1,x2,y2) -> center, scale 변환"""
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    box_w = x2 - x1
    box_h = y2 - y1
    scale = np.array([box_w, box_h], dtype=np.float32)
    return center, scale

def fix_aspect_ratio(scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """scale의 비율을 고정 (width 기준)"""
    w, h = scale
    if w > h * aspect_ratio:
        h = w / aspect_ratio
    else:
        w = h * aspect_ratio
    return np.array([w, h], dtype=np.float32)

def get_warp_matrix(center, scale, rot, output_size):
    """아핀 변환 행렬 생성"""
    src_w, src_h = scale
    src_dir = get_dir([0, src_h * -0.5], rot)
    dst_w, dst_h = output_size
    dst_dir = np.array([0, dst_h * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    src[2:] = get_third_point(src[0, :], src[1, :])

    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = dst[0, :] + dst_dir
    dst[2:] = get_third_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(src, dst)
    return trans

def get_dir(src_point, rot_rad):
    """회전된 방향 벡터 계산"""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [src_point[0] * cs - src_point[1] * sn,
                  src_point[0] * sn + src_point[1] * cs]
    return np.array(src_result)

def get_third_point(a, b):
    """세 번째 점 계산"""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


class EdgeServer:
    """엣지 서버 - 웹캠 캡처 + YOLO 검출 + 크롭 + 전송"""
    
    def __init__(self, 
                 pose_server_url: str,
                 camera_id: int = 0,
                 window_size: Tuple[int, int] = (1280, 720),
                 jpeg_quality: int = 85,
                 max_fps: int = 30):
        
        self.pose_server_url = pose_server_url.rstrip('/')
        self.camera_id = camera_id
        self.window_size = window_size
        self.jpeg_quality = jpeg_quality
        self.max_fps = max_fps
        
        # YOLO 검출기 초기화
        self.detector = EdgeYOLODetector()
        
        # 웹캠 초기화
        self.cap = None
        self.init_camera()
        
        # 성능 측정
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        
        print(f"✅ 엣지 서버 초기화 완료")
        print(f"   - 포즈 서버: {self.pose_server_url}")
        print(f"   - 카메라: {camera_id}")
        print(f"   - 해상도: {window_size}")
        print(f"   - JPEG 품질: {jpeg_quality}%")
    
    def init_camera(self):
        """웹캠 초기화"""
        print(f"📹 웹캠 연결 중... (카메라 ID: {self.camera_id})")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 웹캠 열기 실패 (카메라 ID: {self.camera_id})")
        
        # 웹캠 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)
        
        # 실제 웹캠 해상도 확인
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ 웹캠 연결 성공: {actual_width}x{actual_height}, {actual_fps:.1f}fps")
    
    def send_crop_to_pose_server(self, crop_image: np.ndarray, bbox: List[float], frame_id: int) -> Optional[dict]:
        """크롭된 이미지를 포즈 서버로 전송"""
        try:
            # JPEG 인코딩
            ret, jpeg_data = cv2.imencode('.jpg', crop_image, 
                                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ret:
                return None
            
            # 요청 데이터 준비
            files = {'image': ('crop.jpg', jpeg_data.tobytes(), 'image/jpeg')}
            data = {
                'frame_id': frame_id,
                'bbox': json.dumps(bbox),
                'timestamp': time.time()
            }
            
            # POST 요청
            response = requests.post(
                f"{self.pose_server_url}/estimate_pose",
                files=files,
                data=data,
                timeout=5.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ 포즈 서버 응답 오류: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            print(f"⚠️ 포즈 서버 통신 실패: {e}")
            return None
        except Exception as e:
            print(f"⚠️ 이미지 전송 실패: {e}")
            return None
        
    def transform_keypoints_to_original(self, keypoints: np.ndarray, bbox: List[float]) -> np.ndarray:
        """크롭 좌표(288x384)의 키포인트를 원본 이미지 좌표로 변환"""
        try:
            # RTMW와 동일한 변환 과정을 역변환
            input_width, input_height = 288, 384
            
            # 1. bbox를 center, scale로 변환 (crop 시와 동일)
            bbox_array = np.array(bbox, dtype=np.float32)
            center, scale = bbox_xyxy2cs(bbox_array)
            
            # 2. aspect ratio 고정 (crop 시와 동일)
            aspect_ratio = input_width / input_height  # 0.75
            scale = fix_aspect_ratio(scale, aspect_ratio)
            
            # 3. 아핀 변환 매트릭스 계산 (crop 시와 동일)
            warp_mat = get_warp_matrix(
                center=center,
                scale=scale,
                rot=0.0,
                output_size=(input_width, input_height)
            )
            
            # 4. 역변환 매트릭스 계산
            inv_warp_mat = cv2.invertAffineTransform(warp_mat)
            
            # 5. 키포인트를 homogeneous coordinates로 변환
            num_keypoints = keypoints.shape[0]
            kpts_homo = np.ones((num_keypoints, 3))
            kpts_homo[:, :2] = keypoints[:, :2]
            
            # 6. 역변환 적용
            original_keypoints = np.zeros_like(keypoints)
            for i in range(num_keypoints):
                transformed_pt = inv_warp_mat @ kpts_homo[i]
                original_keypoints[i, 0] = transformed_pt[0]
                original_keypoints[i, 1] = transformed_pt[1]
            
            return original_keypoints
        
        except Exception as e:
            print(f"⚠️ 키포인트 변환 실패: {e}")
            return keypoints  # 실패시 원본 반환
    
    def visualize_results(self, image: np.ndarray, person_boxes: List[List[float]], 
                         pose_results: List[dict]) -> np.ndarray:
        """결과 시각화"""
        vis_image = image.copy()
        
        # 검출된 바운딩박스 그리기
        for i, bbox in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if i == 0 else (255, 0, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, f"Person {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 포즈 결과가 있으면 키포인트 그리기 (원본 좌표로 변환 필요)
        for i, pose_result in enumerate(pose_results):
            if pose_result and 'keypoints' in pose_result:
                keypoints = np.array(pose_result['keypoints'])
                scores = np.array(pose_result['scores'])
                
                original_keypoints = self.transform_keypoints_to_original(keypoints, bbox)
                
                for j, (kpt, score) in enumerate(zip(keypoints, scores)):
                    if score > 0.3:
                        # 크롭 좌표(288x384)를 원본 바운딩박스 좌표로 변환
                        x = int(bbox[0] + (kpt[0] / 288.0) * bbox_w)
                        y = int(bbox[1] + (kpt[1] / 384.0) * bbox_h)
                        
                        if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                            if score > 0.8:
                                kpt_color = (0, 255, 0)    # 높은 신뢰도: 초록
                            elif score > 0.6:
                                kpt_color = (0, 255, 255)  # 중간 신뢰도: 노랑
                            else:
                                kpt_color = (0, 0, 255)    # 낮은 신뢰도: 빨강
                            
                            cv2.circle(vis_image, (x, y), 3, kpt_color, -1)
        
        # 성능 정보 표시
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            cv2.putText(vis_image, f"Edge FPS: {avg_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(vis_image, f"YOLO11L Edge Server", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Pose Server: {self.pose_server_url.split('://')[-1]}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Detections: {len(person_boxes)}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis_image
    
    def run(self):
        """메인 실행 루프"""
        print(f"\n🚀 엣지 서버 실행 시작")
        print(f"🎮 조작법:")
        print(f"   - ESC: 종료")
        print(f"   - S: 스크린샷")
        print(f"   - SPACE: 일시정지/재생")
        
        paused = False
        screenshot_count = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("❌ 프레임 읽기 실패")
                        break
                    
                    start_time = time.time()
                    
                    # 1. YOLO 검출
                    person_boxes = self.detector.detect_persons(frame)
                    
                    # 2. 각 사람에 대해 크롭 + 포즈 서버로 전송
                    pose_results = []
                    for i, bbox in enumerate(person_boxes):
                        bbox = person_boxes[0]
                        crop_image = self.detector.crop_person_image_rtmw(frame, bbox)
                        if crop_image is not None:
                            # 포즈 서버로 전송
                            pose_result = self.send_crop_to_pose_server(
                                crop_image, bbox, self.frame_count * 1000 + i
                            )
                            pose_results.append(pose_result)
                        else:
                            pose_results.append(None)
                    
                    # 3. FPS 계산
                    process_time = time.time() - start_time
                    fps = 1.0 / process_time if process_time > 0 else 0
                    self.fps_history.append(fps)
                    
                    # 4. 시각화
                    vis_frame = self.visualize_results(frame, person_boxes, pose_results)
                    self.frame_count += 1
                
                # 화면 표시
                cv2.imshow('YOLO Edge Server', vis_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('s') or key == ord('S'):  # 스크린샷
                    screenshot_name = f"edge_screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(screenshot_name, vis_frame)
                    print(f"📸 스크린샷 저장: {screenshot_name}")
                    screenshot_count += 1
                elif key == ord(' '):  # 일시정지/재생
                    paused = not paused
                    print(f"⏸️ {'일시정지' if paused else '재생'}")
                
                # 주기적 통계 출력
                if self.frame_count % 60 == 0 and self.frame_count > 0:
                    avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                    print(f"📊 프레임 {self.frame_count}: {avg_fps:.1f}fps, "
                          f"{len(person_boxes)}명 검출")
                    
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 엣지 서버를 중단했습니다.")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"\n📊 엣지 서버 완료:")
            print(f"   - 처리된 프레임: {self.frame_count}")
            print(f"   - 평균 FPS: {avg_fps:.1f}")
