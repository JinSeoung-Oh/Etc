import os
import cv2
import glob
import shutil
from tqdm import tqdm

def extract_frames(video_path, output_folder, frame_interval=1, min_disk_space_gb=10):
    """
    비디오 파일에서 프레임을 추출하여 이미지로 저장합니다.
    
    Args:
        video_path (str): 비디오 파일 경로
        output_folder (str): 프레임 이미지를 저장할 폴더 경로
        frame_interval (int): 프레임 추출 간격 (기본값: 1, 모든 프레임 추출)
    """
    # 비디오 파일명 추출 (확장자 제외)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 출력 폴더 생성
    frame_folder = os.path.join(output_folder, video_name)
    os.makedirs(frame_folder, exist_ok=True)
    
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: 비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    # 총 프레임 수
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 현재 프레임 번호
    frame_count = 0
    saved_count = 0
    
    # 프레임 추출
    while True:
        # 주기적으로 디스크 공간 확인 (100프레임마다)
        if frame_count % 100 == 0 and frame_count > 0:
            if not check_disk_space(output_folder, min_disk_space_gb):
                print(f"\n경고: {video_path} 처리 중 디스크 여유 공간이 부족합니다. 현재 비디오 처리를 중단합니다.")
                break
                
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # frame_interval에 따라 프레임 저장
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(frame_folder, f"{video_name}_frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    # 비디오 파일 닫기
    cap.release()
    
    print(f"완료: {video_path} - {saved_count}개 프레임 저장")
    return saved_count

def check_disk_space(path, min_space_gb=10):
    """
    디스크 여유 공간을 확인하고 최소 요구 공간보다 적으면 False를 반환합니다.
    
    Args:
        path (str): 확인할 경로
        min_space_gb (float): 최소 요구 여유 공간 (GB 단위, 기본값: 10GB)
    
    Returns:
        bool: 충분한 공간이 있으면 True, 부족하면 False
    """
    # 디스크 여유 공간 확인 (바이트 단위)
    free_bytes = shutil.disk_usage(path).free
    # GB 단위로 변환
    free_gb = free_bytes / (1024 ** 3)
    
    return free_gb >= min_space_gb

def main():
    # 입력 폴더 경로 (비디오 클립들이 있는 디렉토리)
    input_folder = "C:\\Users\\kikn1\\AV-ASD\\dataset\\clips_video"  # 여기에 비디오 클립이 저장된 디렉토리 경로를 입력하세요
    
    # 출력 폴더 경로 (프레임 이미지를 저장할 디렉토리)
    output_folder = "C:\\Users\\kikn1\\ASD_fream"  # 여기에 프레임 이미지를 저장할 디렉토리 경로를 입력하세요
    
    # 프레임 간격 설정
    frame_interval = 1  # 모든 프레임 추출 (값 변경 가능)
    
    # 최소 디스크 여유 공간 설정 (GB 단위)
    min_disk_space_gb = 10  # 10GB 이하로 떨어지면 처리 중단
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 비디오 파일 목록 가져오기 (os.walk 사용)
    video_files = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in video_extensions:
                video_path = os.path.join(root, file)
                video_files.append(video_path)
    
    print(f"총 {len(video_files)}개의 비디오 파일을 찾았습니다.")
    
    # 모든 비디오 파일에서 프레임 추출
    total_frames = 0
    disk_space_error = False
    
    for video_path in tqdm(video_files, desc="비디오 처리 중"):
        # 디스크 여유 공간 확인
        if not check_disk_space(output_folder, min_disk_space_gb):
            print(f"\n경고: 디스크 여유 공간이 {min_disk_space_gb}GB 이하로 떨어졌습니다. 작업을 중단합니다.")
            disk_space_error = True
            break
            
        frames = extract_frames(video_path, output_folder, frame_interval, min_disk_space_gb)
        if frames:
            total_frames += frames
    
    if disk_space_error:
        print(f"디스크 용량 부족으로 작업이 중단되었습니다. 총 {total_frames}개의 프레임이 {output_folder}에 저장되었습니다.")
    else:
        print(f"모든 처리 완료! 총 {total_frames}개의 프레임이 {output_folder}에 저장되었습니다.")

if __name__ == "__main__":
    main()
