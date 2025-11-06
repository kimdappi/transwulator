import mediapipe as mp
import cv2, json, os, glob

# === 입력 폴더 & 출력 폴더 설정 ===
INPUT_DIR = "video"   # mp4 파일이 들어있는 폴더 이름
OUTPUT_DIR = "pose"    # json 파일이 저장될 폴더 이름

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# === 폴더 내 mp4 파일 모두 검색 ===
video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))

if not video_files:
    print(f"⚠️ {INPUT_DIR} 폴더에 mp4 파일이 없습니다.")
else:
    print(f"🎥 총 {len(video_files)}개의 영상 처리 시작...")

# === 각 파일별 변환 ===
for video_path in video_files:
    filename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_pose.json")

    print(f"\n▶ 변환 중: {filename}.mp4 ...")

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)
        hands_result = hands.process(rgb)

        frame_data = {"frame": frame_idx, "pose": [], "hands": []}

        if pose_result.pose_landmarks:
            frame_data["pose"] = [
                [lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark
            ]

        if hands_result.multi_hand_landmarks:
            for hand in hands_result.multi_hand_landmarks:
                frame_data["hands"].append(
                    [[lm.x, lm.y, lm.z] for lm in hand.landmark]
                )

        frames.append(frame_data)
        frame_idx += 1

    cap.release()

    # === JSON 저장 ===
    with open(output_path, "w") as f:
        json.dump(frames, f)

    print(f"✅ 완료: {output_path} (총 {len(frames)} 프레임)")

print("\n🎯 모든 영상 변환 완료!")
