from deepface import DeepFace
import cv2
import numpy as np
import json
import os
import tensorflow as tf

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FONT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 255, 0)
OVERLAY_COLOR = (0, 0, 0)

# ============================================
# 🔧 경로 기본 설정
BASE_VIDEO_DIR = os.path.join("data", "video")
DEBUG_DIR = os.path.join(BASE_VIDEO_DIR, "debug_frames")
os.makedirs(DEBUG_DIR, exist_ok=True)
# ============================================


def analyze_frames_deepface(video_path, start_sec, end_sec,
                            save_debug=True,
                            display_in_colab=False):
    """특정 구간의 중앙 프레임 1개만 감정 분석"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 비디오 열기 실패: {video_path}")
        return {"emotion_scores": {"neutral": 1.0}, "dominant_emotion": "neutral"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    # ✅ 중앙 프레임 계산
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    center_frame = int((start_frame + end_frame) / 2)

    all_scores = []
    if save_debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)  # 🔧 data/video/debug_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, center_frame)
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print(f"⚠️ 프레임 추출 실패 (frame={center_frame})")
        cap.release()
        return {"emotion_scores": {"neutral": 1.0}, "dominant_emotion": "neutral"}

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = frame.copy()
    current_dominant_emotion = "No Face"

    try:
        # ✅ RetinaFace로 얼굴 탐지 + 감정 분석
        analysis_results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            detector_backend="retinaface",
            enforce_detection=False
        )

        # DeepFace 버전별 구조 호환
        if isinstance(analysis_results, dict):
            if "instances" in analysis_results:
                analysis_results = analysis_results["instances"]
            else:
                analysis_results = [analysis_results]

        if analysis_results and len(analysis_results) > 0:
            first_face = analysis_results[0]
            emo = first_face.get("emotion", {})
            region = first_face.get("region", {})
            current_dominant_emotion = first_face.get("dominant_emotion", "unknown")

            if emo:
                all_scores.append(emo)

            # ✅ 시각화
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), BOX_COLOR, 2)

            overlay = frame_bgr.copy()
            text_x, text_y = x + w + 5, y
            text_bg_h = (len(emo) + 1) * 20
            cv2.rectangle(overlay, (text_x, text_y),
                          (text_x + 120, text_y + text_bg_h),
                          OVERLAY_COLOR, -1)
            processed_frame = cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0)

            cv2.putText(processed_frame, f"-> {current_dominant_emotion}",
                        (text_x, text_y + 15), FONT, FONT_SCALE, BOX_COLOR, FONT_THICKNESS)

            line_y = text_y + 15
            for emotion, score in emo.items():
                line_y += 20
                text = f"{emotion}: {score:.1f}%"
                cv2.putText(processed_frame, text, (text_x, line_y),
                            FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

    except Exception as e:
        print(f"⚠️ DeepFace 분석 실패 (frame={center_frame}): {e}")
    finally:
        tf.keras.backend.clear_session()

    if save_debug:
        ### 🔧 debug 프레임 저장 경로 수정
        fname = os.path.join(
            DEBUG_DIR,
            f"frame_{round(start_sec,2)}_{round(end_sec,2)}_center_analyzed.jpg"
        )
        cv2.imwrite(fname, processed_frame[:, :, ::-1])
        print(f"💾 디버그 프레임 저장됨: {fname}")
    cap.release()

    if all_scores:
        final_scores = all_scores[0]
        dominant = max(final_scores, key=final_scores.get)
    else:
        final_scores = {"neutral": 1.0}
        dominant = "neutral"

    return {"emotion_scores": final_scores, "dominant_emotion": dominant}


def map_emotions_to_transcript(transcript_json, video_path, display_each_frame=False):
    with open(transcript_json, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    results = []
    for seg_idx, seg in enumerate(transcripts):
        text = seg.get("text", "").strip()
        if not text:
            continue

        start_sec = float(seg.get("start", 0.0))
        end_sec = float(seg.get("end", start_sec + 0.5))

        print(f"\n--- 📝 {seg_idx+1}/{len(transcripts)} 문장 분석 중: [{round(start_sec,2)}s - {round(end_sec,2)}s] \"{text}\" ---")

        emo_scores = analyze_frames_deepface(
            video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            save_debug=True,
            display_in_colab=display_each_frame
        )

        results.append({
            "file": seg.get("file"),
            "text": text,
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "emotion_scores": emo_scores["emotion_scores"],
            "dominant_emotion": emo_scores["dominant_emotion"]
        })

    return results


if __name__ == "__main__":
    ### 🔧 파일 경로 수정
    transcript_path = os.path.join(BASE_VIDEO_DIR, "whisper_transcript.json")
    video_path = os.path.join(BASE_VIDEO_DIR, "fixed_video.mp4")
    output_json = os.path.join(BASE_VIDEO_DIR, "mapped_transcript.json")

    display_each_frame_in_colab = False

    mapped_results = map_emotions_to_transcript(
        transcript_path,
        video_path,
        display_each_frame=display_each_frame_in_colab
    )

    ### 🔧 결과 저장 경로 수정
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(mapped_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 완료: {output_json} 생성됨")
    print(f"📸 디버그 프레임은 {DEBUG_DIR} 폴더에 저장됨")
