import os, json
import pandas as pd

# JSON 폴더 경로
data_dir = "C:/Users/rosie/OneDrive/Desktop/tuningdata"
pairs = []

for file_name in os.listdir(data_dir):
    if file_name.endswith(".json"):
        with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as f:
            d = json.load(f)

            # 한국어 원문
            src_text = d["krlgg_sntenc"]["koreanText"]
            # 도메인/주제 (realm, thema)
            realm = d["krlgg_sntenc"].get("realm", "")
            thema = d["krlgg_sntenc"].get("thema", "")

            # 글로스 시퀀스 (strong 제스처 기준)
            gloss_list = [g["gloss_id"] for g in d["sign_script"]["sign_gestures_strong"]]
            tgt_text = " ".join(gloss_list)

            # ✅ EXAONE Instruct 모델 학습용 프롬프트 포맷
            src_prompt = (
                f"Instruction: 한국어 문장을 한국수어 글로스로 번역하시오.\n"
                f"Domain: {realm} | Thema: {thema}\n"
                f"Input: {src_text}\nOutput:"
            )

            pairs.append({"src": src_prompt, "tgt": tgt_text})

# CSV로 저장
output_path = "C:/Users/rosie/OneDrive/Desktop/tuningdata/ksl_translation_dataset_exaone.csv"
df = pd.DataFrame(pairs)
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ 변환 완료! 파일 저장 위치: {output_path}")
print(df.head())
