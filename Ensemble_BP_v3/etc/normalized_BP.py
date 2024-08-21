import json

# BP 값을 normalization하기 위한 함수
def normalize_bp(bp, min_sbp=70, max_sbp=180, min_dbp=40, max_dbp=120):
    sbp, dbp = bp
    normalized_sbp = (sbp - min_sbp) / (max_sbp - min_sbp)
    normalized_dbp = (dbp - min_dbp) / (max_dbp - min_dbp)
    return [normalized_sbp, normalized_dbp]

# JSON 파일 로드 (경로는 상황에 맞게 변경)
json_file_path = '/mnt/HDD_3/Code_jaehyuk/input_2.json'

# JSON 파일 읽기
with open(json_file_path, 'r') as file:
    data = json.load(file)

# BP 값을 normalization 처리
for subject_key, subject_data in data.items():
    for input_key, input_data in subject_data.items():
        original_bp = input_data['BP']
        normalized_bp = normalize_bp(original_bp)
        data[subject_key][input_key]['BP'] = normalized_bp

# 수정된 데이터를 원래 JSON 파일에 덮어쓰기
with open(json_file_path, 'w') as file:
    json.dump(data, file, indent=4)

# 또는 새로운 파일에 저장하기 (경로는 상황에 맞게 변경)
new_json_file_path = '/mnt/HDD_3/Code_jaehyuk/input_2.json'
with open(new_json_file_path, 'w') as file:
    json.dump(data, file, indent=4)

print("Normalization 완료 및 저장되었습니다.")
