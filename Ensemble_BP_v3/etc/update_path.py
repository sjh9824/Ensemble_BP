import json


def replace_path_in_json(json_data, old_path, new_path):
    # json_data가 dict 또는 list일 경우에만 재귀적으로 탐색
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            json_data[key] = replace_path_in_json(value, old_path, new_path)
    elif isinstance(json_data, list):
        for index, item in enumerate(json_data):
            json_data[index] = replace_path_in_json(item, old_path, new_path)
    elif isinstance(json_data, str):
        # 문자열인 경우, 경로를 교체
        json_data = json_data.replace(old_path, new_path)

    return json_data


# JSON 파일 경로 설정
input_file_path = '/mnt/HDD_3/Code_jaehyuk/input.json'
output_file_path = '/mnt/HDD_3/Code_jaehyuk/input_2.json'

# 교체할 경로 설정
old_path = '/media/user/5E1227AF12278B5B/Github upload/Data'
new_path = '/mnt/HDD_3/Code_jaehyuk/Data'

# JSON 파일 열기
with open(input_file_path, 'r') as file:
    data = json.load(file)

# 경로 교체
updated_data = replace_path_in_json(data, old_path, new_path)

# 수정된 JSON 데이터를 파일에 저장
with open(output_file_path, 'w') as file:
    json.dump(updated_data, file, indent=4)

print(f"경로가 성공적으로 교체되었습니다. 결과는 {output_file_path} 파일에 저장되었습니다.")