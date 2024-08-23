import subprocess
import sys

# requirements.txt 파일 경로
requirements_file = '/media/user/5E1227AF12278B5B/Github upload/requirements.txt'

def install_packages(requirements_file):
    with open(requirements_file, 'r') as file:
        packages = file.readlines()

    for package in packages:
        package = package.strip()
        if package:  # 빈 줄은 건너뜀
            print(f"Installing {package}...")
            try:
                # 패키지 설치
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"{package} installed successfully.")
            except subprocess.CalledProcessError:
                # 설치 실패 시 메시지 출력하고 건너뜀
                print(f"Failed to install {package}. Skipping...")

install_packages(requirements_file)
