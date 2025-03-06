import gdown
import tarfile

#https://drive.google.com/drive/folders/1FKEjZnKfu9KnSGfnr8oGVUBSPqnptzJc

# 파일 ID 및 다운로드 URL 설정
file_id = '18zzqG50xjBI2cqOiG0MSQ4_C5IcEbdZV' #clever-easy
url = f'https://drive.google.com/uc?id={file_id}'
output = 'clevr-easy.tar.gz'

# 파일 다운로드
gdown.download(url, output, quiet=False)

# 압축 해제 경로 설정
extract_path = './data'

# tar.gz 파일 압축 해제
with tarfile.open(output, 'r:gz') as tar:
    tar.extractall(path=extract_path)
