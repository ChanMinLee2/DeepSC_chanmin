import torch
print("PyTorch 버전:", torch.__version__)         # PyTorch 버전 확인
print("CUDA 버전:", torch.version.cuda)           # PyTorch와 연결된 CUDA 버전 확인
print("GPU 사용 가능 여부:", torch.cuda.is_available())  # GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))    # GPU 이름 출력
