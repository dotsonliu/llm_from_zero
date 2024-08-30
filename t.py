import torch
def check_cuda_version():
    cuda_version = torch.version.cuda
    required_driver_version = "<根据你的CUDA运行时版本确定>"
    
    try:
        torch.cuda.init()
        current_driver_version = torch.cuda.driver_version
        print(f"当前CUDA驱动程序版本：{current_driver_version}")
        
        if current_driver_version < required_driver_version:
            raise Exception("CUDA driver version is insufficient for CUDA runtime version")
        
        print("CUDA版本兼容，可以继续执行CUDA计算任务")
        # 在这里执行你的CUDA计算任务
        
    except Exception as e:
        print(f"错误：{e}")
check_cuda_version()
