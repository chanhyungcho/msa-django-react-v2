import numpy as np
import tensorflow as tf
import torch
import sklearn
from tensorflow.python.client import device_lib
if __name__ == '__main__':
    print(f'numpy version : {np.__version__}')
    print(f'tensorflow version : {tf.__version__}')
    print(f'torch version : {torch.__version__}')
    print(f'sklearn version : {sklearn.__version__}')
    print(f'이 PC에 설치된 디바이스 상세보기 : {device_lib.list_local_devices()}')
    print(f'CUDA 프로그래밍 가능여부 :  {torch.cuda.is_available()}')
    print(f'CUDA 프로그래밍 가능여부 : {torch.cuda.get_device_name()}')
    print(f'사용 가능 GPU 갯수 :  {torch.cuda.device_count()}')
    '''
    numpy version : 1.23.4
    tensorflow version : 2.9.1
    torch version : 1.7.1+cu110
    sklearn version : 1.2.0
    2022-12-22 12:19:27.314754: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-12-22 12:19:27.831345: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /device:GPU:0 with 5954 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2080, pci bus id: 0000:01:00.0, compute capability: 7.5
    이 PC에 설치된 디바이스 상세보기 : [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 14913580382836303060
    xla_global_id: -1
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 6243221504
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 5022873307967363584
    physical_device_desc: "device: 0, name: NVIDIA GeForce RTX 2080, pci bus id: 0000:01:00.0, compute capability: 7.5"
    xla_global_id: 416903419
    ]
    CUDA 프로그래밍 가능여부 :  True
    CUDA 프로그래밍 가능여부 : NVIDIA GeForce RTX 2080
    사용 가능 GPU 갯수 :  1
    '''