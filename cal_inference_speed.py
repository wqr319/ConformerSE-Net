import os
from time import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pysepm
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
from torch.autograd.profiler import ProfilerActivity, profile
from tqdm import tqdm

from build import LightningNet
from main import load_cfg

p = psutil.Process()
p.cpu_affinity([0])

model = LightningNet.load_from_checkpoint('log/DNS/O/epoch=360.ckpt').cpu()
duration = 1
start_time = time()
with torch.no_grad():
    for i in range(100):
        dummy_input = torch.ones((1,16000 * duration)).cpu()
        output = model(dummy_input)
        end_time = time()
        if (i+1)%5==0:
            print('warmup {} using time {:.2f}ms'.format(i+1, (end_time-start_time)*1000/(i+1)/duration))