import os
import torch
import numpy as np
import scipy.io.wavfile as wavfile
def rm_rmaudio_audio(path,seconds):
    for root1, dirs, filess in sorted(os.walk(path)):
        if os.path.split(root1)[-1] in ["train", "test", "valid"]:
            for files in filess:
                #print(os.path.join(root1,files))
                sr, x = wavfile.read(os.path.join(root1,files))  # 读取文件
                if len(x) <= sr * seconds:
                    # print("os.path.join(dirs,files)",os.path.join(root1,files))
                    os.system("rm -rf "+os.path.join(root1,files))
                    print(os.path.join(root1,files))
if __name__ == '__main__':
    rm_rmaudio_audio("/home/tcd/NLP/language_recognition/tools/童昌东手机录音/result",2)
