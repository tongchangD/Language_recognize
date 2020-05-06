import random
import os
import wave
import os
import sys
import fnmatch
import ffmpeg
import scipy.io.wavfile as wavfile
import subprocess
from pydub.audio_segment import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np
import webrtcvad
from optparse import OptionParser
import uuid
from multiprocessing import Pool
import os, time, random

def asdasda(path,a,type1,i):
    os.system("ffmpeg -i "+os.path.join(path,a) +" -vn -ar 16000 -ac 1 -ab 192 -f wav train/"+type1+"_acoustic_%05d.wav"%(i))  
seconds=2  # 2秒
lisresen=[]
lisreszh=[]
"""
PATHEN={
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/英0":850,
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/英1":50,
"/home/tcd/Downloads/语种识别数据集/喜马拉雅02/英2":50,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/0/0":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/1":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/3":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/4":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/5":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/103":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/106/106":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/110/110":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/111/111":100,
"/home/tcd/NLP/language_recognition/done/ximalayaolden":50,
"/home/tcd/NLP/language_recognition/done/ximalayanewen":50,
"/home/tcd/NLP/language_recognition/done/baidu_translate_audio_en":50
}
PATHZH={
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/中0":400,
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/中1":200,
"/home/tcd/Downloads/语种识别数据集/喜马拉雅02/中2":300,
"/home/tcd/NLP/language_recognition/done/百度翻译中文音频": 50,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/0":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/1":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/3":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/4":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/5":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/103":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/106":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/110":100,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/111":100,
"/home/tcd/NLP/language_recognition/done/ximalayaoldzh":50,
"/home/tcd/NLP/language_recognition/done/ximalayanewzh":50,
"/home/tcd/NLP/language_recognition/done/baidu_translate_audio_zh":50
}
"""

PATHEN={
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/英0":1730,# 
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/英1":50,
"/home/tcd/Downloads/语种识别数据集/喜马拉雅02/英2":50,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/0/0":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/1":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/3":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/4":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/5":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/103":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/106/106":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/110/110":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音en/111/111":10,
"/home/tcd/NLP/language_recognition/done/ximalayaolden":40,
"/home/tcd/NLP/language_recognition/done/ximalayanewen":30,
"/home/tcd/NLP/language_recognition/done/baidu_translate_audio_en":10
}
PATHZH={
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/中0":700,
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/中1":700,
"/home/tcd/Downloads/语种识别数据集/喜马拉雅02/中2":300,
"/home/tcd/NLP/language_recognition/done/百度翻译中文音频": 50,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/0":12,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/1":12,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/3":12,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/4":12,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/5":12,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/103":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/106":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/110":10,
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/111":10,
"/home/tcd/NLP/language_recognition/done/ximalayaoldzh":50,
"/home/tcd/NLP/language_recognition/done/ximalayanewzh":50,
"/home/tcd/NLP/language_recognition/done/baidu_translate_audio_zh":50
}

for path in PATHEN.keys():
    lisen=[]
    print ("path",path)
    for root1, dirs, filess in sorted(os.walk(path)):
        for files in  filess:
            if files.split(".")[-1]=="wav":
                sr, x = wavfile.read(os.path.join(root1,files))  # 读取文件
                if len(x) >= sr * seconds:
                    lisen.append(os.path.join(root1,files))
    print("len(lisen)",len(lisen),"PATHEN[path]",PATHEN[path])
    lisresen+=random.sample(lisen,len(lisen) if len(lisen)<PATHEN[path] else  PATHEN[path])
for path in PATHZH.keys():
    liszh=[]
    print("path",path)
    for root1, dirs, filess in sorted(os.walk(path)):
        for files in  filess:
            if files.split(".")[-1]=="wav":
                try:
                    sr, x = wavfile.read(os.path.join(root1,files))  # 读取文件
                    if len(x) >= sr * seconds:
                        liszh.append(os.path.join(root1,files))
                except:
                    print("error",os.path.join(root1,files))
    print("len(liszh)",len(liszh),"PATHZH[path]",PATHZH[path])
    lisreszh+=random.sample(liszh,PATHZH[path])
print("en",len(lisresen),"zh",len(lisreszh))

if  os.path.exists("中文录制音频"):
    os.system("rm -rf 中文录制音频")
    os.mkdir("中文录制音频")
else:
    os.mkdir("中文录制音频")

if  os.path.exists("英文录制音频"):
    os.system("rm -rf 英文录制音频")
    os.mkdir("英文录制音频")
else:
    os.mkdir("英文录制音频")
a=input(">>开始执行>> YES or NO\n>>")
f=open("select_luzhi_audio","a+")
if a.lower()!="n":
    for i in range(len(lisresen)):
        os.system("ffmpeg -i "+lisresen[i] +" -vn -ar 16000 -ac 1 -ab 192k -f wav 英文录制音频/en_acoustic_%06d.wav"%(i))  
        f.write(lisresen[i]+"\t英文录制音频/en_acoustic_%06d.wav\n"%(i))
    for i in range(len(lisreszh)):
        os.system("ffmpeg -i "+lisreszh[i] +" -vn -ar 16000 -ac 1 -ab 192k -f wav 中文录制音频/zh_acoustic_%06d.wav"%(i))
        f.write(lisreszh[i] + "\t中文录制音频/zh_acoustic_%06d.wav\n"%(i))
f.close()

"""
for i in range(len(liswaven)):
    if i%10==0:
        os.system("ffmpeg -i "+liswaven[i] +" -vn -ar 16000 -ac 1 -ab 192k -f wav valid/en_acoustic_%06d.wav"%(i))
    else:
        os.system("ffmpeg -i "+lis2[i] +" -vn -ar 16000 -ac 1 -ab 192k -f wav train/en_acoustic_%06d.wav"%(i))
"""
