import random
import os
import wave
import os
import sys
import fnmatch
import ffmpeg
import subprocess
from pydub.audio_segment import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np
import webrtcvad
from optparse import OptionParser
import uuid
from multiprocessing import Pool
import os, time, random
"""
from multiprocessing import Pool
import os, time, random
p = Pool(20)

def asdasda(path,a,type1,i):
    os.system("ffmpeg -i "+os.path.join(path,a) +" -vn -ar 16000 -ac 1 -ab 192 -f wav train/"+type1+"_acoustic_%05d.wav"%(i))  

sum1=3145
path="英文1"
lis1=os.listdir(path)
lis2=random.sample(lis1,sum1)
for i in range(len(lis2)):
    p.apply_async(asdasda, args=(path,lis2[i],"en",i))
p.close()
p.join()
print("!!!!")
path="中文1"
#p1 = Pool(20)
lis1=os.listdir(path)
lis2=random.sample(lis1,sum1)
for i in range(len(lis2)):
    #p1.apply_async(asdasda, args=(path,lis2[i],"zh",i))
    os.system("ffmpeg -i "+os.path.join(path,lis2[i]) +" -vn -ar 16000 -ac 1 -ab 192 -f wav train/zh_acoustic_%05d.wav"%(i))  
#p1.close()
#p1.join()

"""

def asdasda(path,a,type1,i):
    os.system("ffmpeg -i "+os.path.join(path,a) +" -vn -ar 16000 -ac 1 -ab 192 -f wav train/"+type1+"_acoustic_%05d.wav"%(i))  
lisen=[]
liszh=[]
PATHEN=[
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/英0",#850
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/英1",#50
"/home/tcd/Downloads/语种识别数据集/喜马拉雅02/英2",#50
"/home/tcd/NLP/language_recognition/done/百度AI语音en/",#100
#"/home/tcd/NLP/language_recognition/done/百度AI语音en/0/0",#100
#"/home/tcd/NLP/language_recognition/done/百度AI语音en/1",#100
#"/home/tcd/NLP/language_recognition/done/百度AI语音en/3",#100
#"/home/tcd/NLP/language_recognition/done/百度AI语音en/4",#100
#"/home/tcd/NLP/language_recognition/done/百度AI语音en/5",#100
#"/home/tcd/NLP/language_recognition/done/百度AI语音en/103",#100
#"/home/tcd/NLP/language_recognition/done/百度AI语音en/106/106",#100
#"/home/tcd/NLP/language_recognition/done/百度AI语音en/110/110",#100
#"/home/tcd/NLP/language_recognition/done/百度AI语音en/111/111",#100
"/home/tcd/NLP/language_recognition/done/ximalayaolden",#50
"/home/tcd/NLP/language_recognition/done/ximalayanewen",#50
"/home/tcd/NLP/language_recognition/done/baidu_translate_audio_en"#50
]
for path in PATHEN:
    for root1, dirs, filess in sorted(os.walk(path)):
        for files in  filess:
            if files.split(".")[-1]=="wav":
                lisen.append(root1+"/"+files)

PATHZH=[
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/中0",#400
"/home/tcd/Downloads/语种识别数据集/language_recognize/data/中1",#200
"/home/tcd/Downloads/语种识别数据集/喜马拉雅02/中2",#300
"/home/tcd/NLP/language_recognition/done/百度翻译中文音频",# 50
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/0", #100
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/1", #100
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/3", #100
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/4", #100
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/5", #100
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/103", #100
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/106", #100
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/110", #100
"/home/tcd/NLP/language_recognition/done/百度AI语音zh/111", #100
"/home/tcd/NLP/language_recognition/done/ximalayaoldzh",#50
"/home/tcd/NLP/language_recognition/done/ximalayanewzh",#50
"/home/tcd/NLP/language_recognition/done/baidu_translate_audio_zh"#50
]
for path in PATHZH:
    for root1, dirs, filess in sorted(os.walk(path)):
        for files in  filess:
            if files.split(".")[-1]=="wav":
                liszh.append(root1+"/"+files)
print("en",len(lisen),"zh",len(liszh))
sum1=len(lisen) if len(lisen)<len(liszh) else  len(liszh) 

lis2=random.sample(liszh,sum1)
if  os.path.exists("train"):
    os.system("rm -rf train")
    os.mkdir("train")
else:
    os.mkdir("train")

if  os.path.exists("valid"):
    os.system("rm -rf valid")
    os.mkdir("valid")
else:
    os.mkdir("valid")


for i in range(len(lis2)):
    if i%10==0:
        os.system("ffmpeg -i "+lis2[i] +" -vn -ar 16000 -ac 1 -ab 192 -f wav valid/zh_acoustic_%06d.wav"%(i))  
    else:
        os.system("ffmpeg -i "+lis2[i] +" -vn -ar 16000 -ac 1 -ab 192 -f wav train/zh_acoustic_%06d.wav"%(i))  
    #p1.apply_async(asdasda, args=(path,lis2[i],"zh",i))
#p1.close()
#p1.join()
lis2=random.sample(lisen,sum1)
liswaven=[]
for i in range(len(lis2)):
    os.system("ffmpeg -i "+lis2[i] +" -vn -ar 16000 -ac 1 -ab 192 -f wav en/en_acoustic_%06d.wav"%(i))
    liswaven.append("/home/tcd/Downloads/语种识别数据集/language_recognize/data/"+"en/en_acoustic_%06d.wav"%(i))

for i in range(len(liswaven)):
    if i%10==0:
        #sound = AudioSegment.from_mp3(liswaven[i])
        #word = sound[1* 1000:]
        #word.export("valid/en_acoustic_%06d.wav"%(i), format="wav")
        os.system("ffmpeg -i "+liswaven[i] +" -vn -ar 16000 -ac 1 -ab 192 -f wav valid/en_acoustic_%06d.wav"%(i))
    else:
        #sound = AudioSegment.from_mp3(liswaven[i])
        #word = sound[1* 1000:]
        #word.export("train/en_acoustic_%06d.wav"%(i), format="wav")
        os.system("ffmpeg -i "+liswaven[i] +" -vn -ar 16000 -ac 1 -ab 192 -f wav train/en_acoustic_%06d.wav"%(i))

    #p.apply_async(asdasda, args=(path,lis2[i],"en",i))
#p.close()
#p.join()
