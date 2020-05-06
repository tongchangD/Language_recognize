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


def get_second_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    '''
    音频切片，获取部分音频 单位是秒级别
    :param main_wav_path: 原音频文件路径
    :param start_time:  截取的开始时间
    :param end_time:  截取的结束时间
    :param part_wav_path:  截取后的音频路径
    :return:
    '''
    start_time = int(start_time) * 1000
    end_time = int(end_time) * 1000

    sound = AudioSegment.from_mp3(main_wav_path)
    word = sound[start_time:end_time]

    word.export(part_wav_path, format="wav")

def reference(PATH,path,files,seconds):
    print(os.path.join(os.path.join(PATH, path), files))
    f = wave.open(r"" + os.path.join(os.path.join(PATH, path), files), "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    f.close()
    for x in range(len(range(0, int(nframes / 16000) - 10, seconds)) - 1):
        if x <= 5:
            if not os.path.exists(os.path.join(PATH,"result")):
                os.makedirs(os.path.join(PATH,"result"))
            get_second_part_wav(os.path.join(os.path.join(PATH, path), files), 10 + x * seconds, 10 + (x + 1) * seconds,
                            os.path.join(PATH,"result") + "/"+path+"_acoustic_%s.wav"%str(uuid.uuid1()).split("-")[0])
def cut_audio(PATH,seconds):
    sum1=0
    for path in os.listdir(PATH):
        for files in os.listdir(os.path.join(PATH,path)):
            if files.split(".")[1]=="m4a":
                if os.path.exists(os.path.join(os.path.join(PATH,path),files).replace("m4a","wav")):
                    os.remove(os.path.join(os.path.join(PATH,path),files).replace("m4a","wav"))
                os.system("ffmpeg -i "+os.path.join(os.path.join(PATH,path),files) +" -vn -ar 16000 -ac 1 -ab 192 -f wav "+os.path.join(os.path.join(PATH,path),files).replace("m4a","wav"))
                reference(PATH,path,files.replace("m4a","wav"),seconds)#剪切

def change_Multilayer_audio_type(PATH,sum1,srctype,restype):
    for path in os.listdir(PATH):
        for files in os.listdir(os.path.join(PATH,path)):
            if files.split(".")[1]==srctype:
                #/home/tcd/NLP/language_recognition/tools/语种识别/机器人录音文件/recording1401778814.3gpp
                #/home/tcd/NLP/language_recognition/tools/语种识别/机器人录音文件/recording1401778814.3gpp
                if os.path.exists(os.path.join(os.path.join(PATH,path),files).replace(srctype,restype)):
                    os.remove(os.path.join(os.path.join(PATH,path),files).replace(srctype,restype))
                sum1+=1
                if not os.path.exists(os.path.join(PATH,"result")):
                    os.makedirs(os.path.join(PATH,"result"))
                os.system("ffmpeg -i "+os.path.join(os.path.join(PATH,path),files) +
                          " -vn -ar 16000 -ac 1 -ab 192 -f wav "+os.path.join(os.path.join(PATH,"result"),"acoustic_11%04d.wav"%sum1))
    print("sum1",sum1)
def ffmpegs(PATH,files,sum1,restype):
    os.system("ffmpeg -i "+os.path.join(PATH,files) +
                      " -vn -ar 16000 -ac 1 -ab 192 -f wav "+os.path.join(os.path.join(os.path.split(PATH)[0],"result"),"zh_acoustic_%05d.%s"%(sum1,restype)))    
def change_Single_layer_audio_type(PATH,sum1,srctype,restype):
    p = Pool(20)
    for files in os.listdir(PATH):
        if files.split(".")[1]==srctype:
            sum1+=1
            if not os.path.exists(os.path.join(os.path.split(PATH)[0],"result")):
                os.makedirs(os.path.join(os.path.split(PATH)[0],"result"))
            
            p.apply_async(ffmpegs, args=(PATH,files,sum1,restype))
    p.close()
    p.join()
            #os.system("ffmpeg -i "+os.path.join(PATH,files) +
            #          " -vn -ar 16000 -ac 1 -ab 192 -f wav "+os.path.join(os.path.join(os.path.split(PATH)[0],"result"),"zh_acoustic_%05d.%s"%(sum1,restype)))
    print("sum1",sum1)
if __name__ == '__main__':
    sum1=1190
    # path="童昌东手机录音"#文件夹文件夹格式["zh","en","ru"]
    seconds=5 #单位秒
    # srctype="m4a"
    # restype="wav"
    # change_Multilayer_audio_type(path,sum1,srctype,restype)#改变音频文件格式["zh"]
    path="/home/tcd/Downloads/语种识别数据集/za1"
    srctype="m4a"
    restype="wav"
    #change_Single_layer_audio_type(path,sum1,srctype,restype)#改变音频文件格式["zh"]
    change_Multilayer_audio_type(path,sum1,srctype,restype)#改变音频文件格式["zh"]
    
    cut_audio(path,seconds)#剪切大文件音频

