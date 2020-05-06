import librosa as lr
from scipy.io import wavfile
from fastai.vision import *
from fastai.torch_core import *
import datetime

f_min = 0
f_max = None
ref = 'max'
top_db = 80.0
norm_db = True

n_fft = 1024  # fft的输出格式 [513 x n_frames] n_fft/2 + 1 =513
n_hop = 256  # 75% 帧之间的重叠
n_mels = 40  # 通过mel频率标度将513维压缩到40维
sample_rate = 16000  # 采样频率
batch_size = 1  # 1  # batch_size

amin = 1e-7
device = None
power = 2
constant = 10.0 if power == 2 else 20.0
top_db = abs(top_db) if top_db else top_db
normalized = norm_db
add_channel_dim = True


def steps(x):
    # window = to_device(torch.hann_window(n_fft), device)  ####???这里转成GPU数据了
    # print("window GPU",window)
    window = torch.hann_window(n_fft)
    # print("window CPU",window)
    X = torch.stft(x, n_fft=n_fft, hop_length=n_hop, win_length=n_fft, window=window, onesided=True, center=True,
                   pad_mode='constant', normalized=True)
    X.pow_(2.0)
    power = X[:, :, :, 0] + X[:, :, :, 1]
    # print("power.shape", power.shape)
    mel_fb = lr.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
                            fmin=f_min, fmax=f_max).astype(np.float32)
    # mel_filterbank = to_device(torch.from_numpy(mel_fb), device)  # GPU
    # print("mel_filterbank GPU",mel_filterbank)
    mel_filterbank = torch.from_numpy(mel_fb)  # GPU
    # print("mel_filterbank CPU",mel_filterbank)
    spec_m = mel_filterbank @ power
    ref_value = spec_m.contiguous().view(batch_size, -1).max(dim=-1)[0]  # copy新的内存空间储存信息
    ref_value.unsqueeze_(1).unsqueeze_(1)  # [64,1,1]
    spec_db = spec_m.clamp_min(amin).log10_().mul_(constant)  # [64.40,126] #修改的x数据
    spec_db.sub_(ref_value.clamp_min_(amin).log10_().mul_(10.0))
    max_spec = spec_db.view(batch_size, -1).max(dim=-1)[0]
    max_spec.unsqueeze_(1).unsqueeze_(1)
    spec_db = torch.max(spec_db, max_spec - top_db)
    spec_db.add_(top_db).div_(top_db)
    return spec_db


def test(audio_path):
    seconds = 1
    sr, x = wavfile.read(audio_path)  # 读取文件
    # xs = torch.from_numpy(x[sr:sr * (seconds+1)].astype(np.float32, copy=False))
    if len(x) >= sr * seconds:
        xs = torch.from_numpy(x[:sr * seconds].astype(np.float32, copy=False))#前两秒
        if x.dtype == np.int16:
            xs.div_(32767)
        elif x.dtype != np.float32:
            raise OSError('Encountered unexpected dtype: {}'.format(x.dtype))
        # 得到输入的音频 得到 数据 xs
        # print("xs",xs)
        # print("xs.shape",xs.shape)
        xs = xs.unsqueeze(0)
        # print("xs.shape",xs.shape)

        xs = steps(xs)
        if add_channel_dim:
            xs.unsqueeze_(1)
        path = "app"
        learn = load_learner(path, 'models/export.pkl')
        a, b, losses = learn.predict(xs[0])
        # print("a", a)
        # print("b", b)
        # print("losses", losses)
        return a
    else:return ""

if __name__ == '__main__':
    totalsum1 = 0
    totalcorrect = 0
    totalerror = 0
    # path = "/home/tcd/NLP/language_recognition/data/tcd_phone_zh"
    """
    PATH = [#"/home/tcd/Downloads/语种识别数据集/language_recognize/data/test",
            "/home/tcd/NLP/language_recognition/done/机器人录制的zhwav",
            "/home/tcd/NLP/language_recognition/done/机器人录制的enwav",
            "/home/tcd/NLP/language_recognition/done/zh0906童昌东机器录制wav",
            "/home/tcd/NLP/language_recognition/done/zh0906机器录制彭瑾wav",
            "/home/tcd/NLP/language_recognition/done/wangjie_机器录制音频_zh",
            "/home/tcd/NLP/language_recognition/done/tcd_机器录制音频_zh",
            "/home/tcd/NLP/language_recognition/done/tcd_机器录制音频_en",
            "/home/tcd/NLP/language_recognition/done/en0903童昌东机器录制wav",
            "/home/tcd/Downloads/语种识别数据集/language_recognize/data/test"]
    """
    PATH = ["/home/tcd/NLP/language_recognition/done/0918wav/梅森中文0918wav",
            "/home/tcd/NLP/language_recognition/done/0918wav/梅森英文0918wav",
            "/home/tcd/NLP/language_recognition/done/0918wav/郭子威中文0918wav",
            "/home/tcd/NLP/language_recognition/done/0918wav/郭子威英文0918wav",
            "/home/tcd/NLP/language_recognition/done/0918wav/谢雅馨中文0918wav",
            "/home/tcd/NLP/language_recognition/done/0918wav/谢雅馨英文0918wav",
            "/home/tcd/Downloads/语种识别数据集/language_recognize/data/test"]
    for path in PATH:
        correct = 0
        error = 0
        sumzh = 0
        sumen = 0
        sumyue = 0
        sumru = 0
        sum1 = 0
        for files in sorted(os.listdir(path)):
            # print(os.path.join(path,files))
            tic = datetime.datetime.now()
            a = test(os.path.join(path, files))
            toc = datetime.datetime.now()
            # print(os.path.join(path, files))
            # print("Answer a question in %s seconds" % (toc - tic))
            if a!="":
                sum1 += 1
                if str(a) == "zh" and str(a) in os.path.join(path, files):
                    sumzh += 1
                elif str(a) == "en" and str(a) in os.path.join(path, files):
                    sumen += 1
                elif str(a) == "yue" and str(a) in os.path.join(path, files):
                    sumyue += 1
                elif str(a) == "ru" and str(a) in os.path.join(path, files):
                    sumru += 1
                if str(a) not in os.path.join(path, files):
                    error += 1
                    print("error", os.path.join(path, files))
                else:
                    # print("correct",os.path.join(path, files))
                    correct += 1
        totalerror+=error
        totalcorrect+=correct
        totalsum1+=sum1
        print("sum1",sum1)
        print("error", error)
        print("correct", correct)
        print("sumzh", sumzh)
        print("sumyue", sumyue)
        print("sumen", sumen)
        print("sumru", sumru)
    print("totalsum1",totalsum1)
    print("totalerror",totalerror)
    print("totalcorrect",totalcorrect)
