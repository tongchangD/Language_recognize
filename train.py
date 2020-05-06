import fastai
#fastai.__version__

from fastai.metrics import accuracy
from fastai.torch_core import *
from fastai_audio import *
import fastai_audio.rm_audio as rm_audio

from fastai.vision import models, ClassificationInterpretation
DATA = Path('data')
rm_audio.rm_rmaudio_audio(DATA,1)

n_fft = 1024  # fft的输出格式 [513 x n_frames] n_fft/2 + 1 =513
n_hop = 256  # 75% 帧之间的重叠
n_mels = 40  # 通过mel频率标度将513维压缩到40维
sample_rate = 16000  # 采样频率
batch_size = 64  # 64  # batch_size

# 得到频率批处理转换
tfms = get_frequency_batch_transforms(n_fft=n_fft, n_hop=n_hop,n_mels=n_mels, sample_rate=sample_rate)  # 返回 torch.Size([batch_size, 1, 40, 126]) # ys [batch_size个标签]


#pattern = r'(\w+)_\w+_\d+.wav$'
pattern = r'(\w+)_\w+_.+.wav$'
data = (AudioItemList
        .from_folder(DATA)  # 获取文件相对路径 所有文件，包括模型文件
        .filter_by_func(lambda fn: 'acoustic' in fn.name) # 只读名称中存在acoustic的 匿名函数
        .split_by_folder()  # 切割文件 train 和 test
        .label_from_re(pattern) # data_block 文件内方法 正则做数据标签
        .databunch(bs=batch_size, tfms=tfms)) #
print("data",data)
xs, ys = data.one_batch()
print(xs.shape, ys.shape)
print("data.c",data.c, "data.classes",data.classes)
learn = create_cnn(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(30)
learn.save('stage-1')
# print("learn",learn)
#learn = create_cnn(data, models.resnet50, metrics=accuracy)
#learn.fit_one_cycle(6)

"""
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(4, 4))
print(interp.confusion_matrix())
"""
print("learn.path",learn.path)

print("learn.data.path",learn.data.path)

#learn.path = Path('models')
#learn.data.path = Path('models')
# print("learn",learn)
# learn.fit(0.01, 3)# 第一次训练最后一层，试探性
"""
learn.fit(0.05, 5, cycle_len=1, cycle_mult=2) # 发现第一次训练有效果，在进一步训练
learn.precompute = False # 开启全部层都可以训练，对每层进行微训练
learn.fit([1e-4,1e-3,1e-2], 4, cycle_len=1) # 初步训练
learn.unfreeze()
learn.fit([1e-3,1e-3, 1e-2], 5, cycle_len=1, cycle_mult=2) # 发现上一次有效果，在进行深度训练
"""


learn.export()
learn.save("da")
