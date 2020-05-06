from fastai.torch_core import *
from scipy.io import wavfile
from IPython.display import display, Audio

__all__ = ['AudioClip', 'open_audio']


class AudioClip(ItemBase):
    def __init__(self, signal, sample_rate):
        # print("signal",signal)
        self.data = signal
        self.sample_rate = sample_rate

    def __str__(self):
        return '(duration={}s, sample_rate={:.1f}KHz)'.format(
            self.duration, self.sample_rate/1000)

    def clone(self):
        return self.__class__(self.data.clone(), self.sample_rate)

    def apply_tfms(self, tfms, **kwargs):

        print("self.clone()",self.clone())
        x = self.clone()
        print("x"*60,x)
        for tfm in tfms:
            x.data = tfm(x.data)
        return x

    @property
    def num_samples(self):
        return len(self.data)

    @property
    def duration(self):
        return self.num_samples / self.sample_rate # 数据长度 除以 采样率 等于 时间长度

    def show(self, ax=None, figsize=(5, 1), player=True, title=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title(title)
        timesteps = np.arange(len(self.data)) / self.sample_rate
        ax.plot(timesteps, self.data)
        ax.set_xlabel('Time (s)')
        plt.show()
        if player:
            # unable to display an IPython 'Audio' player in plt axes
            display(Audio(self.data, rate=self.sample_rate))


def open_audio(fn):
    # 音频长度 单位：秒
    #tcd
    seconds=1
    sr, x = wavfile.read(fn)#读取文件

    if len(x)>=sr*seconds:
        t = torch.from_numpy(x[:sr*seconds].astype(np.float32, copy=False))
        if x.dtype == np.int16:
            t.div_(32767)
        elif x.dtype != np.float32:
            raise OSError('Encountered unexpected dtype: {}'.format(x.dtype))
        # print("t",len(t),"sr",sr)
        return AudioClip(t, sr) # 采样音频数据  采样率采样频率 设置的16000


    # sr, x = wavfile.read(fn)#读取文件
    # t = torch.from_numpy(x.astype(np.float32, copy=False))
    # if x.dtype == np.int16:
    #     t.div_(32767)
    # elif x.dtype != np.float32:
    #     raise OSError('Encountered unexpected dtype: {}'.format(x.dtype))
    # # print("t",len(t),"sr",sr)
    # return AudioClip(t, sr)  # 采样音频数据  采样率采样频率 设置的16000


if __name__ == '__main__':
    pass
