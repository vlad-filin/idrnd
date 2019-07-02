import warnings
import numpy as np
import torch
import torch.nn as nn

class DftSpectrogram(nn.Module):
    def __init__(self,
                 length=200,
                 shift=150,
                 nfft=256,
                 mode="abs",
                 normalize_feature=False,
                 normalize_signal=False,
                 top=0,
                 bottom=0,
                 trainable=False,
                 window=None):
        """
                Requirements
                ------------
                input shape must meet the conditions: mod((input.shape[0] - length), shift) == 0
                nfft >= length

                Parameters
                ------------
                :param length: Length of each segment.
                :param shift: Number of points to step for segments
                :param nfft: number of dft points, if None => nfft === length
                :param normalize_feature: zero mean, and unit std for 2d features, doesn't work for "complex" mode
                :param normalize_spectrogram: zero mean, and unit std for 1d input signal
                :param mode: "abs" - amplitude spectrum; "real" - only real part, "imag" - only imag part,
                "complex" - concatenate real and imag part, "log" - log10 of magnitude spectrogram
                :param kwargs: unuse

                Input
                -----
                input mut have shape: [n_batch, signal_length, 1]

                Returns
                -------
                A model that has output shape of
                (None, nfft / 2, n_time) (if type == "abs" || "real" || "imag") or
                (None, nfft / 2, n_frame, 2) (if type = "abs" & `img_dim_ordering() == 'tf').
                (None, 1, nfft / 2, n_frame) (if type = "abs" & `img_dim_ordering() == 'th').
                (None, nfft / 2, n_frame, 2) (if type = "complex" & `img_dim_ordering() == 'tf').
                (None, 2, nfft / 2, n_frame) (if type = "complex" & `img_dim_ordering() == 'th').

                number of time point of output spectrogram: n_time = (input.shape[0] - length) / shift + 1
        """
        super(DftSpectrogram, self).__init__()
        assert mode in ["abs", "complex", "real", "imag", "log", "phase"], NotImplementedError

        self.trainable = trainable
        self.length = length
        self.shift = shift
        self.mode = mode
        self.normalize_feature = normalize_feature
        self.top = top
        self.bottom = bottom
        self.normalize_signal = normalize_signal
        self.window = window
        if nfft is None:
            self.nfft = length
        else:
            self.nfft = nfft

        assert self.nfft >= length
        # code from build
        length = self.length

        #assert len(input_shape) >= 2
        assert nfft >= length

        self.__real_kernel = np.asarray([np.cos(2 * np.pi * np.arange(0, nfft) * n / nfft)
                                                  for n in range(nfft)],  dtype=np.float32)
        self.__imag_kernel = -np.asarray([np.sin(2 * np.pi * np.arange(0, nfft) * n/ nfft)
                                                  for n in range(nfft)],  dtype=np.float32)

        self.__real_kernel = self.__real_kernel[:, np.newaxis, :]
        self.__imag_kernel = self.__imag_kernel[:, np.newaxis, :]

        if self.length < self.nfft:
            self.__real_kernel[length - nfft:, :, :] = 0.0
            self.__imag_kernel[length - nfft:, :, :] = 0.0
        self.real_conv = nn.Conv1d(in_channels=1, out_channels=nfft, kernel_size=nfft,
                                   stride=self.shift, bias=False)
        self.imag_conv = nn.Conv1d(in_channels=1, out_channels=nfft, kernel_size=nfft,
                                   stride=self.shift, bias=False)
        self.real_conv.weight = torch.nn.Parameter(torch.from_numpy(self.__real_kernel))
        self.imag_conv.weight = torch.nn.Parameter(torch.from_numpy(self.__imag_kernel))
        if self.trainable is False:
            for m in [self.real_conv, self.imag_conv]:
                for param in m.parameters():
                    param.requires_grad = False
        assert self.length >= self.nfft, "need to add padding to work"
        self.epsilon = 1e-7


    def forward(self, inputs):
        if self.normalize_signal:
            inputs = (inputs - torch.mean(inputs, dim=(1,2), keepdim=True)) /\
                     (inputs.std(dim=(1,2)) + self.epsilon)

        real_part = []
        imag_part = []
        for n in range(inputs.shape[1]):
            real_part.append(self.real_conv(inputs[:, n, :][:, None, :]))
            imag_part.append(self.imag_conv(inputs[:, n, :][:, None, :]))


        real_part = torch.stack(real_part, dim=-1)
        imag_part = torch.stack(imag_part, dim=-1)
        if self.mode == "abs":
            fft = torch.sqrt(real_part ** 2 + imag_part ** 2)
        if self.mode == "phase":
            fft = tf.atan(real_part / imag_part)
        elif self.mode == "real":
            fft = real_part
        elif self.mode == "imag":
            fft = imag_part
        elif self.mode == "complex":
            fft = torch.concatenate((real_part, imag_part), axis=-1)
        elif self.mode == "log":
            fft = torch.sqrt(real_part ** 2 + imag_part ** 2 + self.epsilon)
            fft = torch.log(fft) / np.log(10)

        fft = fft.permute((0, 3, 1, 2,))[:, :, :self.nfft // 2, :]
        if self.normalize_feature:
            if self.mode == "complex":
                warnings.warn("spectrum normalization will not applied with mode == \"complex\"")
            else:
                fft = (fft - torch.mean(fft, dim=2, keepdim=True)) / (
                            fft.std(dim=2, keepdim=True, unbiased=False) + self.epsilon)

        # fft = fft[:, self.bottom:-1 * self.top, :, :]

        return fft
