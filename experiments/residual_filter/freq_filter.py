# freq_filter.py
import numpy as np


def _check_cutoff(cutoff):
    cutoff = float(cutoff)
    if cutoff <= 0.0 or cutoff > 1.0:
        raise ValueError('fft_cutoff must be in (0, 1], got {}'.format(cutoff))
    return cutoff


def _build_lowpass_mask(h, w, cutoff):
    """
    构造中心低通圆形掩码。
    cutoff 是归一化半径，范围 (0, 1]。
    """
    cutoff = _check_cutoff(cutoff)

    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    rr = np.sqrt(xx * xx + yy * yy)

    mask = (rr <= cutoff).astype(np.float32)
    return mask


def _fft_lowpass_single_channel(channel, cutoff):
    """
    对单通道图像做 FFT 低通滤波。
    输入:
        channel: H x W, uint8/float 均可
    输出:
        filtered: H x W, uint8
    """
    channel = channel.astype(np.float32)

    h, w = channel.shape
    mask = _build_lowpass_mask(h, w, cutoff)

    freq = np.fft.fft2(channel)
    freq_shift = np.fft.fftshift(freq)

    filtered_shift = freq_shift * mask
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    filtered = np.real(filtered)

    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered


def fft_lowpass_filter(img, cutoff=0.25):
    """
    对 HxW 或 HxWxC 图像做 FFT 低通滤波。
    默认按通道分别处理。
    """
    if img.ndim == 2:
        return _fft_lowpass_single_channel(img, cutoff)

    if img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_fft_lowpass_single_channel(img[:, :, c], cutoff))
        return np.stack(channels, axis=2)

    raise ValueError('Unsupported img ndim: {}'.format(img.ndim))


def apply_freq_filter(img, filter_type='fft_lowpass', fft_cutoff=0.25):
    """
    统一入口。
    当前第一版仅支持 fft_lowpass。
    """
    if filter_type is None or filter_type == 'none':
        return img

    if filter_type == 'fft_lowpass':
        return fft_lowpass_filter(img, cutoff=fft_cutoff)

    raise ValueError('Unsupported freq filter type: {}'.format(filter_type))