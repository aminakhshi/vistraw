import numpy as np
import math
from scipy import signal

"""
Different Filter designs to filter a data series with FIR coefficients.
Please cite github for bug reports and usage
Examples is given from scipy docs
@author: Amin Akhshi
Email: amin.akhshi@gmail.com
Github: https://aminakhshi.github.io
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
rng = np.random.default_rng()


def roll(data, shift):
    """shift vector
    """
    new_arr = np.zeros(len(data), dtype=data.dtype)

    if shift < 0:
        shift = shift - len(data) * (shift // len(data))
    if shift == 0:
        return
    new_arr[0:shift] = data[len(data)-shift: len(data)]
    new_arr[shift:len(data)] = data[0:len(data)-shift]
    return new_arr

def rolling_window(data, window_len, noverlap=1, padded=False, axis=-1, copy=True):
    """
    Calculate a rolling window over a data
    :param data: numpy array. The array to be slided over.
    :param window_len: int. The rolling window size
    :param noverlap: int. The rolling window stepsize. Defaults to 1.
    :param padded:
    :param axis: int. The axis to roll over. Defaults to the last axis.
    :param copy: bool. Return strided array as copy to avoid sideffects when manipulating the
        output array.
    :return:
    numpy array
        A matrix where row in last dimension consists of one instance
        of the rolling window.
    See Also
    --------
    pieces : Calculate number of pieces available by rolling
    """
    data = data if isinstance(data, np.ndarray) else np.array(data)
    assert axis < data.ndim, "Array dimension is less than axis"
    assert noverlap >= 1, "Minimum overlap cannot be less than 1"
    assert window_len <= data.shape[axis], "Window size cannot exceed the axis length"
    
    arr_shape = list(data.shape)
    arr_shape[axis] = np.floor(data.shape[axis] / noverlap - window_len / noverlap + 1).astype(int)
    arr_shape.append(window_len)

    strides = list(data.strides)
    strides[axis] *= noverlap
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=arr_shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided

def fir_filt(data, coeff):
    """
    Filter the data series with a set of FIR coefficients
    :param data: a numpy array or pandas series
    :param coeff: FIR coefficients. Should be with odd length and sysmmetric.
    :return: The filtered data series
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    # apply the filter
    data_filtered = signal.lfilter(coeff, 1.0, data)
    # data_filtered = pd.Series(data_filtered)

    # reverse the time shift caused by the filter,
    # corruption regions contain zeros
    # If the number of filter coefficients is odd, the central point *should*
    # be included in the output so we only zero out a region of len(coeff) - 1
    data_filtered[:(len(coeff) // 2) * 2] = 0
    data_filtered = roll(data_filtered, -len(coeff) // 2)
    return data_filtered


def lpass(data, fs, f_cut, order, beta=5.0):
    """
    FIR lowpass filter design
    :param data: a numpy array or pandas series
    :param fs: sampling frequency of the data array
    :param f_cut: The low frequency cutoff treshhiold
    :param order:   The number of corrupted samples on each  side of the data
    :param beta: side lobe attenuation parameter in kaiser window
    :return:
    """
    nyqst = float(fs/2)
    k = f_cut / nyqst
    coeff = signal.firwin(order * 2 + 1, k, window=('kaiser', beta))
    return fir_filt(data, coeff)


def hpass(data, fs, f_cut, order, beta=5.0):
    """
    FIR hpass filter design
    :param data: a numpy array or pandas series
    :param fs: sampling frequency of the data array
    :param f_cut: The high frequency cutoff treshhiold
    :param order:   The number of corrupted samples on each  side of the data
    :param beta: side lobe attenuation parameter in kaiser window
    :return:
    """
    nyqst = float(fs/2)
    k = f_cut / nyqst
    coeff = signal.firwin(
        order * 2 + 1, k, window=('kaiser', beta), pass_zero=False)
    return fir_filt(data, coeff)


def bpass(data, fs, cut_low, cut_high, order, beta=5.0):
    """
    FIR band pass filter design
    :param data: a numpy array or pandas series
    :param fs: sampling frequency of the data array
    :param cut_low: The low frequency cutoff treshhiold
    :param cut_high: The high frequency cutoff treshhiold
    :param order: The number of corrupted samples on each  side of the data
    :param beta: side lobe attenuation parameter in kaiser window
    :return:
    """
    nyqst = float(fs/2)
    k1 = cut_low / nyqst
    k2 = cut_high / nyqst
    coeff = signal.firwin(order * 2 + 1, [k1, k2], window=('kaiser', beta))
    return fir_filt(data, coeff)


def notch_filt(data, fs, f_cut, Q=30):
    """
    IIR notch filter design
    :param data: numpy array or pandas series
    :param fs: sampling frequency of the data array
    :param f_cut: The cutoff frequency in Hz
    :param Q: Order of the quality factor filter
    :return:
    """
    nyqst = float(fs/2)
    K = f_cut / nyqst  # Normalized Frequency
    b, a = signal.iirnotch(K, Q)
    data_filtered = signal.lfilter(b, a, data)
    return data_filtered



def p_corr(x, y, tau):
    """
    :param x:
    :param y:
    :param tau:
    :return:
    """
    if len(x) != len(y):
        print("The arrays must be of the same length!")
        return

    if tau > 0:
        x = x[:-tau]
        y = y[tau:]
    elif tau != 0:
        x = x[np.abs(tau):]
        y = y[:-np.abs(tau)]

    n = len(x)

    x = x - np.mean(x)
    y = y - np.mean(y)

    std_x = np.std(x)
    std_y = np.std(y)

    cov = np.sum(x * y) / (n - 1)
    cr = cov / (std_x * std_y)
    cr_err = np.std(x * y) / (np.sqrt(n) * std_x * std_y) \
             + cr * (np.std(x ** 2) / (2 * std_x ** 2)
                     + np.std(y ** 2) / (2 * std_y ** 2)) / np.sqrt(n)

    return cr, cr_err


# =============================================================================
# 
# =============================================================================

