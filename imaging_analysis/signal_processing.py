#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
signal_processing.py: Python script that contains functions for signal
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "07 Mar 2018"


import scipy.signal as ssp
import numpy as np
import neo
import quantities as pq
import copy as cp
from neo.core import AnalogSignal

def TruncateSignal(signal, start=0, end=0):
    """Given an AnalogSignal object, will remove the first 'start' seconds and the 
    last 'end' seconds. Please note that this is different logic
    than TruncateEvent! TruncateSignals trims start/end seconds from the ends no matter
    the length. TruncateEvent start/end truncates at those specific timestamps. It
    is not relative."""
    # Makes sure signal is an AnalogSignal object
    if not isinstance(signal, neo.core.AnalogSignal):
        raise TypeError('%s must be an AnalogSignal object.' % signal)
    # converts start and end to times
    start = start * pq.s 
    end = end * pq.s
    # Truncate signal
    truncated_signal = signal.time_slice(signal.t_start + start, signal.t_stop - end)

    return truncated_signal

def TruncateSignals(signal_list, start=0, end=0):
    """Given a list of AnalogSignal objects, will iterate through each one and remove
    values before 'start' and after 'end'. Start and end must be in seconds."""
    # Makes sure a list is passed
    if not isinstance(signal_list, list):
        raise TypeError('%s must be a list' % signal_list)
    # Iterate through each item with TrunacteSignal
    truncated_list = [TruncateSignal(sig, start=start, end=end) for sig in signal_list]
    
    return truncated_list

def ButterFilterDesign(lowcut=None, highcut=None, fs=381.469726562, order=5, 
                    btype='lowpass'):
    """Constructs the numerator and denominator for the low/high/bandpass/bandstop
    filter. low and high are the cutoff frequencies, fs is the sampling frequency, 
    order is the desired filter order, and btype designates what type of filter will
    ultimately be constructed (lowpass, highpass, bandpass, bandstop)."""
    # calculate nyquist frequency (half the sampling frequency)
    nyquist = fs * 0.5
    # try constructing normalized lowcut
    try:
        norm_low = lowcut / nyquist
    except TypeError:
        pass
    # try constructing normalized highcut
    try:
        norm_high = highcut / nyquist
        # Makes sure highcut is lower than nyquist frequency
        if norm_high > 1:
            raise ValueError('%s is larger than the nyquist frequency. Choose a smaller highcut (at most half the sampling rate).' % highcut)
    except TypeError:
        pass
    # Construct param to be bassed into butter
    if btype == 'lowpass':
        params = norm_high
    elif btype == 'highpass':
        params = norm_low
    elif (btype == 'bandpass') or (btype == 'bandstop'):
        params = [norm_low, norm_high]
    else:
        raise ValueError("%s must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'" % btype)
    # Return butter
    return ssp.butter(order, params, btype=btype, analog=False)

def FilterSignal(signal, lowcut=None, highcut=None, fs=381, order=5, 
                btype='lowpass', axis=0, window_length=3001, savgol_order=1):
    """Given some signal data, uses ButterFilterDesign or savgol_filter to 
    construct a filter and applies it to signal.
    signal: data signal
    lowcut: lower bound for filter
    highcut: upper bound for filter
    fs: sampling frequnecy
    order: filter order
    btype: 'lowpass', 'highpass', 'bandpass', 'bandstop', 'savgol'
    axis: axis of matrix to apply filter"""
    # If btype is savgol, performs a Savitzky-Golay filter
    if btype == 'savgol':
        return ssp.savgol_filter(signal, window_length, savgol_order, axis=axis)
    # Otherwise performs filfilt (backwards and forwards filtering)
    else:
        b, a = ButterFilterDesign(lowcut=lowcut, highcut=highcut, fs=fs, 
                                order=order, btype=btype)
        return ssp.filtfilt(b, a, signal, axis=axis)

def DeltaFOverF(signal, reference=None, period=None, mode='median'):
    """Calcualte DeltaF/F for a signal. Units are %. There are several modes:
    'median': (signal - median(signal))/median(signal)

    'mean': (signal - mean(signal))/mean(signal)
    
    'reference': (signal - reference_signal)/reference_signal
    Note that reference must be a same length array as signal
    
    'period_mean': ref = mean(signal[period[0]:period[1]] 
                   (signal - ref)/ref
    
    'period_median': same as 'period_mean' but uses median instead
    Note that period must be a list or tuple of beginning and end timstamps"""
    if mode == 'median':
        return (signal - np.median(signal))/np.median(signal) * 100.0
    elif mode == 'mean':
        return (signal - np.mean(signal))/np.mean(signal) * 100.0
    elif mode == 'reference':
        return (signal - reference)/reference * 100.0
    elif mode == 'period_mean':
        reference = signal[period[0]:period[1]]
        return (signal - np.mean(reference))/np.mean(reference) * 100.0
    elif mode == 'period_median':
        reference = signal[period[0]:period[1]]
        return (signal - np.median(reference))/np.median(reference) * 100.0
    else:
        raise ValueError('%s is not an accepted mode for calculating deltaf' % mode)

def NormalizeSignal(signal=None, reference=None, **kwargs):
    """The current method for correcting a signal. These are the steps:
    1) Lowpass filter signal and reference
    2) Calculate deltaf/f for signal and reference
    3) Detrend deltaf/f using a savgol filter (if detrend is True)
    4) Subtract reference from signal

    There are no code tests for this since it is likely to change when you come
    up with a better strategy."""
    # Default options
    options = {
        'btype': 'lowpass', 
        'highcut': 40.0, 
        'lowcut': None,
        'order': 5,
        'axis': 0,
        'fs': 381,
        'window_length': 3001,
        'savgol_order': 1,
        'detrend': True,
        'mode': 'median',
        'period': None,
        'return_all_signals': False
        }
    # Update based on kwargs
    options.update(kwargs)
    # Pass signal and reference through filters
    filt_signal = FilterSignal(signal, lowcut=options['lowcut'], highcut=options['highcut'], 
                            fs=options['fs'], order=options['order'], btype=options['btype'], 
                            axis=options['axis'])
    filt_ref = FilterSignal(reference, lowcut=options['lowcut'], highcut=options['highcut'], 
                            fs=options['fs'], order=options['order'], btype=options['btype'], 
                            axis=options['axis'])
    # Calculate deltaf/f
    deltaf_sig = DeltaFOverF(filt_signal, reference=filt_ref, mode=options['mode'], 
                            period=options['period'])
    deltaf_ref = DeltaFOverF(filt_ref, reference=filt_ref, mode=options['mode'], 
                            period=options['period'])
    # Detrend data if detrend is true
    if options['detrend']:
        # for signal
        trend_sig = FilterSignal(deltaf_sig, fs=options['fs'], btype='savgol', 
                                axis=options['axis'], window_length=options['window_length'], 
                                savgol_order=options['savgol_order'])
        deltaf_sig = deltaf_sig - trend_sig
        # for reference
        trend_ref = FilterSignal(deltaf_ref, fs=options['fs'], btype='savgol', 
                                axis=options['axis'], window_length=options['window_length'], 
                                savgol_order=options['savgol_order'])
        deltaf_ref = deltaf_ref - trend_ref
    # Subtract reference out
    subtracted_signal = deltaf_sig - deltaf_ref
    # Return signal with reference subtracted out
    if not options['return_all_signals']:
        return subtracted_signal
    else:
        # returns all processing steps (but will jump from filt to deltaf detrended)
        # if detrend = True
        return subtracted_signal, deltaf_sig, deltaf_ref, filt_signal, filt_ref

def ProcessSignalData(seg=None, sig_ch='LMag 1', ref_ch='LMag 2', name='deltaf_f', **kwargs):
    """Given a segment object, it will extract the analog signal channels specified as
    signal (sig_ch) and reference (ref_ch), and will perform NormalizeSignal on them.
    Will append the new signal to the segment object as 'name'."""
    # Gets any keyword arguments
    options = kwargs
    # Check that segment object was passed
    if not isinstance(seg, neo.core.Segment):
        raise TypeError('%s must be a Segment object' % seg)
    # Retrieves signal and reference
    signal = filter(lambda x: x.name == sig_ch, seg.analogsignals)[0]
    reference = filter(lambda x: x.name == ref_ch, seg.analogsignals)[0]
    # Build a new AnalogSignal based on the other ones
    new_signal = NormalizeSignal(signal=signal.magnitude, reference=reference.magnitude, **options)
    units = pq.percent # new units are in %
    t_start = signal.t_start # has the same start time as all other analog signal objects in segment
    fs = signal.sampling_rate # has the same sampling rate as all other analog signal objects in segment
    # Creates new AnalogSignal object
    deltaf_f = AnalogSignal(new_signal, units=units, t_start=t_start, sampling_rate=fs, name=name)
    # Adds processed signal back to segment
    seg.analogsignals.append(deltaf_f)
