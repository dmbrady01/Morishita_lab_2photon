#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analysis.py: Python script that processes and analyzes fiber photometry data.
"""


__author__ = "DM Brady"
__datewritten__ = "07 Mar 2018"
__lastmodified__ = "08 Mar 2018"

import sys
from imaging_analysis.event_processing import LoadEventParams
from imaging_analysis.segment_processing import TruncateSegments
from imaging_analysis.utils import ReadNeoPickledObj, ReadNeoTdt, WriteNeoPickledObj, PrintNoNewLine
from imaging_analysis.signal_processing import ProcessSignalData
#######################################################################
# VARIABLES TO ALTER
path_to_event_params = 'imaging_analysis/event_params.json'
truncate_begin = 10 # how many seconds to remove from the beginning of the recording
truncate_end = 0 # how many seconds to remove from the end of the recording
lowpass_filter = 40.0 # Will remove frequencies above this number
signal_channel = 'LMag 1'
reference_channel = 'LMag 2'
deltaf_ch_name = 'DeltaF_F'
deltaf_options = {}

##########################################################################
# This loads our event dictionary {'1': 'correct', '2': 'incorrect', ...}
evtframe = LoadEventParams(dpath=path_to_event_params)

# Checks if a directory path to the data is provided, if not, will
# use what is specified in except
try:
    dpath = sys.argv[1]
except IndexError:
    dpath = '/Users/DB/Development/Monkey_frog/data/TDT-LockinRX8-22Oct2014_20-4-15_DT4_1024174'

# Tries to load a processed pickle object, othewise reads the Tdt folder,
# processes the data and writes a pickle object
try:
    # Attempting to load pickled object
    PrintNoNewLine('Trying to load processed pickled object...')
    seglist = ReadNeoPickledObj(path=dpath, name="processed.pkl", 
                                return_block=False)
    print('Done!')

except IOError:
    # Reads data from Tdt folder
    PrintNoNewLine('\nCannot find processed pkl object, reading TDT folder instead...')
    block = ReadNeoTdt(path=dpath, return_block=True)
    seglist = block.segments
    print('Done!')

    # Trunactes first/last seconds of recording
    PrintNoNewLine('Truncating signals and events...')
    seglist = TruncateSegments(seglist, start=truncate_begin, end=truncate_end, clip_same=True)
    print('Done!')

    # Iterates through each segment in seglist. Right now, there is only one segment
    for segment in seglist:
        # Extracts the sampling rate from the signal channel
        sampling_rate = filter(lambda x: x.name == signal_channel, segment.analogsignals)[0].sampling_rate
        # Appends an analog signal object that is delta F/F. The name of the channel is
        # specified by deltaf_ch_name above. It is calculated using the function
        # NormalizeSignal in signal_processing.py. As of right now it:
        # 1) Lowpass filters signal and reference (default cutoff = 40 Hz, order = 5)
        # 2) Calculates deltaf/f for signal and reference (default is f - median(f) / median(f))
        # 3) Detrends deltaf/f using a savgol filter (default window_lenght = 3001, poly order = 1)
        # 4) Subtracts reference from signal
        # NormalizeSignal has a ton of options, you can pass in paramters using
        # the deltaf_options dictionary above. For example, if you want it to be mean centered
        # and not run the savgol_filter, set deltaf_options = {'mode': 'mean', 'detrend': False}
        PrintNoNewLine('\nCalculating delta_f/f...')
        ProcessSignalData(seg=segment, sig_ch=signal_channel, ref_ch=reference_channel,
                        name=deltaf_ch_name, fs=sampling_rate, highcut=lowpass_filter, **deltaf_options)
        print('Done!')

