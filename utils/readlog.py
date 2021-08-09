
from pylab import *
import struct
import sys
import os, fnmatch
sys.path.append( sys.path[0] +'/..')
import argparse
from time import sleep
import numpy as np


def readLogFile(filename, verbose=True):
    f = open(filename, 'rb')

    print('Opened'),
    print(filename)

    keys = f.readline().decode('utf8').rstrip('\n').split(',')
    fmt = f.readline().decode('utf8').rstrip('\n')

    # The byte number of one record
    sz = struct.calcsize(fmt)
    # The type number of one record
    ncols = len(fmt)

    if verbose:
        print('Keys:'),
        print(keys)
        print('Format:'),
        print(fmt)
        print('Size:'),
        print(sz)
        print('Columns:'),
        print(ncols)

    # Read data
    wholeFile = f.read()
    # split by alignment word
    chunks = wholeFile.split(b'\xaa\xbb')
    log = list()
    for chunk in chunks:
        if len(chunk) == sz:
            
            values = struct.unpack(fmt, chunk)
            record = list()
            for i in range(ncols):
                record.append(values[i])
            log.append(record)

    return log

if __name__ == '__main__':
    
    log = readLogFile("log.txt")
    position = []
    for record in log:
        Id = record[2]
        pos = [record[3], record[4], record[5]]
        orn = [record[6], record[7], record[8], record[9]]
        position.append(pos)
    position = np.array(position)
    np.savetxt("log.csv", position, delimiter=",")

