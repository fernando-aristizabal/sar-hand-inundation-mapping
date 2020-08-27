#!/usr/bin/env python3

## Python 3 script to extract GCP's to raster file with gdal_translate utility

# call: ./extractGCPS <gcps file> <source raster>

import os
import sys
import re

gcpsFile = sys.argv[1]
sourceFile = sys.argv[2]

# save raster metadata to temp file
os.system('gdalinfo ' + sourceFile + ' > tempFile.txt')

# open, parse, write new file
prevLine = False
with open('tempFile.txt') as ip, open(gcpsFile,'w+') as op: # open temp file
	for line in ip: # parse
		if prevLine:
			parsedList = re.findall(r"[-+]?\d*\.\d+|\d+",line) # extract numbers
			for val in parsedList:
				op.write(val + " ") # write numbers
			op.write("\n")
			prevLine = False
		else:
			if line[0:4] == 'GCP[': # find lines
				prevLine = True
			else:
				prevLine = False

os.remove('tempFile.txt')


