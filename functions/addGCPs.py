#!/usr/bin/env python3

## Python 3 script to add GCP's to raster file with gdal_translate utility

# call: ./addGCPS <gcps file> <source raster> <destination raster>

import os
import sys

gcpsFile = sys.argv[1]
source = sys.argv[2]
destination = sys.argv[3]

# remove destination file if present
if os.path.isfile(destination):
	os.remove(destination)


# create allGCPS command
allGCPS = str()
with open(gcpsFile) as fp:
    for line in fp:
        line = line.rstrip()
        allGCPS = allGCPS + " -gcp " + line

# complete command
finalCommand = "gdal_translate -a_srs EPSG:4326" + allGCPS + " " + source + " " + destination

# execute
os.system(finalCommand)
