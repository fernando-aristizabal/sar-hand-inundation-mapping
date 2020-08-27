#!/usr/bin/env python3

import numpy as np
import gis
import copy


def generateValidation(handRasterFileName, outputFileName,waterHeightFt):
	hand = gis.raster(handRasterFileName)
	validation = copy.copy(hand)
	validation.ndv = 0
	waterHeightM = waterHeightFt / 3.28084

	validation.array = np.ones((hand.nrows,hand.ncols),dtype=int) + 1
	validation.array[hand.array[2,:,:] > waterHeightM] = 1
	validation.array[hand.array[2,:,:] == hand.ndv] = 0
	

	gis.writeGeotiff(validation,outputFileName)