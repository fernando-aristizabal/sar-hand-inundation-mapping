#!/usr/bin/env python3

import os
import sys
from importlib import reload
from getpass import getuser

# append path
projectDirectory = os.path.join('/home',getuser(),'sarHand')
os.chdir(projectDirectory)
sys.path.append(projectDirectory + '/scripts/functions')

import download
import analysis
import gisPreprocessing
import gis
import generateValidation
import landcoverProcessing

#### TO - DO ############
# testing
# setup script?
# organize like real python program?
# centralized variable file
# rscript organization
# auto calibrate/filtering

## DATA DOWNLOAD ##############################################################################
sar = False ; hand = False ; lc = False ; gage = False ; nhd = False

#download.downloadFiles(sar,hand,lc,gage,nhd,sentinelUSERNAME='fern6050')

## SNAP TOOLBOX: CALIBRATION & FILTERING #########################################################
"""
print("Complete SAR calibration and filtering using SNAP Toolbox now.\n " +
	"Place VV within \'data/processed/\' as \'vv_cal_spk.tif\'.\n " +
	"Place VH within \'data/processed/\' as \'vh_cal_spk.tif\'")
while True:
	snap = input('Have you completed SAR calibration and filtering using SNAP? (y)es or (n)o ')
	if (snap == 'Y') or (snap == 'y') or (snap == 'Yes') or (snap == 'yes'):
		break
	if (snap == 'N') or (snap == 'n') or (snap == 'No') or (snap == 'no'):
		print("\nPlease complete calibration and filtering using SNAP prior to continuing\n")
		snap = input('Have you completed SAR calibration and filtering using SNAP? (y)es or (n)o ')
"""


## PREPROCESS GIS DATA ##############################################################################
cartesianProjection = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
sphericalProjection = "EPSG:4326"
noDataValue = -9999
# peak dates in october: 9,12,14
areas = ["Smithfield","Goldsboro","Kinston"]
stationIDs = ["02087570",'02089000','02089500']

#print('\n..... Preprocessing rasters and vector data ....\n')
#gisPreprocessing.gisPreprocessing(cartesianProjection,sphericalProjection,areas,stationIDs,noDataValue)

## AGGREGATE & SCALE LANDCOVER DATA #################################################################
ndvLC = 0
levelIIs = [11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95]
levelIINames = ['Open Water','Perennial Ice/Snow','Developed Open Space','Developed Low Intensity','Developed Medium Intensity','Developed High Intensity','Barren Land','Deciduous Forest','Evergreen Forest','Mixed Forest','Dwarf Shrub','Shrub/Scrub','Grassland/Herbaceous','Sedge/Herbaceous','Lichens','Moss','Pasture/Hay','Cultivated Crops','Woody Wetlands','Emergent Herbaceous Wetlands']
groups =  {'Other': [11,12,31,51,71,72,73,74,95],
			'Developed' : [21,22,23,24],
			'Canopy' : [41,42,43,52,90],
			'Agriculture' : [81,82]
		   }
origLandCover = 'data/lc/processed/lcNC_proj_{0}_proj_scaled_{0}.img'
groupedLandCover = 'data/lc/processed/lcNC_proj_{0}_proj_scaled_{0}_grouped.img'

#print('\n.........LC Grouping...............\n')
#landcoverProcessing.combineLC(origLandCover, groupedLandCover, ndvLC,levelIIs,levelIINames,groups,areas)

## CLASSIFICATION ###################################################################################

#print('\n.......Classification......\n')
#os.system('Rscript ./scripts/functions/classification.R {} {} {} {} {}'.format(projectDirectory, str(noDataValue), areas[0],areas[1],areas[2]))

## STATISTICS & ANALYSIS #######################################################################################

classificationModels = ["qda","svm", "knn"]
predictors = ['three','two']
groups = ['Overall','Developed','Canopy','Agriculture']
# writeDiffRasters = True
writeDiffRasters = False

print('\n.......Analysis......\n')
# analysis.statsCalculate(predictors,classificationModels,areas,groups,writeDiffRasters)
analysis.plots(areas,classificationModels)
