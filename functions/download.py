import urllib.request
import zipfile
import glob
import sentinelsat
import os
import shutil
import sys
import getpass

def reporthook(blocknum, blocksize, totalsize):
	readsofar = blocknum * blocksize
	if totalsize > 0:
		percent = readsofar * 1e2 / totalsize
		s = "\r%5.1f%% %*d / %d" % (
			percent, len(str(totalsize)), readsofar, totalsize)
		sys.stderr.write(s)
		if readsofar >= totalsize: # near the end
			sys.stderr.write("\n")
	else: # total size is unknown
		sys.stderr.write("read %d\n" % (readsofar,))


def downloadFiles(sar,hand,lc,gage,nhd,sentinelUSERNAME='',sentinelPASSWD=''):

	# sar
	if sar:
		if sentinelUSERNAME == '':
			sentinelUSERNAME = input('Copernicus Username: ')
		if sentinelPASSWD == '':
			sentinelPASSWD = getpass.getpass('Copernicus Password: ')
		print('.... Downloading SAR data .....')
		if os.path.isdir('data/sar/S1A_IW_GRDH_1SDV_20161012T111514_20161012T111543_013456_01580C_1783.SAFE'): 
			for f in glob.glob('data/sar/S1A_IW_GRDH_1SDV_20161012T111514_20161012T111543_013456_01580C_1783.SAFE/*'):
				os.remove(f)
			os.rmdir('data/sar/S1A_IW_GRDH_1SDV_20161012T111514_20161012T111543_013456_01580C_1783.SAFE')
		api = sentinelsat.SentinelAPI(sentinelUSERNAME, sentinelPASSWD , 'https://scihub.copernicus.eu/apihub/.')
		api.download('28e8847a-dce6-4df1-90b5-c0307c643373','data/sar')
		print('.... Extracting SAR data .....')
		zip_ref = zipfile.ZipFile('data/sar/S1A_IW_GRDH_1SDV_20161012T111514_20161012T111543_013456_01580C_1783.zip', 'r')
		zip_ref.extractall('data/sar/')
		zip_ref.close()
		os.remove('data/sar/S1A_IW_GRDH_1SDV_20161012T111514_20161012T111543_013456_01580C_1783.zip')
	else:
		print('.... Skipping SAR data download ....')

	# hand
	if hand:
		print('.... Downloading HAND data .....')
		if os.path.isfile('data/hand/original/030202hand.tif'): os.remove('data/hand/original/030202hand.tif')
		urllib.request.urlretrieve('https://web.corral.tacc.utexas.edu/nfiedata/HAND/030202/030202hand.tif', 'data/hand/original/030202hand.tif',reporthook)
	else:
		print('.... Skipping HAND data download ....')

	# lc 
	if lc:
		print('.... Downloading Landcover data .....')
		if os.path.isfile('data/lc/original/nlcd_2011_landcover_2011_edition_2014_10_10.img'): 
			for f in glob.glob('data/lc/original/*'):
				os.remove(f)
		urllib.request.urlretrieve('http://www.landfire.gov/bulk/downloadfile.php?TYPE=nlcd2006&FNAME=nlcd_2006_landcover_2011_edition_2014_10_10.zip','data/lc/original/nlcd.zip',reporthook)
		print('.... Extracting Landcover data .....')
		zip_ref = zipfile.ZipFile('data/lc/original/nlcd.zip', 'r')
		zip_ref.extractall('data/lc/original/')
		zip_ref.close()
		os.remove('data/lc/original/nlcd.zip')
		for f in os.listdir('data/lc/original/nlcd_2006_landcover_2011_edition_2014_10_10/'):
			shutil.move(os.path.join('data/lc/original/nlcd_2006_landcover_2011_edition_2014_10_10',f),'data/lc/original/')
		os.rmdir('data/lc/original/nlcd_2006_landcover_2011_edition_2014_10_10')
	else:
		print('.... Skipping Landcover data download ....')

	# gages
	if gage:
		print('.... Downloading Stream Gage Data ....')
		if os.path.isfile('data/gages/original/gagesII_9322_sept30_2011.shp'):
			for f in glob.glob('data/gages/original/gagesII_9322_sept30_2011.*'):
				os.remove(f)
		urllib.request.urlretrieve('https://water.usgs.gov/GIS/dsdl/gagesII_9322_point_shapefile.zip','data/gages/original/gages.zip',reporthook)
		print('.... Extracting Gage data .....')
		zip_ref = zipfile.ZipFile('data/gages/original/gages.zip', 'r')
		zip_ref.extractall('data/gages/original/')
		zip_ref.close()
		os.remove('data/gages/original/gages.zip')
	else:
		print('.... Skipping Stream Gage data download ....')

	# nhd
	if nhd:
		print('.... Downloading NHD data .....')
		if os.path.isfile('data/nhd/original/NHDFlowline.shp'):
			for f in glob.glob('data/nhd/original/*'):
				os.remove(f)
		urllib.request.urlretrieve('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/State/HighResolution/Shape/NHD_H_North_Carolina_Shape.zip','data/nhd/original/NHD_H_North_Carolina_Shape.zip',reporthook)
		print('.... Extracting NHD data .....')
		zip_ref = zipfile.ZipFile('data/nhd/original/NHD_H_North_Carolina_Shape.zip', 'r')
		zip_ref.extractall('data/nhd/original/')
		zip_ref.close()
		os.remove('data/nhd/original/NHD_H_North_Carolina_Shape.zip')
		for f in os.listdir('data/nhd/original/Shape/'):
			shutil.move(os.path.join('data/nhd/original/Shape',f),'data/nhd/original/')
		os.rmdir('data/nhd/original/Shape')
	else:
		print('.... Skipping NHD data download ....')