#!/usr/bin/env python3


import os
import glob
import copy
import gis
import numpy as np
import ogr

# X sar,different dates sar , X hand, Xflowlines, Xgages, Xhwms, Xlc, all validation (boundary, inundation)

def gisPreprocessing(cartesianProjection,sphericalProjection,areas,stationIDs,noDataValue):

	# extract ground control points
	print('.... Extracting Ground Control Points from Original SAR rasters ....')
	os.system('./scripts/functions/extractGCPs.py data/sar/processed/gcp.txt data/sar/S1A_IW_GRDH_1SDV_20161012T111514_20161012T111543_013456_01580C_1783.SAFE/measurement/s1a-iw-grd-vv-20161012t111514-20161012t111543-013456-01580c-001.tiff')

	# Re-add ground control points
	print('.... Adding Ground Control Points To SAR Rasters ....')
	os.system('./scripts/functions/addGCPs.py data/sar/processed/gcp.txt data/sar/processed/vv_cal_spk.tif data/sar/processed/vv_cal_spk_gcp.tiff')
	os.system('./scripts/functions/addGCPs.py data/sar/processed/gcp.txt data/sar/processed/vh_cal_spk.tif data/sar/processed/vh_cal_spk_gcp.tiff')

	# extract gages
	print('.... Isolating Gage for Areas ....')
	for stID in zip(stationIDs,areas):
		cmd = 'ogr2ogr -overwrite -where "STAID = '+ "'" + stID[0] + "'" + '" data/gages/processed/gage_' + stID[1] + '.shp data/gages/original/gagesII_9322_sept30_2011.shp'
		os.system(cmd)

	# projections to spherical coordinates

	print('.... Reprojecting GIS files to {} Projections ....'.format(sphericalProjection))
	os.system('gdalwarp -r bilinear -t_srs \'{}\' -overwrite data/sar/processed/vv_cal_spk_gcp.tiff data/sar/processed/vv_cal_spk_gcp_proj.tiff'.format(sphericalProjection))
	os.remove('data/sar/processed/vv_cal_spk_gcp.tiff')
	os.system('gdalwarp -r bilinear -t_srs \'{}\' -overwrite data/sar/processed/vh_cal_spk_gcp.tiff data/sar/processed/vh_cal_spk_gcp_proj.tiff'.format(sphericalProjection))
	os.remove('data/sar/processed/vh_cal_spk_gcp.tiff')
	os.system('gdalwarp -r bilinear -t_srs \'{}\' -overwrite data/hand/original/030202hand.tif data/hand/processed/hand_proj.tiff'.format(sphericalProjection))
	os.system('gdalwarp -t_srs \'{}\' -overwrite -of HFA data/lc/original/nlcd_2006_landcover_2011_edition_2014_10_10.img data/lc/processed/lcNC_proj.img'.format(sphericalProjection))

	for a in areas:
		os.system('ogr2ogr data/validation/processed/Boundary_{}_proj.shp -overwrite -progress -t_srs \'{}\' data/validation/{}/Boundary_{}.shp'.format(a,sphericalProjection,a,a))

	os.system('ogr2ogr data/nhd/processed/NHDFlowline_proj.shp -overwrite -progress -t_srs \'{}\' data/nhd/original/NHDFlowline.shp'.format(sphericalProjection))
	os.system('ogr2ogr data/nhd/processed/huc6_neuse_proj.shp -overwrite -progress -t_srs \'{}\' data/nhd/processed/huc6_neuse.shp'.format(sphericalProjection))


	# extract Neuse River and Neuse Basin
	cmd = 'ogr2ogr -overwrite -where "GNIS_NAME=\'Neuse River\'" data/nhd/processed/neuseRiver.shp data/nhd/processed/NHDFlowline_proj.shp'
	os.system(cmd)
	cmd = 'ogr2ogr -overwrite -where "NAME=\'Neuse\'" data/nhd/processed/huc6_neuse.shp data/nhd/original/WBDHU6.shp'
	os.system(cmd)

	# clip to study area
	print('.... Clipping Files to General Study Area ....')

	for a in areas:
		driver = ogr.GetDriverByName('ESRI Shapefile')
		dataSource = driver.Open('data/validation/processed/Boundary_{}_proj.shp'.format(a), 0)
		layer = dataSource.GetLayer()
		extent = layer.GetExtent()
		dataSource.Destroy()

		ulx = extent[0] ; uly = extent[3] ; lrx = extent[1] ; lry = extent[2]
		xd = abs(ulx-lrx) ; yd = abs(uly-lry)
		marginFactor = 0.05
		ulx = ulx - xd * marginFactor ; lrx = lrx + xd * marginFactor
		uly = uly + yd * marginFactor ; lry = lry - yd * marginFactor

		os.system('gdalwarp -r bilinear -overwrite -te {} {} {} {} data/sar/processed/vv_cal_spk_gcp_proj.tiff data/sar/processed/vv_cal_spk_gcp_proj_{}.tiff'.format(ulx,lry,lrx,uly,a,a))
		os.system('gdalwarp -r bilinear -overwrite -te {} {} {} {} data/sar/processed/vh_cal_spk_gcp_proj.tiff data/sar/processed/vh_cal_spk_gcp_proj_{}.tiff'.format(ulx,lry,lrx,uly,a,a))
		os.system('gdalwarp -r bilinear -overwrite -te {} {} {} {} data/hand/processed/hand_proj.tiff data/hand/processed/hand_proj_{}.tiff'.format(ulx,lry,lrx,uly,a,a))
		os.system('gdalwarp -overwrite -te {} {} {} {} data/hand/processed/catchmask_proj.tiff data/hand/processed/catchmask_proj_{}.tiff'.format(ulx,lry,lrx,uly,a,a))

		os.system('gdalwarp -overwrite -of HFA -te {} {} {} {} data/lc/processed/lcNC_proj.img data/lc/processed/lcNC_proj_{}.img'.format(ulx,lry,lrx,uly,a,a))
		os.system('ogr2ogr -overwrite -progress -clipsrc data/validation/processed/Boundary_{}_proj.shp data/nhd/processed/NHDFlowline_proj_{}.shp data/nhd/processed/NHDFlowline_proj.shp'.format(a,a))

	#os.remove('data/sar/processed/vh_cal_spk_gcp_proj.tiff')
	#os.remove('data/sar/processed/vv_cal_spk_gcp_proj.tiff')
	#os.remove('data/hand/processed/hand_proj.tiff')
	#for f in glob.glob('data/lc/processed/lcNC_proj.*'): os.remove(f)
	#for f in glob.glob('data/nhd/processed/NHDFlowline_proj.*'): os.remove(f)

	# convert to db
	print('.... Converting SAR data to Decibels ....')
	for a in areas:
		os.system('gdal_calc.py --overwrite -A data/sar/processed/vv_cal_spk_gcp_proj_{}.tiff --outfile=data/sar/processed/vv_cal_spk_gcp_proj_{}_db.tiff --calc="10*log10(A)"'.format(a,a))
		os.remove('data/sar/processed/vv_cal_spk_gcp_proj_{}.tiff'.format(a))
		os.system('gdal_calc.py --overwrite -A data/sar/processed/vh_cal_spk_gcp_proj_{}.tiff --outfile=data/sar/processed/vh_cal_spk_gcp_proj_{}_db.tiff --calc="10*log10(A)"'.format(a,a))
		os.remove('data/sar/processed/vh_cal_spk_gcp_proj_{}.tiff'.format(a))

	# convert raster data types to accept signed values and convert sar/hand to same no data values
	print('.... Changing No Data Value of Rasters to Standard Values ....')
	for a in areas:
		os.system('gdalwarp -r bilinear -overwrite -ot Float32 -srcnodata -inf -dstnodata {} data/sar/processed/vv_cal_spk_gcp_proj_{}_db.tiff data/sar/processed/vv_cal_spk_gcp_proj_{}_db_nodata.tiff'.format(noDataValue,a,a))
		os.remove('data/sar/processed/vv_cal_spk_gcp_proj_{}_db.tiff'.format(a))
		os.system('gdalwarp -r bilinear -overwrite -ot Float32 -srcnodata -inf -dstnodata {} data/sar/processed/vh_cal_spk_gcp_proj_{}_db.tiff data/sar/processed/vh_cal_spk_gcp_proj_{}_db_nodata.tiff'.format(noDataValue,a,a))
		os.remove('data/sar/processed/vh_cal_spk_gcp_proj_{}_db.tiff'.format(a))
		os.system('gdalwarp -r bilinear -overwrite -ot Float32 -srcnodata -3.4028234663852886e+38 -dstnodata {} data/hand/processed/hand_proj_{}.tiff data/hand/processed/hand_proj_{}_nodata.tiff'.format(noDataValue,a,a))
		os.remove('data/hand/processed/hand_proj_{}.tiff'.format(a))
		os.system('gdalwarp -overwrite -ot Int32 -srcnodata 0 -dstnodata {} data/hand/processed/catchmask_proj_{}.tiff data/hand/processed/catchmask_proj_{}_nodata.tiff'.format(noDataValue,a,a))
		os.remove('data/hand/processed/catchmask_proj_{}.tiff'.format(a))

	# project datasets
	print('.... Reprojecting GIS files to {} Projections ....'.format(cartesianProjection))
	for a in areas:
		os.system('gdalwarp -r bilinear -t_srs \'{}\' -overwrite data/sar/processed/vv_cal_spk_gcp_proj_{}_db_nodata.tiff data/sar/processed/vv_cal_spk_gcp_proj_{}_db_nodata_proj.tiff '.format(cartesianProjection,a,a))
		os.system('gdalwarp -r bilinear -t_srs \'{}\' -overwrite data/sar/processed/vh_cal_spk_gcp_proj_{}_db_nodata.tiff data/sar/processed/vh_cal_spk_gcp_proj_{}_db_nodata_proj.tiff '.format(cartesianProjection,a,a))
		os.system('gdalwarp -r bilinear -t_srs \'{}\' -overwrite data/hand/processed/hand_proj_{}_nodata.tiff data/hand/processed/hand_proj_{}_nodata_proj.tiff'.format(cartesianProjection,a,a))
		os.system('gdalwarp -t_srs \'{}\' -overwrite data/hand/processed/catchmask_proj_{}_nodata.tiff data/hand/processed/catchmask_proj_{}_nodata_proj.tiff'.format(cartesianProjection,a,a))
		os.remove('data/sar/processed/vv_cal_spk_gcp_proj_{}_db_nodata.tiff'.format(a))
		os.remove('data/sar/processed/vh_cal_spk_gcp_proj_{}_db_nodata.tiff'.format(a))
		os.remove('data/hand/processed/hand_proj_{}_nodata.tiff'.format(a))
		os.remove('data/hand/processed/catchmask_proj_{}_nodata.tiff'.format(a))

		os.system('ogr2ogr data/validation/processed/Boundary_{}_proj_proj.shp -overwrite -progress -t_srs \'{}\' data/validation/processed/Boundary_{}_proj.shp'.format(a,cartesianProjection,a))
		os.system('ogr2ogr data/validation/processed/Inundated_area_{}_proj.shp -overwrite -progress -t_srs \'{}\' data/validation/{}/Inundated_area_{}.shp'.format(a,cartesianProjection,a,a))
		os.system('gdalwarp -dstnodata 0 -t_srs \'{}\' -overwrite -of HFA data/lc/processed/lcNC_proj_{}.img data/lc/processed/lcNC_proj_{}_proj.img'.format(cartesianProjection,a,a))
		for f in glob.glob('data/validation/processed/Boundary_{}_proj.*'.format(a)): os.remove(f)
		for f in glob.glob('data/lc/processed/lcNC_proj_{}.img*'.format(a)): os.remove(f)

		os.system('ogr2ogr data/gages/processed/gage_{}_proj.shp -overwrite -progress -t_srs \'{}\' data/gages/processed/gage_{}.shp'.format(a,cartesianProjection,a))
		for f in glob.glob('data/gages/processed/gage_{}.*'.format(a)): os.remove(f)
		os.system('ogr2ogr data/validation/processed/hwm_{}_proj.shp -overwrite -progress -t_srs \'{}\' data/validation/{}/*_HWM_{}.shp'.format(a,cartesianProjection,a,a))
		os.system('ogr2ogr -overwrite -progress -t_srs \'{}\' data/nhd/processed/NHDFlowline_proj_{}_proj.shp data/nhd/processed/NHDFlowline_proj_{}.shp'.format(cartesianProjection,a,a))
		for f in glob.glob('data/nhd/processed/NHDFlowline_proj_{}.*'.format(a)): os.remove(f)


	print('.... Rescaling rasters ....')
	for a in areas:
		reference = gis.raster('data/sar/processed/vv_cal_spk_gcp_proj_{}_db_nodata_proj.tiff'.format(a))
		ulx = reference.gt[0] ; uly = reference.gt[3] ; px = reference.gt[1] ; py = reference.gt[5]
		lrx = ulx + px * reference.ncols ; lry = uly + py * reference.nrows

		os.system('gdalwarp -overwrite -te {} {} {} {} -tr {} {} -of HFA data/lc/processed/lcNC_proj_{}_proj.img data/lc/processed/lcNC_proj_{}_proj_scaled.img'.format(ulx,lry,lrx,uly,px,py,a,a))
		os.system('gdalwarp -r bilinear -overwrite -te {} {} {} {} -tr {} {} data/hand/processed/hand_proj_{}_nodata_proj.tiff data/hand/processed/hand_proj_{}_nodata_proj_scaled.tiff'.format(ulx,lry,lrx,uly,px,py,a,a))
		os.system('gdalwarp -overwrite -te {} {} {} {} -tr {} {} data/hand/processed/catchmask_proj_{}_nodata_proj.tiff data/hand/processed/catchmask_proj_{}_nodata_proj_scaled.tiff'.format(ulx,lry,lrx,uly,px,py,a,a))
		for f in glob.glob('data/lc/processed/lcNC_proj_{}_proj.img*'.format(a)): os.remove(f)
		os.remove("data/hand/processed/hand_proj_{}_nodata_proj.tiff".format(a))
		os.remove("data/hand/processed/catchmask_proj_{}_nodata_proj.tiff".format(a))

		os.system('gdalwarp -r bilinear -overwrite -cutline data/validation/processed/Boundary_{}_proj_proj.shp -crop_to_cutline data/sar/processed/vv_cal_spk_gcp_proj_{}_db_nodata_proj.tiff data/sar/processed/vv_cal_spk_gcp_proj_{}_db_nodata_proj_{}.tiff'.format(a,a,a,a))
		os.system('gdalwarp -r bilinear -overwrite -cutline data/validation/processed/Boundary_{}_proj_proj.shp -crop_to_cutline data/sar/processed/vh_cal_spk_gcp_proj_{}_db_nodata_proj.tiff data/sar/processed/vh_cal_spk_gcp_proj_{}_db_nodata_proj_{}.tiff'.format(a,a,a,a))
		os.system('gdalwarp -r bilinear -overwrite -cutline data/validation/processed/Boundary_{}_proj_proj.shp -crop_to_cutline data/hand/processed/hand_proj_{}_nodata_proj_scaled.tiff data/hand/processed/hand_proj_{}_nodata_proj_scaled_{}.tiff'.format(a,a,a,a))
		os.system('gdalwarp -overwrite -cutline data/validation/processed/Boundary_{}_proj_proj.shp -crop_to_cutline data/hand/processed/catchmask_proj_{}_nodata_proj_scaled.tiff data/hand/processed/catchmask_proj_{}_nodata_proj_scaled_{}.tiff'.format(a,a,a,a))
		exit()
		os.system('gdalwarp -overwrite -of HFA -cutline data/validation/processed/Boundary_{}_proj_proj.shp -crop_to_cutline data/lc/processed/lcNC_proj_{}_proj_scaled.img data/lc/processed/lcNC_proj_{}_proj_scaled_{}.img'.format(a,a,a,a))
		os.remove("data/sar/processed/vv_cal_spk_gcp_proj_{}_db_nodata_proj.tiff".format(a))
		os.remove("data/sar/processed/vh_cal_spk_gcp_proj_{}_db_nodata_proj.tiff".format(a))
		os.remove("data/hand/processed/hand_proj_{}_nodata_proj_scaled.tiff".format(a))
		for f in glob.glob('data/lc/processed/lcNC_proj_{}_proj_scaled.img*'.format(a)): os.remove(f)

	# stack predictors
	print('.... Stacking Predictors ....')
	for a in areas:
		if os.path.isfile('data/results/vv_vh_hand_{}.tiff'.format(a)): os.remove('data/results/vv_vh_hand_{}.tiff'.format(a))
		os.system('gdal_merge.py -separate -n {} -a_nodata {} -init {} -o data/results/vv_vh_hand_{}.tiff data/sar/processed/vv_cal_spk_gcp_proj_{}_db_nodata_proj_{}.tiff data/sar/processed/vh_cal_spk_gcp_proj_{}_db_nodata_proj_{}.tiff data/hand/processed/hand_proj_{}_nodata_proj_scaled_{}.tiff'.format(noDataValue,noDataValue,noDataValue,a,a,a,a,a,a,a))
		os.remove('data/sar/processed/vv_cal_spk_gcp_proj_{}_db_nodata_proj_{}.tiff'.format(a,a))
		os.remove('data/sar/processed/vh_cal_spk_gcp_proj_{}_db_nodata_proj_{}.tiff'.format(a,a))
		os.remove('data/hand/processed/hand_proj_{}_nodata_proj_scaled_{}.tiff'.format(a,a))

	# create raster mask of study area
	print('.... Creating inundation mask ....')
	for a in areas:
		reference = gis.raster('data/results/vv_vh_hand_{}.tiff'.format(a))
		ulx = reference.gt[0] ; uly = reference.gt[3] ; px = reference.gt[1] ; py = reference.gt[5]
		lrx = ulx + px * reference.ncols ; lry = uly + py * reference.nrows

		os.system('gdal_rasterize -burn 1 -a_nodata 0 -init 0 -ot Byte -te {} {} {} {} -ts {} {} data/validation/processed/Inundated_area_{}_proj.shp data/validation/processed/Inundation_area_{}.tiff'.format(ulx,lry,lrx,uly,reference.ncols,reference.nrows,a,a))

		inundation = gis.raster('data/validation/processed/Inundation_area_{}.tiff'.format(a))
		finalInundation = copy.copy(reference)
		finalInundation.array = reference.array[0,:,:]
		finalInundation.dim = finalInundation.array.shape
		finalInundation.nrows,finalInundation.ncols = finalInundation.dim

		reference.array = reference.array[0,:,:]
		reference.dim = reference.array.shape
		reference.nrows,reference.ncols = reference.dim

		finalInundation.array[np.all(np.stack((inundation.array == 1,reference.array != reference.ndv)),axis=0)] = 2
		finalInundation.array[np.all(np.stack((inundation.array == 0,reference.array != reference.ndv)),axis=0)] = 1
		finalInundation.array[reference.array == reference.ndv] = 0
		finalInundation.ndv = 0
		gis.writeGeotiff(finalInundation,'data/validation/processed/finalInundation_{}.tiff'.format(a),2) # 2 = gdal.GDT_UInt16
		os.remove('data/validation/processed/Inundation_area_{}.tiff'.format(a))


	# adjust raster mask for Goldsboro area due to lake
	print('.... Correcting Goldsboro inundation mask ....')
	originalInundation = gis.raster('data/validation/processed/finalInundation_Goldsboro.tiff')
	ulx = originalInundation.gt[0] ; uly = originalInundation.gt[3] ; px = originalInundation.gt[1] ; py = originalInundation.gt[5]
	lrx = ulx + px * originalInundation.ncols ; lry = uly + py * originalInundation.nrows

	os.system('gdal_rasterize -burn 1 -a_nodata 0 -init 0 -ot Byte -te {} {} {} {} -ts {} {} data/validation/processed/goldsboroLake.shp data/validation/processed/goldsboroLake.tiff'.format(ulx,lry,lrx,uly,originalInundation.ncols,originalInundation.nrows))
	goldsboroLake = gis.raster('data/validation/processed/goldsboroLake.tiff')
	finalInundation = copy.copy(originalInundation)

	finalInundation.array[np.any(np.stack((originalInundation.array == 2,goldsboroLake.array == 1)),axis=0)] = 2
	finalInundation.array[np.all(np.stack((originalInundation.array == 1,goldsboroLake.array == 0)),axis=0)] = 1

	gis.writeGeotiff(finalInundation,'data/validation/processed/finalInundation_Goldsboro_withLake.tiff',2) # 2 = gdal.GDT_UInt16
	os.rename('data/validation/processed/finalInundation_Goldsboro.tiff','data/validation/processed/finalInundation_Goldsboro_noLake.tiff')
	os.rename('data/validation/processed/finalInundation_Goldsboro_withLake.tiff','data/validation/processed/finalInundation_Goldsboro.tiff')
