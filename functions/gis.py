#!/usr/bin/env python3


from osgeo import gdal
import numpy as np

class raster:

	def __init__(self,fname):
		stream = gdal.Open(fname,gdal.GA_ReadOnly)
		self.array = stream.ReadAsArray()
		self.gt = stream.GetGeoTransform()
		self.proj = stream.GetProjection()
		self.ndv = stream.GetRasterBand(1).GetNoDataValue()
		self.dim = self.array.shape
		if len(self.dim) == 3:
				self.nbands,self.nrows,self.ncols = self.dim
		if len(self.dim) == 2:
			self.nrows,self.ncols = self.dim
		stream = None

def writeGeotiff(raster,fname,dtype,driverName='GTiff'):
	"""Create a GeoTIFF file with the given data."""
	driver = gdal.GetDriverByName(driverName)
	dataset = driver.Create(fname, raster.ncols, raster.nrows, 1, dtype)
	dataset.SetGeoTransform(raster.gt)
	dataset.SetProjection(raster.proj)
	band = dataset.GetRasterBand(1)
	band.SetNoDataValue(raster.ndv)
	band.WriteArray(raster.array)
	dataset = None  # Close the file
