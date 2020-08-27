#!/usr/bin/env python3


## calculate stats

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.markers import MarkerStyle
import sys
from os import chdir
import pandas as pd
from scipy import stats
import gis
from math import pi
from mpl_toolkits import mplot3d
from itertools import product
import pickle
import copy
from scipy.linalg import toeplitz,hankel
from scipy.ndimage.filters import gaussian_filter

def generateDifferenceRaster(predictedRaster,observedRaster,maskingRaster=None,maskingValue=None,output_fileName=None,mapping={'TP':1,'FP':2,'FN':3,'TN':4,'ndv':0}):

	if (maskingRaster is not None) | (maskingValue is not None):
		predictedRaster.array[maskingRaster.array != maskingValue] = predictedRaster.ndv
		observedRaster.array[maskingRaster.array != maskingValue] = observedRaster.ndv

	differenceRaster = copy.deepcopy(observedRaster)
	differenceRaster.array = np.zeros((differenceRaster.nrows,differenceRaster.ncols),dtype=int)

	differenceRaster.array[np.all(np.stack((predictedRaster.array==2,observedRaster.array==2)),axis=0)] = mapping['TP']
	differenceRaster.array[np.all(np.stack((predictedRaster.array==2,observedRaster.array==1)),axis=0)] = mapping['FP']
	differenceRaster.array[np.all(np.stack((predictedRaster.array==1,observedRaster.array==2)),axis=0)] = mapping['FN']
	differenceRaster.array[np.all(np.stack((predictedRaster.array==1,observedRaster.array==1)),axis=0)] = mapping['TN']
	differenceRaster.ndv = mapping['ndv']

	if output_fileName is not None:
		gis.writeGeotiff(differenceRaster,output_fileName,gdal.GDT_Byte)

	return(differenceRaster)


def calculateBinaryClassificationStatistics(differenceRaster,mapping={'TP':1,'FP':2,'FN':3,'TN':4,'ndv':0}):

	TP = (differenceRaster.array == mapping['TP']).sum()
	FP = (differenceRaster.array == mapping['FP']).sum()
	TN = (differenceRaster.array == mapping['TN']).sum()
	FN = (differenceRaster.array == mapping['FN']).sum()

	totalPopulation = TP + FP + TN + FN

	TP_perc = (TP / totalPopulation) * 100
	FP_perc = (FP / totalPopulation) * 100
	TN_perc = (TN / totalPopulation) * 100
	FN_perc = (FN / totalPopulation) * 100

	cellArea = abs(differenceRaster.gt[1] * differenceRaster.gt[5])

	TP_area = TP * cellArea
	FP_area = FP * cellArea
	TN_area = TN * cellArea
	FN_area = FN * cellArea

	totalPopulation = TP + FP + TN + FN
	predPositive = TP + FP
	predNegative = TN + FN
	obsPositive = TP + FN
	obsNegative = TN + FP

	predPositive_perc = predPositive / totalPopulation
	predNegative_perc = predNegative / totalPopulation
	obsPositive_perc = obsPositive / totalPopulation
	obsNegative_perc = obsNegative / totalPopulation

	predPositive_area = predPositive * cellArea
	predNegative_area = predNegative * cellArea
	obsPositive_area =  obsPositive * cellArea
	obsNegative_area =  obsNegative * cellArea

	positiveDiff = predPositive - obsPositive
	positiveDiff_area = predPositive_area - obsPositive_area
	positiveDiff_perc = predPositive_perc - obsPositive_perc

	prevalance = (TP + FN) / totalPopulation
	PPV = TP / predPositive
	NPV = TN / predNegative
	TPR = TP / obsPositive
	TNR = TN / obsNegative
	ACC = (TP + TN) / totalPopulation
	F1_score = (2*TP) / (2*TP + FP + FN)
	BACC = np.mean([TPR,TNR])
	MCC = (TP_area*TN_area - FP_area*FN_area)/ np.sqrt((TP_area+FP_area)*(TP_area+FN_area)*(TN_area+FP_area)*(TN_area+FN_area))
	CSI = ( ( (TPR)**-1 ) + ( (PPV)**-1 ) - 1)**-1

	stats = { 'TP' : TP,'FP' : FP,'TN' : TN,'FN' : FN,
			  'TP_perc' : TP_perc,'FP_perc' : FP_perc,
			  'TN_perc' : TN_perc,'FN_perc' : FN_perc,
			  'TP_area' : TP_area,'FP_area' : FP_area,
			  'TN_area' : TN_area,'FN_area' : FN_area,
			  'totalPopulation' : totalPopulation,
			  'predPositive' : predPositive,
			  'predNegative' : predNegative,
			  'obsPositive' : obsPositive,
			  'obsNegative' : obsNegative,
			  'prevalance' : prevalance,
			  'predPositive_perc' : predPositive_perc,
			  'predNegative_perc' : predNegative_perc,
			  'obsPositive_perc' : obsPositive_perc,
			  'obsNegative_perc' : obsNegative_perc,
			  'predPositive_area' : predPositive_area,
			  'predNegative_area' : predNegative_area,
			  'obsPositive_area' : obsPositive_area,
			  'obsNegative_area' : obsNegative_area,
			  'positiveDiff' : positiveDiff,
			  'positiveDiff_area' : positiveDiff_area,
			  'positiveDiff_perc' : positiveDiff_perc,
			  'PPV' : PPV,
			  'NPV' : NPV,
			  'TPR' : TPR,
			  'TNR' : TNR,
			  'ACC' : ACC,
			  'F1_score' : F1_score,
			  'BACC' : BACC,
			  'MCC' : MCC,
			  'CSI' : CSI}

	return(stats)

def scatterPlots3D(vv_vh_hand_data,validationRaster,area,sampleFrac,xyzLim):

	dataIndices = np.all(np.stack([np.array([vv_vh_hand_data.array[0,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[1,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[2,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[0,:,:] >= xyzLim[0]]), np.array([vv_vh_hand_data.array[0,:,:] <= xyzLim[1]]),
									np.array([vv_vh_hand_data.array[1,:,:] >= xyzLim[2]]), np.array([vv_vh_hand_data.array[1,:,:] <= xyzLim[3]]),
									np.array([vv_vh_hand_data.array[2,:,:] >= xyzLim[4]]), np.array([vv_vh_hand_data.array[2,:,:] <= xyzLim[5]])],axis=0),axis=0)
	dataIndices = dataIndices[0,:,:]
	dataIndices = np.where(dataIndices)

	indicesOfDataIndicesToSampleFrom = np.sort(np.random.choice(np.arange(len(dataIndices[0])),size=int(sampleFrac*len(dataIndices[0])),replace=False))
	rowIndices = dataIndices[0][indicesOfDataIndicesToSampleFrom]
	columnIndices = dataIndices[1][indicesOfDataIndicesToSampleFrom]
	dataToPlot = vv_vh_hand_data.array[:,rowIndices,columnIndices].T
	x,y,z = dataToPlot[:,0].ravel(),dataToPlot[:,1].ravel(),dataToPlot[:,2].ravel()
	validationLabel = validationRaster.array[rowIndices,columnIndices]

	fig1 = plt.figure()
	ax = fig1.add_subplot('111',projection='3d')

	line1 = ax.scatter3D(x[validationLabel == 1], y[validationLabel == 1], z[validationLabel == 1],c='brown',alpha=0.1)
	line2 = ax.scatter3D(x[validationLabel == 2], y[validationLabel == 2], z[validationLabel == 2],c='blue',alpha=0.1)

	ax.set_title("{} Features By Class".format(area),pad=10)
	ax.set_xlabel("VV (dB)", rotation=0, size='large',labelpad=10)
	ax.set_ylabel("VH (dB)", rotation=0, size='large',labelpad=10)
	ax.set_zlabel("HAND (m)", rotation=90, size='large',labelpad=0)

	ax.set_xlim(xyzLim[0:2])
	ax.set_ylim(xyzLim[2:4])
	ax.set_zlim(xyzLim[4:6])

	ax.xaxis.set_ticks(np.arange(xyzLim[0], xyzLim[1], 5))
	ax.yaxis.set_ticks(np.arange(xyzLim[2], xyzLim[3], 5))
	ax.zaxis.set_ticks(np.arange(xyzLim[4],xyzLim[5],5))

	#print(ax.azim,ax.elev)
	ax.view_init(elev=15, azim=-45)

	fig1.legend((line1,line2),("Non-Inundated","Inundated"),'upper right')
	fig1.tight_layout()
	# fig1.savefig('manuscript/figures/3d_scatterPlot_{}.jpeg'.format(area),format='jpeg',dpi=300,quality=100)
	fig1.savefig('/media/lqc/fernandoa_si2018/archive_not_on_lqc/hand/3d_scatterPlot_{}.jpeg'.format(area),format='jpeg',dpi=300,quality=100)
	plt.show()

	return()

def contourPlots(vv_vh_hand_data,validationRaster,area,sampleFrac,xyzLim):
	dataIndices = np.all(np.stack([np.array([vv_vh_hand_data.array[0,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[1,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[2,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[0,:,:] >= xyzLim[0]]), np.array([vv_vh_hand_data.array[0,:,:] <= xyzLim[1]]),
									np.array([vv_vh_hand_data.array[1,:,:] >= xyzLim[2]]), np.array([vv_vh_hand_data.array[1,:,:] <= xyzLim[3]]),
									np.array([vv_vh_hand_data.array[2,:,:] >= xyzLim[4]]), np.array([vv_vh_hand_data.array[2,:,:] <= xyzLim[5]])],axis=0),axis=0)
	dataIndices = dataIndices[0,:,:]
	dataIndices = np.where(dataIndices)

	indicesOfDataIndicesToSampleFrom = np.sort(np.random.choice(np.arange(len(dataIndices[0])),size=int(sampleFrac*len(dataIndices[0])),replace=False))
	rowIndices = dataIndices[0][indicesOfDataIndicesToSampleFrom]
	columnIndices = dataIndices[1][indicesOfDataIndicesToSampleFrom]
	dataToPlot = vv_vh_hand_data.array[:,rowIndices,columnIndices].T
	x,y,z = dataToPlot[:,0].ravel(),dataToPlot[:,1].ravel(),dataToPlot[:,2].ravel()
	validationLabel = validationRaster.array[rowIndices,columnIndices]
	# z = hankel(z)

	fig1 = plt.figure()
	ax = fig1.add_subplot('111')

	# line1 = ax.scatter(x[validationLabel == 1], y[validationLabel == 1],c='brown',alpha=0.9)
	# line2 = ax.scatter(x[validationLabel == 2], y[validationLabel == 2],c='blue',alpha=0.9)

	indicesOfDataIndicesToSampleFrom = np.sort(np.random.choice(np.arange(len(dataIndices[0])),size=int(1*len(dataIndices[0])),replace=False))
	rowIndices = dataIndices[0][indicesOfDataIndicesToSampleFrom]
	columnIndices = dataIndices[1][indicesOfDataIndicesToSampleFrom]
	dataToPlot = vv_vh_hand_data.array[:,rowIndices,columnIndices].T
	# dataToPlot[:,2] = gaussian_filter(vv_vh_hand_data.array[2,rowIndices,columnIndices],1)
	x,y,z = dataToPlot[:,0].ravel(),dataToPlot[:,1].ravel(),dataToPlot[:,2].ravel()
	# x,y,z = dataToPlot[:,0].ravel(),dataToPlot[:,1].ravel(),z.ravel()

	# z = gaussian_filter(z,10)
	conts = ax.tricontourf(x,y,z,levels=list(range(21)),alpha=0.95)
	ax.set_xlabel('VV (dB)')
	ax.set_ylabel('VH (dB)')

	cbar = fig1.colorbar(conts, ax=ax)
	cbar.set_label('HAND (m)')
	fig1.savefig('manuscript/reviews/r2/contour_plot_{}.jpeg'.format(area),format='jpeg',dpi=300)

	ax.set_xlim(xyzLim[0:2])
	ax.set_ylim(xyzLim[2:4])
	plt.show()


def scatterPlots(vv_vh_hand_data,area,validationRaster,sampleFrac,xyzLim):

	dataIndices = np.all(np.stack([np.array([vv_vh_hand_data.array[0,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[1,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[2,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[0,:,:] >= xyzLim[0]]), np.array([vv_vh_hand_data.array[0,:,:] <= xyzLim[1]]),
									np.array([vv_vh_hand_data.array[1,:,:] >= xyzLim[2]]), np.array([vv_vh_hand_data.array[1,:,:] <= xyzLim[3]]),
									np.array([vv_vh_hand_data.array[2,:,:] >= xyzLim[4]]), np.array([vv_vh_hand_data.array[2,:,:] <= xyzLim[5]])],axis=0),axis=0)
	dataIndices = dataIndices[0,:,:]
	dataIndices = np.where(dataIndices)

	indicesOfDataIndicesToSampleFrom = np.sort(np.random.choice(np.arange(len(dataIndices[0])),size=int(sampleFrac*len(dataIndices[0])),replace=False))
	rowIndices = dataIndices[0][indicesOfDataIndicesToSampleFrom]
	columnIndices = dataIndices[1][indicesOfDataIndicesToSampleFrom]
	dataToPlot = vv_vh_hand_data.array[:,rowIndices,columnIndices].T
	vv,vh,hand = dataToPlot[:,0].ravel(),dataToPlot[:,1].ravel(),dataToPlot[:,2].ravel()
	validationLabel = validationRaster.array[rowIndices,columnIndices]
	#landcoverLabel = landcoverRaster.array[rowIndices,columnIndices]
	#print(np.unique(landcoverLabel))

	fig1 = plt.figure()
	ax = fig1.add_subplot('111')

	line1 = ax.scatter(vv[validationLabel == 1], vh[validationLabel == 1],c='brown',alpha=0.15)
	line2 = ax.scatter(vv[validationLabel == 2], vh[validationLabel == 2],c='blue',alpha=0.15)

	ax.set_xlabel("VV (dB)", rotation=0, size='large',labelpad=10)
	ax.set_ylabel("VH (dB)", rotation=90, size='large',labelpad=10)

	ax.set_xlim(xyzLim[0:2])
	ax.set_ylim(xyzLim[2:4])

	ax.xaxis.set_ticks(np.arange(xyzLim[0], xyzLim[1], 5))
	ax.yaxis.set_ticks(np.arange(xyzLim[2], xyzLim[3], 5))

	fig1.legend((line1,line2),("Non-Inundated","Inundated"),'upper right')
	# fig1.savefig('manuscript/figures/2d_scatterPlot_vv_vh_{}_{}.tiff'.format(area,cl),format='tiff',dpi=300)

	fig2 = plt.figure()
	ax = fig2.add_subplot('111')

	line1 = ax.scatter(vv[validationLabel == 1], hand[validationLabel == 1],c='brown',alpha=0.15)
	line2 = ax.scatter(vv[validationLabel == 2], hand[validationLabel == 2],c='blue',alpha=0.15)

	ax.set_xlabel("VV (dB)", rotation=0, size='large',labelpad=10)
	ax.set_ylabel("HAND (m)", rotation=90, size='large',labelpad=10)

	ax.set_xlim(xyzLim[0:2])
	ax.set_ylim(xyzLim[4:6])

	ax.xaxis.set_ticks(np.arange(xyzLim[0], xyzLim[1], 5))
	ax.yaxis.set_ticks(np.arange(xyzLim[4], xyzLim[5], 5))

	fig2.legend((line1,line2),("Non-Inundated","Inundated"),'upper right')
	# fig2.savefig('manuscript/figures/2d_scatterPlot_vv_hand_{}_{}.tiff'.format(area,cl),format='tiff',dpi=300)

	fig3 = plt.figure()
	ax = fig3.add_subplot('111')

	line1 = ax.scatter(vh[validationLabel == 1], hand[validationLabel == 1],c='brown',alpha=0.15)
	line2 = ax.scatter(vh[validationLabel == 2], hand[validationLabel == 2],c='blue',alpha=0.15)

	ax.set_xlabel("VH (dB)", rotation=0, size='large',labelpad=10)
	ax.set_ylabel("HAND (m)", rotation=90, size='large',labelpad=10)

	ax.set_xlim(xyzLim[2:4])
	ax.set_ylim(xyzLim[4:6])

	ax.xaxis.set_ticks(np.arange(xyzLim[2], xyzLim[3], 5))
	ax.yaxis.set_ticks(np.arange(xyzLim[4], xyzLim[5], 5))

	fig3.legend((line1,line2),("Non-Inundated","Inundated"),'upper right')
	# fig3.savefig('manuscript/figures/2d_scatterPlot_vh_hand_{}_{}.tiff'.format(area,cl),format='tiff',dpi=300)


	plt.show()

	return()

#def scatterPlots_subplots(vv_vh_hand_data,area,classificationModels,xyzLim,predictor):
def scatterPlots_subplots(areas,sampleFrac,xyzLim):

	sarDataLabels = ['VV','VH']

	fig, axs = plt.subplots(figsize=(4.75,7),nrows=len(areas), ncols=len(sarDataLabels),sharex=True, sharey=True)

	for r,row in enumerate(axs):

		area = areas[r]

		vv_vh_hand_filename = 'data/results/vv_vh_hand_{}.tiff'.format(area)
		validationRasterFileName = 'data/validation/processed/finalInundation_{}.tiff'.format(area)

		vv_vh_hand_data = gis.raster(vv_vh_hand_filename)
		validation = gis.raster(validationRasterFileName)

		dataIndices = np.all(np.stack([np.array([vv_vh_hand_data.array[0,:,:] != vv_vh_hand_data.ndv]),
										np.array([vv_vh_hand_data.array[1,:,:] != vv_vh_hand_data.ndv]),
										np.array([vv_vh_hand_data.array[2,:,:] != vv_vh_hand_data.ndv]),
										np.array([vv_vh_hand_data.array[0,:,:] >= xyzLim[0]]), np.array([vv_vh_hand_data.array[0,:,:] <= xyzLim[1]]),
										np.array([vv_vh_hand_data.array[1,:,:] >= xyzLim[2]]), np.array([vv_vh_hand_data.array[1,:,:] <= xyzLim[3]]),
										np.array([vv_vh_hand_data.array[2,:,:] >= xyzLim[4]]), np.array([vv_vh_hand_data.array[2,:,:] <= xyzLim[5]])],axis=0),axis=0)
		dataIndices = dataIndices[0,:,:]
		dataIndices = np.where(dataIndices)

		sf = sampleFrac[area]

		indicesOfDataIndicesToSampleFrom = np.sort(np.random.choice(np.arange(len(dataIndices[0])),size=int(sf*len(dataIndices[0])),replace=False))
		rowIndices = dataIndices[0][indicesOfDataIndicesToSampleFrom]
		columnIndices = dataIndices[1][indicesOfDataIndicesToSampleFrom]
		dataToPlot = vv_vh_hand_data.array[:,rowIndices,columnIndices].T
		vv,vh,hand = dataToPlot[:,0].ravel(),dataToPlot[:,1].ravel(),dataToPlot[:,2].ravel()
		validationLabel = validation.array[rowIndices,columnIndices]

		for c,ax in enumerate(row):

			sarDataLabel = sarDataLabels[c]
			xData = {'VV' : vv , 'VH' : vh}[sarDataLabel]

			line1 = ax.scatter(xData[validationLabel == 1], hand[validationLabel == 1],c='red',alpha=0.18,marker='$o$',s=5)
			line2 = ax.scatter(xData[validationLabel == 2], hand[validationLabel == 2],c='blue',alpha=0.1,marker='$x$',s=5)

			if r == (len(areas)-1):
				ax.set_xlabel("{} (dB)".format(sarDataLabel), rotation=0, size='medium',labelpad=5)
			if c == 0:
				ax.set_ylabel("HAND (m)", rotation=90, size='medium',labelpad=5)
			if c == 0:
				 text = ax.text(-36,12.5,area, size='large',verticalalignment='center', rotation=90)

			ax.set_xlim(xyzLim[0:2])
			ax.set_ylim(xyzLim[4:6])
			ax.set_aspect(abs(xyzLim[1]-xyzLim[0])/abs(xyzLim[4]-xyzLim[5]))

			ax.xaxis.set_ticks(np.arange(xyzLim[0], xyzLim[1], 5))
			ax.yaxis.set_ticks(np.arange(xyzLim[4], xyzLim[5], 5))

	leg = fig.legend((line1,line2),("Non-Inundated","Inundated"),fontsize='medium',
		bbox_to_anchor=(0.875,0.07),ncol=2,markerscale=1.5)

	for handle in leg.legendHandles:
		handle.set_alpha(0.5)

	plt.tight_layout(rect=(0.06,0.05,1,1),h_pad=0.1,w_pad=0.1)
	fig.subplots_adjust(wspace=0.05, hspace=0.05)
	fig.savefig('manuscript/reviews/r2/2d_scatterPlot.jpeg',format='jpeg',dpi=300,quality=100)

	plt.show()

	return()



def diffScatterPlots(vv_vh_hand_data,area,classificationModels,xyzLim,predictor):

	dataIndices = np.all(np.stack([np.array([vv_vh_hand_data.array[0,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[1,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[2,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[0,:,:] >= xyzLim[0]]), np.array([vv_vh_hand_data.array[0,:,:] <= xyzLim[1]]),
									np.array([vv_vh_hand_data.array[1,:,:] >= xyzLim[2]]), np.array([vv_vh_hand_data.array[1,:,:] <= xyzLim[3]]),
									np.array([vv_vh_hand_data.array[2,:,:] >= xyzLim[4]]), np.array([vv_vh_hand_data.array[2,:,:] <= xyzLim[5]])],axis=0),axis=0)
	dataIndices = dataIndices[0,:,:]
	dataIndices = np.where(dataIndices)

	indicesOfDataIndicesToSampleFrom = np.sort(np.random.choice(np.arange(len(dataIndices[0])),size=int(0.002*len(dataIndices[0])),replace=False))
	rowIndices = dataIndices[0][indicesOfDataIndicesToSampleFrom]
	columnIndices = dataIndices[1][indicesOfDataIndicesToSampleFrom]
	dataToPlot = vv_vh_hand_data.array[:,rowIndices,columnIndices].T
	vv,vh,hand = dataToPlot[:,0].ravel(),dataToPlot[:,1].ravel(),dataToPlot[:,2].ravel()
	sarDataLabels = ['VV','VH']

	fig, axs = plt.subplots(figsize=(4.75,7),nrows=len(classificationModels), ncols=len(sarDataLabels),sharex=True, sharey=True)

	for r,row in enumerate(axs):

		cl = classificationModels[r]

		diffRaster_filename = 'data/results/diff_{}_{}_{}.tif'.format(predictor,area,cl)
		diffRaster = gis.raster(diffRaster_filename)
		diffLabel = diffRaster.array[rowIndices,columnIndices]

		for c,ax in enumerate(row):

			sarDataLabel = sarDataLabels[c]
			xData = {'VV' : vv , 'VH' : vh}[sarDataLabel]

			line1 = ax.scatter(xData[diffLabel == 1], hand[diffLabel == 1],c='blue',alpha=0.1,marker='$x$',s=7)
			line2 = ax.scatter(xData[diffLabel == 2], hand[diffLabel == 2],c='red',alpha=0.1,marker='$x$',s=7)
			line3 = ax.scatter(xData[diffLabel == 3], hand[diffLabel == 3],c='black',alpha=0.18,marker='$o$',s=7)
			line4 = ax.scatter(xData[diffLabel == 4], hand[diffLabel == 4],c='orange',alpha=0.18,marker='$o$',s=7)


			if r == (len(classificationModels)-1):
				ax.set_xlabel("{} (dB)".format(sarDataLabel), rotation=0, size='medium',labelpad=5)
			if c == 0:
				ax.set_ylabel("HAND (m)", rotation=90, size='medium',labelpad=5)
			if c == 0:
				 text = ax.text(-36,12.5,cl.upper(), size='large',verticalalignment='center', rotation=90)

			ax.set_xlim(xyzLim[0:2])
			ax.set_ylim(xyzLim[4:6])
			ax.set_aspect(abs(xyzLim[1]-xyzLim[0])/abs(xyzLim[4]-xyzLim[5]))

			ax.xaxis.set_ticks(np.arange(xyzLim[0], xyzLim[1], 5))
			ax.yaxis.set_ticks(np.arange(xyzLim[4], xyzLim[5], 5))

	leg = fig.legend((line1,line2,line3,line4),("TP","FP","FN","TN"),fontsize='medium',
		bbox_to_anchor=(0.75,0.12),ncol=2,markerscale=1.5)

	for handle in leg.legendHandles:
		handle.set_alpha(0.5)

	plt.tight_layout(rect=(0.06,0.1,1,1),h_pad=0.1,w_pad=0.1)
	fig.subplots_adjust(wspace=0.05, hspace=0.05)
	fig.savefig('manuscript/figures/2d_diffScatterPlot_{}_{}_v2.jpeg'.format(area,predictor),format='jpeg',dpi=300,quality=100)

	plt.show()

	return()

"""

def diffScatterPlots(vv_vh_hand_data,area,classificationModels,xyzLim,predictor):

	dataIndices = np.all(np.stack([np.array([vv_vh_hand_data.array[0,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[1,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[2,:,:] != vv_vh_hand_data.ndv]),
									np.array([vv_vh_hand_data.array[0,:,:] >= xyzLim[0]]), np.array([vv_vh_hand_data.array[0,:,:] <= xyzLim[1]]),
									np.array([vv_vh_hand_data.array[1,:,:] >= xyzLim[2]]), np.array([vv_vh_hand_data.array[1,:,:] <= xyzLim[3]]),
									np.array([vv_vh_hand_data.array[2,:,:] >= xyzLim[4]]), np.array([vv_vh_hand_data.array[2,:,:] <= xyzLim[5]])],axis=0),axis=0)
	dataIndices = dataIndices[0,:,:]
	dataIndices = np.where(dataIndices)

	indicesOfDataIndicesToSampleFrom = np.sort(np.random.choice(np.arange(len(dataIndices[0])),size=int(0.005*len(dataIndices[0])),replace=False))
	rowIndices = dataIndices[0][indicesOfDataIndicesToSampleFrom]
	columnIndices = dataIndices[1][indicesOfDataIndicesToSampleFrom]
	dataToPlot = vv_vh_hand_data.array[:,rowIndices,columnIndices].T
	vv,vh,hand = dataToPlot[:,0].ravel(),dataToPlot[:,1].ravel(),dataToPlot[:,2].ravel()


	for cl in classificationModels:
		diffRaster_filename = 'data/results/diff_{}_{}_{}.tif'.format(predictor,area,cl)
		diffRaster = gis.raster(diffRaster_filename)

		predictedRasterFileName = 'data/results/predictedInundation_three_{}_{}.tif'.format(area,cl)
		validationRaster = gis.raster(predictedRasterFileName)

		diffLabel = diffRaster.array[rowIndices,columnIndices]

		fig1 = plt.figure()
		ax = fig1.add_subplot('111')

		line1 = ax.scatter(vv[diffLabel == 1], vh[diffLabel == 1],c='blue',alpha=0.3,marker='$TP$',s=100)
		line2 = ax.scatter(vv[diffLabel == 2], vh[diffLabel == 2],c='red',alpha=0.3,marker='$FP$',s=100)
		line3 = ax.scatter(vv[diffLabel == 3], vh[diffLabel == 3],c='black',alpha=0.3,marker='$FN$',s=100)
		line4 = ax.scatter(vv[diffLabel == 4], vh[diffLabel == 4],c='orange',alpha=0.3,marker='$TN$',s=100)

		#ax.set_title("{} {} Prediction".format(area,cl),pad=10)
		ax.set_xlabel("VV (dB)", rotation=0, size='x-large',labelpad=10)
		ax.set_ylabel("VH (dB)", rotation=90, size='x-large',labelpad=10)

		ax.set_xlim(xyzLim[0:2])
		ax.set_ylim(xyzLim[2:4])

		ax.xaxis.set_ticks(np.arange(xyzLim[0], xyzLim[1], 5))
		ax.yaxis.set_ticks(np.arange(xyzLim[2], xyzLim[3], 5))

		fig1.legend((line1,line2,line3,line4),("TP","FP","FN","TN"), fontsize='x-large',bbox_to_anchor=(0.32,0.97))

		plt.tight_layout()
		fig1.savefig('manuscript/figures/2d_diffScatterPlot_vv_vh_{}_{}_{}.jpeg'.format(area,cl,predictor),format='jpeg',dpi=300,quality=100)

		fig2 = plt.figure()
		ax = fig2.add_subplot('111')

		line1 = ax.scatter(vv[diffLabel == 1], hand[diffLabel == 1],c='blue',alpha=0.3,marker='$TP$',s=100)
		line2 = ax.scatter(vv[diffLabel == 2], hand[diffLabel == 2],c='red',alpha=0.3,marker='$FP$',s=100)
		line3 = ax.scatter(vv[diffLabel == 3], hand[diffLabel == 3],c='black',alpha=0.3,marker='$FN$',s=100)
		line4 = ax.scatter(vv[diffLabel == 4], hand[diffLabel == 4],c='orange',alpha=0.3,marker='$TN$',s=100)

		#ax.set_title("{} {} Prediction".format(area,cl),pad=10)
		ax.set_xlabel("VV (dB)", rotation=0, size='x-large',labelpad=10)
		ax.set_ylabel("HAND (m)", rotation=90, size='x-large',labelpad=10)

		ax.set_xlim(xyzLim[0:2])
		ax.set_ylim(xyzLim[4:6])

		ax.xaxis.set_ticks(np.arange(xyzLim[0], xyzLim[1], 5))
		ax.yaxis.set_ticks(np.arange(xyzLim[4], xyzLim[5], 5))

		fig2.legend((line1,line2,line3,line4),("TP","FP","FN","TN"), fontsize='x-large',bbox_to_anchor=(0.32,0.97))
		plt.tight_layout()
		fig2.savefig('manuscript/figures/2d_diffScatterPlot_vv_hand_{}_{}_{}.jpeg'.format(area,cl,predictor),format='jpeg',dpi=300,quality=100)


		fig3 = plt.figure()
		ax = fig3.add_subplot('111')

		line1 = ax.scatter(vh[diffLabel == 1], hand[diffLabel == 1],c='blue',alpha=0.3,marker='$TP$',s=100)
		line2 = ax.scatter(vh[diffLabel == 2], hand[diffLabel == 2],c='red',alpha=0.3,marker='$FP$',s=100)
		line3 = ax.scatter(vh[diffLabel == 3], hand[diffLabel == 3],c='black',alpha=0.3,marker='$FN$',s=100)
		line4 = ax.scatter(vh[diffLabel == 4], hand[diffLabel == 4],c='orange',alpha=0.3,marker='$TN$',s=100)

		#ax.set_title("{} {} Prediction".format(area,cl),pad=10)
		ax.set_xlabel("VH (dB)", rotation=0, size='x-large',labelpad=10)
		ax.set_ylabel("HAND (m)", rotation=90, size='x-large',labelpad=10)

		ax.set_xlim(xyzLim[2:4])
		ax.set_ylim(xyzLim[4:6])

		ax.xaxis.set_ticks(np.arange(xyzLim[2], xyzLim[3], 5))
		ax.yaxis.set_ticks(np.arange(xyzLim[4], xyzLim[5], 5))

		fig3.legend((line1,line2,line3,line4),("TP","FP","FN","TN"), fontsize='x-large',bbox_to_anchor=(0.32,0.97))
		plt.tight_layout()
		fig3.savefig('manuscript/figures/2d_diffScatterPlot_vh_hand_{}_{}_{}.jpeg'.format(area,cl,predictor),format='jpeg',dpi=300,quality=100)


		plt.show()

	return()

"""


def barplotByArea():

	statsDF = pd.read_csv('data/results/stats_with_csi.csv')
	primaryStats = ['TP','FP','FN','TN']
	secondaryStats = ['ACC','CSI','TPR','TNR','PPV','NPV']
	numberOfSecondaryStats = len(secondaryStats)

	# allPrimaryStats = primaryStats + ['TP_area','FP_area','FN_area','TN_area']
	means = statsDF[:][statsDF['group']=='Overall'].groupby(['area','predictor']).sum()[primaryStats]
	overallMeans = statsDF[:][statsDF['group']=='Overall'].groupby(['predictor']).sum()[primaryStats]
	means = pd.concat([means, pd.concat([overallMeans], keys=["Overall"])])
	means = means.reindex(['Smithfield', 'Goldsboro', 'Kinston', 'Overall'], level='area')
	means = means.reindex(['two','three'],level='predictor')

	meanSecondaries = means.copy()
	for i,col in enumerate(secondaryStats):
		meanSecondaries[col] = np.zeros(len(meanSecondaries),dtype=float)
	for i in primaryStats:
		meanSecondaries = meanSecondaries.drop(i,1)

	# means[['TP_area','FP_area','FN_area','TN_area']] = means[['TP_area','FP_area','FN_area','TN_area']]/10**5
	print(means)

	for R,r in meanSecondaries.iterrows():

		TP = means.loc[R,'TP'] ; FP = means.loc[R,'FP'] ; TN = means.loc[R,'TN'] ; FN = means.loc[R,'FN']

		totalPopulation = TP + FP + TN + FN
		predPositive = TP + FP
		predNegative = TN + FN
		obsPositive = TP + FN
		obsNegative = TN + FP

		meanSecondaries.loc[R,'PPV'] = TP / predPositive
		meanSecondaries.loc[R,'NPV'] = TN / predNegative
		meanSecondaries.loc[R,'TPR'] = TP / obsPositive
		meanSecondaries.loc[R,'TNR'] = TN / obsNegative
		meanSecondaries.loc[R,'ACC'] = (TP + TN) / totalPopulation
		meanSecondaries.loc[R,'CSI'] = ( ( (meanSecondaries.loc[R,'TPR'])**-1 ) + ( (meanSecondaries.loc[R,'PPV'])**-1 ) - 1)**-1

	models = statsDF[['area','predictor','model']+primaryStats][statsDF['group'] == 'Overall'].reset_index(drop=True)
	overallModels = models[:][:].groupby(['predictor','model']).sum().reset_index()
	overallModels['area'] = pd.Series(np.repeat('Overall',len(overallModels)), index=overallModels.index)
	models = pd.concat([models, overallModels],sort=True)

	models.predictor = models.predictor.astype('category')
	models.predictor.cat.set_categories(['two','three'],inplace=True)

	models.area = models.area.astype('category')
	models.area.cat.set_categories(['Smithfield', 'Goldsboro', 'Kinston', 'Overall'],inplace=True)

	models.model = models.model.astype('category')
	models.model.cat.set_categories(['qda','svm','knn'],inplace=True)

	models = models.sort_values(['area','model','predictor']).reset_index()
	models = models[['area','model','predictor']+primaryStats]

	modelSecondaries = models.copy()
	for i,col in enumerate(secondaryStats):
		modelSecondaries[col] = np.zeros(len(modelSecondaries),dtype=float)
	for i in primaryStats:
		modelSecondaries = modelSecondaries.drop(i,1)

	for R,r in modelSecondaries.iterrows():

		TP = models.loc[R,'TP'] ; FP = models.loc[R,'FP'] ; TN = models.loc[R,'TN'] ; FN = models.loc[R,'FN']

		totalPopulation = TP + FP + TN + FN
		predPositive = TP + FP
		predNegative = TN + FN
		obsPositive = TP + FN
		obsNegative = TN + FP

		modelSecondaries.loc[R,'PPV'] = TP / predPositive
		modelSecondaries.loc[R,'NPV'] = TN / predNegative
		modelSecondaries.loc[R,'TPR'] = TP / obsPositive
		modelSecondaries.loc[R,'TNR'] = TN / obsNegative
		modelSecondaries.loc[R,'ACC'] = (TP + TN) / totalPopulation
		modelSecondaries.loc[R,'CSI'] = ( ( (modelSecondaries.loc[R,'TPR'])**-1 ) + ( (modelSecondaries.loc[R,'PPV'])**-1 ) - 1)**-1

	print(meanSecondaries)
	print(modelSecondaries)

	# create plot
	fig, ax = plt.subplots(figsize=(12,8))
	index = np.arange(len(meanSecondaries))
	bar_width = 0.11
	colors = ['b','g','r','m','y','k']

	for i,metric in enumerate(secondaryStats):
		plt.bar(index + i*bar_width, meanSecondaries[metric], bar_width,alpha=0.5,color=colors[i],label=metric,zorder=5)

	allIndices = np.array(list())
	for i in index:
		allIndices = np.hstack((allIndices,i + bar_width * np.arange(6)))

	points = ['.','^','*']
	for i,m in enumerate(['qda','svm','knn']):
		y = modelSecondaries[:][modelSecondaries['model']== m].as_matrix(columns=secondaryStats).ravel()
		plt.scatter(allIndices,y,color=(0,0,0),label=['QDA','SVM','KNN'][i],marker=points[i],s=55,zorder=10)

	#plt.xlabel('M',labelpad=0.05)
	plt.ylabel('Metric Values',labelpad=5,size='xx-large')
	ax.set_yticks(np.arange(0,1.1,.1))
	ax.set_yticks(np.arange(0.1,1,.1),minor=True)
	ax.set_yticklabels(np.round(np.arange(0,1,.1),1), fontdict={'fontsize':'x-large'})
	plt.grid(b=True,which='both',axis='y',zorder=-1,alpha=0.25)
	#plt.title('Scores by person')
	plt.xticks(index + 2*bar_width, ['VV,VH','VV,VH,HAND','VV,VH','VV,VH,HAND','VV,VH','VV,VH,HAND','VV,VH','VV,VH,HAND'],size='x-large')

	# texts
	plt.text(ax.get_xticks()[1]/(ax.get_xlim()[1]-ax.get_xlim()[0]), -0.1,'Smithfield', ha='center', transform=ax.transAxes,size='xx-large')
	plt.text(ax.get_xticks()[3]/(ax.get_xlim()[1]-ax.get_xlim()[0]), -0.1,'Goldsboro', ha='center', transform=ax.transAxes,size='xx-large')
	plt.text(ax.get_xticks()[5]/(ax.get_xlim()[1]-ax.get_xlim()[0]), -0.1,'Kinston', ha='center', transform=ax.transAxes,size='xx-large')
	plt.text(ax.get_xticks()[7]/(ax.get_xlim()[1]-ax.get_xlim()[0]), -0.1,'Overall', ha='center', transform=ax.transAxes,size='xx-large')

	# add lines
	x = [ax.get_xticks()[1] + (ax.get_xticks()[2]-ax.get_xticks()[1])/2,ax.get_xticks()[1] + (ax.get_xticks()[2]-ax.get_xticks()[1])/2]
	y = [-0.01,-0.12]
	line = plt.Line2D(x, y,color='black')
	line.set_clip_on(False)
	ax.add_line(line)

	x = [ax.get_xticks()[3] + (ax.get_xticks()[4]-ax.get_xticks()[3])/2,ax.get_xticks()[3] + (ax.get_xticks()[4]-ax.get_xticks()[3])/2]
	y = [-0.01,-0.12]
	line = plt.Line2D(x, y,color='black')
	line.set_clip_on(False)
	ax.add_line(line)

	x = [ax.get_xticks()[5] + (ax.get_xticks()[6]-ax.get_xticks()[5])/2,ax.get_xticks()[5] + (ax.get_xticks()[6]-ax.get_xticks()[5])/2]
	y = [-0.01,-0.12]
	line = plt.Line2D(x, y,color='black')
	line.set_clip_on(False)
	ax.add_line(line)

	plt.minorticks_on()
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),mode="expand", loc=3,ncol=9, borderaxespad=0.,fontsize='xx-large',markerscale=1.5,handletextpad=0.15)

	plt.tight_layout(rect=(0,0.06,1,1))

	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())

	fig.savefig('manuscript/figures/barPlotByArea_weightedAverage_with_csi.jpeg',format='jpeg',dpi=300,quality=100)
	plt.show()

def barplotByLC():

	statsDF = pd.read_csv('data/results/stats_with_csi.csv')
	primaryStats = ['TP','FP','FN','TN']
	secondaryStats = ['ACC','TPR','TNR','PPV','NPV','CSI']
	numberOfSecondaryStats = len(secondaryStats)
	overallOnly = True

	means = statsDF[:][statsDF['group']!='Overall'].groupby(['area','group','predictor']).sum()[primaryStats]
	overallMeans = statsDF[:][statsDF['group']!='Overall'].groupby(['group','predictor']).sum()[primaryStats]
	means = pd.concat([means, pd.concat([overallMeans], keys=["Overall"])])
	means = means.reindex(['Smithfield', 'Goldsboro', 'Kinston', 'Overall'], level='area')
	means = means.reindex(['Agriculture','Canopy','Developed'],level='group')
	meanSecondaries = means.copy()
	for col in meanSecondaries:
		meanSecondaries[col] = np.zeros(len(meanSecondaries),dtype=float)
	# meanSecondaries[secondaryStats[len(secondaryStats)-1]] = np.zeros(len(meanSecondaries),dtype=float)
	meanSecondaries['foo1'] = pd.Series(np.zeros(len(meanSecondaries),dtype=float))
	meanSecondaries['foo2'] = pd.Series(np.zeros(len(meanSecondaries),dtype=float))
	meanSecondaries.columns = secondaryStats
	# print(meanSecondaries);exit()

	for R,r in meanSecondaries.iterrows():

		TP = means.loc[R,'TP'] ; FP = means.loc[R,'FP'] ; TN = means.loc[R,'TN'] ; FN = means.loc[R,'FN']

		totalPopulation = TP + FP + TN + FN
		predPositive = TP + FP
		predNegative = TN + FN
		obsPositive = TP + FN
		obsNegative = TN + FP

		meanSecondaries.loc[R,'PPV'] = TP / predPositive
		meanSecondaries.loc[R,'NPV'] = TN / predNegative
		meanSecondaries.loc[R,'TPR'] = TP / obsPositive
		meanSecondaries.loc[R,'TNR'] = TN / obsNegative
		meanSecondaries.loc[R,'ACC'] = (TP + TN) / totalPopulation
		meanSecondaries.loc[R,'CSI'] = ( ( (meanSecondaries.loc[R,'TPR'])**-1 ) + ( (meanSecondaries.loc[R,'PPV'])**-1 ) - 1)**-1

	meanSecondaries = meanSecondaries.xs('three',level='predictor')-meanSecondaries.xs('two',level='predictor')

	models = statsDF[['area','predictor','model','group']+primaryStats][statsDF['group'] != 'Overall'].reset_index(drop=True)
	#modelsColumns = models[['area','model','group']][models['predictor']=='three']

	#models = pd.DataFrame(models[secondaryStats][models['predictor']=='three'].as_matrix()-models[secondaryStats][models['predictor']=='two'].as_matrix(),columns=secondaryStats)
	#models = pd.concat([modelsColumns.reset_index(drop=True),models],axis=1)

	overallModels = models[:][:].groupby(['model','group','predictor']).sum().reset_index()
	overallModels['area'] = pd.Series(np.repeat('Overall',len(overallModels)), index=overallModels.index)

	models = pd.concat([models, overallModels])

	models.group = models.group.astype('category')
	models.group.cat.set_categories(['Agriculture','Canopy','Developed'],inplace=True)

	models.area = models.area.astype('category')
	models.area.cat.set_categories(['Smithfield', 'Goldsboro', 'Kinston', 'Overall'],inplace=True)

	models.model = models.model.astype('category')
	models.model.cat.set_categories(['qda','svm','knn'],inplace=True)

	models.predictor = models.predictor.astype('category')
	models.predictor.cat.set_categories(['three','two'],inplace=True)

	models = models.sort_values(['area','model','group','predictor']).reset_index()
	models = models[['area','group','model','predictor']+primaryStats]

	secondaryModels = models.copy()
	secondaryModels = secondaryModels.drop(primaryStats,1)
	for col in secondaryStats:
		secondaryModels[col] = np.zeros(len(secondaryModels),dtype=float)

	for R,r in secondaryModels.iterrows():

		TP = models.loc[R,'TP'] ; FP = models.loc[R,'FP'] ; TN = models.loc[R,'TN'] ; FN = models.loc[R,'FN']

		totalPopulation = TP + FP + TN + FN
		predPositive = TP + FP
		predNegative = TN + FN
		obsPositive = TP + FN
		obsNegative = TN + FP

		secondaryModels.loc[R,'PPV'] = TP / predPositive
		secondaryModels.loc[R,'NPV'] = TN / predNegative
		secondaryModels.loc[R,'TPR'] = TP / obsPositive
		secondaryModels.loc[R,'TNR'] = TN / obsNegative
		secondaryModels.loc[R,'ACC'] = (TP + TN) / totalPopulation
		secondaryModels.loc[R,'CSI'] = ( ( (secondaryModels.loc[R,'TPR'])**-1 ) + ( (secondaryModels.loc[R,'PPV'])**-1 ) - 1)**-1

	modelsColumns = secondaryModels[['area','model','group']][secondaryModels['predictor']=='three']

	secondaryModels = pd.DataFrame(secondaryModels[secondaryStats][secondaryModels['predictor']=='three'].as_matrix()-secondaryModels[secondaryStats][secondaryModels['predictor']=='two'].as_matrix(),columns=secondaryStats)
	secondaryModels = pd.concat([modelsColumns.reset_index(drop=True),secondaryModels],axis=1)

	if overallOnly:
		meanSecondaries = meanSecondaries.xs('Overall',level='area')
		secondaryModels = secondaryModels[:][secondaryModels['area']=='Overall']

	print(meanSecondaries)
	#print(secondaryModels)
	#exit()

	# create plot
	fig, ax = plt.subplots(figsize=(12,8))
	index = np.arange(len(meanSecondaries))
	if overallOnly:
		bar_width = 0.125
	else:
		bar_width = 0.1

	colors = ['b','g','r','m','y','k']


	for i,metric in enumerate(secondaryStats):
		plt.bar(index + i*bar_width, meanSecondaries[metric], bar_width,alpha=0.5,color=colors[i],label=metric,zorder=5)

	allIndices = np.array(list())

	for i in index:
		allIndices = np.hstack((allIndices,i + bar_width * np.arange(6)))

	points = ['.','^','*']
	for i,m in enumerate(['qda','svm','knn']):
		y = secondaryModels[:][secondaryModels['model']== m].as_matrix(columns=secondaryStats).ravel()
		plt.scatter(allIndices,y,color=(0,0,0),label=['QDA','SVM','KNN'][i],marker=points[i],s=55,zorder=10)


	plt.ylabel('Change in Metric Values Across Feature Sets \n (VV+VH,HAND - VV+VH)',labelpad=5,size='xx-large')
	ax.set_yticks(np.arange(-.3,.9,.1))
	ax.set_yticklabels(np.round(np.arange(-.3,.9,.1),1),fontsize='x-large')
	plt.grid(b=True,which='both',axis='y',zorder=-1,alpha=0.25)
	if overallOnly:
		plt.xticks(index + 2.5*bar_width, ['Agriculture','Canopy','Developed'],size='xx-large')
	else:
		plt.xticks(index + 2*bar_width, ['A','C','D','A','C','D','A','C','D','A','C','D'])

	# texts
	if not overallOnly:
		plt.text((ax.get_xticks()[2]-4*bar_width)/(ax.get_xlim()[1]-ax.get_xlim()[0]), -0.08,'Smithfield', ha='center', transform=ax.transAxes)
		plt.text((ax.get_xticks()[5]-4*bar_width)/(ax.get_xlim()[1]-ax.get_xlim()[0]), -0.08,'Goldsboro', ha='center', transform=ax.transAxes)
		plt.text((ax.get_xticks()[8]-4*bar_width)/(ax.get_xlim()[1]-ax.get_xlim()[0]), -0.08,'Kinston', ha='center', transform=ax.transAxes)
		plt.text((ax.get_xticks()[11]-4*bar_width)/(ax.get_xlim()[1]-ax.get_xlim()[0]), -0.08,'Overall', ha='center', transform=ax.transAxes)

	# add lines
	if not overallOnly:
		x = [ax.get_xticks()[2] + (ax.get_xticks()[2]-ax.get_xticks()[1])/2,ax.get_xticks()[2] + (ax.get_xticks()[2]-ax.get_xticks()[1])/2]
		y = [ax.get_ylim()[0]-0.01,ax.get_ylim()[0]-0.15]
		line = plt.Line2D(x, y,color='black')
		line.set_clip_on(False)
		ax.add_line(line)

		x = [ax.get_xticks()[5] + (ax.get_xticks()[4]-ax.get_xticks()[3])/2,ax.get_xticks()[5] + (ax.get_xticks()[4]-ax.get_xticks()[3])/2]
		y = [ax.get_ylim()[0]-0.01,ax.get_ylim()[0]-0.15]
		line = plt.Line2D(x, y,color='black')
		line.set_clip_on(False)
		ax.add_line(line)

		x = [ax.get_xticks()[8] + (ax.get_xticks()[6]-ax.get_xticks()[5])/2,ax.get_xticks()[8] + (ax.get_xticks()[6]-ax.get_xticks()[5])/2]
		y = [ax.get_ylim()[0]-0.01,ax.get_ylim()[0]-0.15]
		line = plt.Line2D(x, y,color='black')
		line.set_clip_on(False)
		ax.add_line(line)

	plt.minorticks_on()
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),mode="expand", loc=3,ncol=9, borderaxespad=0.,fontsize='xx-large',markerscale=1.5,handletextpad=0.15)
	#if overallOnly:
	#	plt.legend(loc='upper right',ncol=4)
	#else:
	#	plt.legend(loc='upper center',ncol=4)
	#plt.tight_layout()

	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	plt.tight_layout()

	if overallOnly:
		fig.savefig('manuscript/figures/barPlotByLC_weightedAverage_overallOnly_with_csi.jpeg',format='jpeg',dpi=300,quality=100)
	# else:
		# fig.savefig('manuscript/figures/barPlotByLC_weightedAverage.jpeg',format='jpeg',dpi=300,quality=100)
	plt.show()


def statsCalculate(predictors,classificationModels,areas,groups,writeDiffRasters):

	contigency_encoding = {'TP':1,'FP':2,'FN':3,'TN':4,'ndv':0}

	statsList = [ 'TP','FP','TN','FN',
				'TP_perc','FP_perc','TN_perc','FN_perc',
				'TP_area','FP_area','TN_area','FN_area','totalPopulation',
				'predPositive','predNegative','obsPositive','obsNegative',
				'prevalance','predPositive_perc','predNegative_perc','obsPositive_perc',
				'obsNegative_perc','predPositive_area','predNegative_area','obsPositive_area',
				'obsNegative_area','PPV','NPV','TPR','TNR','ACC','CSI']

	rows_list = []
	for c in product(areas,predictors,classificationModels,range(1,len(groups)+1)):

		print(c)
		area = c[0] ; predictor = c[1] ; model = c[2] ; group = c[3]

		predictedRasterFileName = 'data/results/predictedInundation_{}_{}_{}.tif'.format(predictor,area,model)
		validationRasterFileName = 'data/validation/processed/finalInundation_{}.tiff'.format(area)
		landcoverRasterFileName = 'data/lc/processed/lcNC_proj_{0}_proj_scaled_{0}_grouped.img'.format(area)

		predicted = gis.raster(predictedRasterFileName)
		validation = gis.raster(validationRasterFileName)
		landcover = gis.raster(landcoverRasterFileName)

		#if writeDiffRasters:
		if groups[group-1] == 'Overall':
			output_diffRaster_filename = 'data/results/diff_{}_{}_{}.tif'.format(predictor,area,model)
			output_diffRaster_filename = None
		else:
			output_diffRaster_filename = None
		diff = generateDifferenceRaster(predicted,validation,maskingRaster=landcover,maskingValue=group,output_fileName=output_diffRaster_filename,mapping=contigency_encoding)

		contingency_stats = calculateBinaryClassificationStatistics(diff,mapping=contigency_encoding)

		rowDictionary = { 'area': area, 'predictor' : predictor, 'model' : model, 'group' : groups[group-1] }

		for stat in statsList:
			rowDictionary[stat] = contingency_stats[stat]

		rows_list = rows_list + [rowDictionary]

		#statsDF = primary_secondary_stats(predicted,validation,landcover,statsDF,c,groups)

	statsDF = pd.DataFrame(rows_list)

	statsDF.to_csv('data/results/stats_with_csi.csv',index=False,index_label=False)

	## print overall stats
	# print(statsDF.loc[statsDF.loc[:,'group'] == 'Overall',['area','predictor','model','TP','FP','FN','TN']])



def plots(areas,classificationModels):

	sampleFractions = [0.005,0.003,0.005]
	# for area,sampleFrac in zip(areas,sampleFractions):
		# vv_vh_hand_filename = 'data/results/vv_vh_hand_{}.tiff'.format(area)
		# validationRasterFileName = 'data/validation/processed/finalInundation_{}.tiff'.format(area)

		# vv_vh_hand = gis.raster(vv_vh_hand_filename)
		# validation = gis.raster(validationRasterFileName)

		# scatterPlots(vv_vh_hand,area,validation,sampleFrac,xyzLim=[-25,0,-25,0,0,25])
		# scatterPlots3D(vv_vh_hand,validation,area,sampleFrac,xyzLim=[-25,0,-25,0,0,25])
		# contourPlots(vv_vh_hand,validation,area,sampleFrac,xyzLim=[-25,0,-25,-10,0,25])


	# scatterPlots_subplots(areas,sampleFrac=dict(zip(areas,sampleFractions)),xyzLim=[-25,0,-25,0,0,25])


	# barplotByArea()
#
	# barplotByLC()

	area = "Goldsboro"
	vv_vh_hand_filename = 'data/results/vv_vh_hand_{}.tiff'.format(area)
	vv_vh_hand_data = gis.raster(vv_vh_hand_filename)

	diffScatterPlots(vv_vh_hand_data,area,classificationModels,xyzLim=[-25,0,-25,0,0,25],predictor='three')
	diffScatterPlots(vv_vh_hand_data,area,classificationModels,xyzLim=[-25,0,-25,0,0,25],predictor='two')
