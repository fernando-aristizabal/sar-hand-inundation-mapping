#!/usr/bin/env python3


import numpy as np
import pandas as pd
from osgeo import gdal
import sys
import gis
import copy


"""
Description: Outputs statistics on NLCD11 land cover classifications for study area.
             Also aggregates level II classes to study defined classes including canopy, developed, & agriculture.

Use: python3 landcoverProcessing.py input.img output.img
	 input.img is the NLCD11 file for the specified region
"""

def printDivider():
	print('-----------------------------------------------------------')

def lcStats(rast,levels,heading,grpNames,area,output):
	
	# assign variables from object
	array = rast.array
	gt = rast.gt

	# generate number of occurences
	counts = np.repeat(0,len(levels))
	for i,lev in enumerate(levels):
		counts[i] = (array == lev).sum()
	totalCount = sum(counts)

	# determine prevalences
	prevalence = counts / totalCount
	totalPrevalence = sum(prevalence)

	# determine areas in sq km
	pixelArea = abs(gt[1] * gt[5]) # determine area of pixel
	areas = (counts * pixelArea) / (1000**2)
	totalAreas = sum(areas)

	# Generate data frame
	if isinstance(output, pd.DataFrame):
		output = output.append(pd.DataFrame({'Area': area,
											 'LC' : levels,
											 'Name': grpNames,
											 'Count' : counts,
											 'Prevalence' : prevalence,
											 'Areas(sqkm)' : areas}),
								ignore_index=True)
	else:
		output = pd.DataFrame({'Area': area,
							   'LC' : levels,
							   'Name': grpNames,
							   'Count' : counts,
							   'Prevalence' : prevalence,
							   'Areas(sqkm)' : areas})

	"""
	# print output
	printDivider()
	print('{} Summary'.format(heading))
	printDivider()
	print('LC\tCount\tPrevalence\tArea(sq km)\tName')
	for i,row in output.iterrows():
		print('{}\t{:,}\t{:.4f}\t\t{:.4f}\t\t{}'.format(row['LC'],row['Count'],row['Prevalence'],row['Areas(sqkm)'],row['Name']))
	printDivider()
	print('Total:\t{:,}\t{:.4f}\t\t{:,.4f}'.format(totalCount,totalPrevalence,totalAreas))
	"""

	return(output)
	

def aggregateLC(rast,groups,ndv):

	array = rast.array
	nrows = rast.nrows
	ncols = rast.ncols

	lcGroupsRaster = copy.copy(rast)

	# initiate output array
	lcGroups = np.zeros(nrows*ncols,dtype=int)

	# flatten input array
	array = np.ravel(array)

	# write array based on inputs and 
	names = [''] * len(groups.keys())
	for i,k in enumerate(sorted(groups.keys(),reverse=True)): 
		names[i] = k
		lcGroups[np.in1d(array,groups[k])] = i+1

	lcGroups = np.reshape(lcGroups,[nrows, ncols])

	lcGroupsRaster.array = lcGroups
	lcGroupsRaster.ndv = ndv
	
	return(lcGroupsRaster,names)


# main script
def combineLC(inputFileName, outputFileName, ndv,levelIIs,levelIINames,groups,areas): 
	
	originalStats,aggregateStats = None,None

	for a in areas:
		inputFileNameArea = inputFileName.format(a)
		outputFileNameArea = outputFileName.format(a)

		origLC = gis.raster(inputFileNameArea)

		originalStats = lcStats(origLC,levelIIs, "Level II Landcover",levelIINames,a,originalStats)

		lcGroups, grpNames = aggregateLC(origLC,groups,ndv)

		aggregateStats = lcStats(lcGroups,range(1,len(groups.keys())+1),"Landcover Groups",grpNames,a,aggregateStats)

		#gis.writeGeotiff(lcGroups, outputFileNameArea,2,'HFA')

	### sort by area, then count, then number ###
	originalStats = originalStats.sort_values(['Area','Count','LC'],ascending=[False,False,True])
	aggregateStats = aggregateStats.sort_values(['Area','Count','LC'],ascending=[False,False,True])

	
	# write to csv
	originalStats.to_csv('data/results/originalLCstats.csv',mode='w+',index=False)
	aggregateStats.to_csv('data/results/groupedLCstats.csv',mode='w+',index=False)

	### Group By landcover ###
	byLcOriginalStats = originalStats.groupby(['LC']).sum()
	byLcAggregateStats = aggregateStats.groupby(['Name']).sum()

	# re write prevalance
	byLcOriginalStats['Prevalence'] = byLcOriginalStats['Count'] / byLcOriginalStats['Count'].sum()
	byLcAggregateStats['Prevalence'] = byLcAggregateStats['Count'] / byLcAggregateStats['Count'].sum()

	# remove LC
	byLcAggregateStats.drop(['LC'], axis=1, inplace=True)

	# write to csv
	byLcOriginalStats.to_csv('data/results/byLcOriginalLCstats.csv',mode='w+')
	byLcAggregateStats.to_csv('data/results/byLcAggregateLCstats.csv',mode='w+')


	## Group By study area ##
	byAreaOriginalStats = originalStats.groupby(['Area']).sum()
	byAreaAggregateStats = aggregateStats.groupby(['Area']).sum()

	# rewrite prevalence
	byAreaOriginalStats['Prevalence'] = byAreaOriginalStats['Count'] / byAreaOriginalStats['Count'].sum()
	byAreaAggregateStats['Prevalence'] = byAreaAggregateStats['Count'] / byAreaAggregateStats['Count'].sum()

	# remove LC
	byAreaOriginalStats.drop(['LC'], axis=1, inplace=True)
	byAreaAggregateStats.drop(['LC'], axis=1, inplace=True)

	# write to csv
	byAreaOriginalStats.to_csv('data/results/byAreaOriginalLCstats.csv',mode='w+')
	byAreaAggregateStats.to_csv('data/results/byAreaAggregateLCstats.csv',mode='w+')
	
	print(originalStats)
	print(aggregateStats)
	print(byLcOriginalStats)
	print(byLcAggregateStats)
	print(byAreaOriginalStats)
	print(byAreaAggregateStats)






