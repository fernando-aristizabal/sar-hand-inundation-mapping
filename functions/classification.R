############################################## SAR INUNDATION DETECTION ##################################################

## usage:
## Rscript classification.R <workingDirectory> <noDataValue> <inputRaster1> <inputRaster2> <trainingPoints> <outputRaster1> <outputRaster2> <outputRaster3> 
## Rscript ./scripts/functions/classification.R /home/lqc/Documents/research/sarHand -9999 data/results/vv_vh_YadkinBranch-NeuseRiver_proj.tiff data/results/vv_vh_hand_YadkinBranch-NeuseRiver_proj.tiff data/trainData/water4.shp data/trainData/water3.shp data/results/vv_vh data/results/vv_vh_hand

#### initialize ##########################################################################################################                                      

# assign arguments
workingDirectory <- commandArgs(trailingOnly = T)[1]
noData <- as.numeric(commandArgs(trailingOnly = T)[2])
areas <- commandArgs(trailingOnly = T)[3:5]

workingDirectory <- '/home/lqc/sarHand'
noData <- -9999
areas <- c("Smithfield","Goldsboro","Kinston")

# set working directory
setwd(workingDirectory)

# set seed for repeatability
set.seed(1) 

# source functions file
source(file.path(workingDirectory,'scripts/functions/functions.R'))

# install and load all necessary packages
install_load(c("MASS","rgdal","raster","sp","e1071","parallel","lattice","class")) 

# set number of repetitions
n <- 10

# factor sets
times <- expand.grid(predictors=c("three","two"),model=c("qda","svm","knn"),areas=areas)
#finalTimes <- data.frame(predictors=rep(NA,nrow(times)*n),model=rep(NA,nrow(times)*n),
#                         areas=rep(NA,nrow(times)*n),k=rep(NA,nrow(times)*n),
#                         cost=rep(NA,nrow(times)*n),gamma=rep(NA,nrow(times)*n),
#                         time=rep(NA,nrow(times)*n))
finalTimes <- data.frame(predictors=factor(),model=factor(),
                         areas=factor(),time=double())
parameters <- data.frame(predictors=factor(),model=factor(),areas=factor(),
                         k=integer(),cost=double(),gamma=double(),error=double())
bestParameters <- data.frame(predictors=factor(),model=factor(),
                            areas=factor(),k=integer(),
                            cost=double(),gamma=double(),
                            error=double())

# begin parrallel cluster
beginCluster()


iii <- 0
# both sets of predictors
for (i in 1:nrow(times)) {
  
  message(paste('classifying....',times$predictors[i],'with',times$model[i],'for',times$areas[i],'area.'))
  
  inputRaster <- stack(paste0('data/results/vv_vh_hand_',times$areas[i],'.tiff'))
  names(inputRaster) <- c("VV","VH","HAND")
  NAvalue(inputRaster) <- rep(noData,3)
  
  if (times$predictors[i] == 'two') {
    inputRaster <- inputRaster[[1:2]]
  }
  
  outputRaster <- paste0('data/results/predictedInundation_',times$predictors[i],'_',times$areas[i],'_',times$model[i],'.tiff')
  
  validation <- raster(paste0('data/validation/processed/finalInundation_',times$areas[i],'.tiff'))
  
  NAvalue(validation) <- 0
  
  trainingDataFileName <- paste0('data/trainData/trainingData_',times$predictors[i],'_',times$areas[i],'.shp')

  if (!file.exists(trainingDataFileName)) {
    sampledData <- sampleStratified(validation,40,sp=T,na.rm=T)
    names(sampledData) <- c('cell','water')
    writeOGR(sampledData,dsn=trainingDataFileName,layer='water',driver='ESRI Shapefile',overwrite_layer = T)
  }

  trData <- extractTrainingData(inputRaster,trainingDataFileName)
  
  #### classification ##########################################################################################################
  
  # qda
  if (times$model[i] == 'qda') {
    
    for (ii in 1:n) {
      iii <- iii + 1
      start <- proc.time()[3]
      modelQDA <- qda(class ~ . , data=trData,CV=F) # train
      classifyImage(inputRaster,modelQDA,outputRaster,"QDA",F,F)
      finalTimes <- rbind(finalTimes,
                          data.frame(predictors=times$predictors[i],model=times$model[i],
                                     areas=times$areas[i],time=unname(proc.time()[3]-start)))
    }
  }
  
  # support vector machine
  if (times$model[i] == 'svm') {
    
    parametersSVM <- list(kernel='radial',cost=seq(0.02,1,0.02),gamma=seq(0.02,1,0.02))
    modelSVM <- tune(svm, class ~ ., data=trData, ranges=parametersSVM)
    bestCost <- unlist(unname(modelSVM$best.parameters['cost']))
    bestGamma <- unlist(unname(modelSVM$best.parameters['gamma']))
    parameters <- rbind(parameters,data.frame(predictors=times$predictors[i],model=times$model[i],areas=times$areas[i],
                                k=NA,cost=modelSVM$performances$cost,gamma=modelSVM$performances$gamma,error=modelSVM$performances$error))
    bestParameters <- rbind(bestParameters,data.frame(predictors=times$predictors[i],model=times$model[i],areas=times$areas[i],k=NA,
                                                      cost=bestCost,gamma=bestGamma,error=modelSVM$best.performance))
    # if (times$predictors[i]=='three' & times$model[i]=='svm' & times$areas[i]== 'Smithfield') {
    #   tiff('data/results/svm_cv_three_smithfield.tiff',width=504,height = 300)
    #   levelplot((1-error)~cost*gamma,data=modelSVM$performances,at=c(seq(0.80,0.88,0.005)))
    #   dev.off()
    # }

    for (ii in 1:n) { 
      iii <- iii + 1
      start <- proc.time()[3]
      modelSVM <- svm(class ~ ., data=trData,kernel='radial',cost=bestCost,gamma=bestGamma)
      classifyImage(inputRaster,modelSVM,outputRaster,"SVM",F,F)
      finalTimes <- rbind(finalTimes,
                          data.frame(predictors=times$predictors[i],model=times$model[i],
                                     areas=times$areas[i],time=unname(proc.time()[3]-start)))
    }
  }
  ## KNN
  if (times$model[i] == 'knn') {
    
    modelKNN <- tune.knn(trData[,names(trData) !='class'],trData[,'class'],k=1:25,l=0)
    best_k <- modelKNN$best.parameters$k
    best_l <- modelKNN$best.parameters$l
    naPixels <- apply(is.na(values(inputRaster)),MARGIN = 1,any)
    parameters <- rbind(parameters,data.frame(predictors=times$predictors[i],model=times$model[i],areas=times$areas[i],
                                              k=modelKNN$performances$k,cost=NA,gamma=NA,error=modelKNN$performances$error))
    bestParameters <- rbind(bestParameters,data.frame(predictors=times$predictors[i],model=times$model[i],areas=times$areas[i],
                                                      k=modelKNN$performances$k,cost=NA,gamma=NA,error=modelKNN$performances$error))
    # if (times$predictors[i]=='three' & times$model[i]=='knn' & times$areas[i]== 'Smithfield') {
    #   tiff('data/results/knn_cv_three_smithfield.tiff',width=504,height = 300)
    #   plot(modelKNN$performances$k,1-modelKNN$performances$error,
    #        xlab="k",ylab="CV Accuracy",type='l')
    #   points(modelKNN$performances$k[which.max(1-modelKNN$performances$error)],max(1-modelKNN$performances$error),
    #          pch=4,col="red",lwd=3)
    #   dev.off()
    # }
    
    for (ii in 1:n) {
      iii <- iii + 1
      start <- proc.time()[3]
      classifyImage(inputRaster,modelKNN,outputRaster,"KNN",F,F,trData,best_k,best_l,naPixels)
      finalTimes <- rbind(finalTimes,
                          data.frame(predictors=times$predictors[i],model=times$model[i],
                                     areas=times$areas[i],time=unname(proc.time()[3]-start)))
    }
  }
  
# print(finalTimes)
# print('')
# print(parameters)
# print('')
# print(bestParameters)

  
}

endCluster()

write.csv(finalTimes,file='data/results/times.csv',row.names = F,quote=F)
write.csv(parameters,file='data/results/parameters.csv',row.names = F,quote=F)
write.csv(bestParameters,file='data/results/bestParameters.csv',row.names = F,quote=F)

## rename training data files
#for (i in list.files('data/trainData',pattern='*_three_*')){
 # file.rename(i,paste0('data/trainData/',i,'_prev'))
#}

print(times)