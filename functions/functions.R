###################################################################################################
######################### Contains Additional Functions Required ##################################
###################################################################################################

###################################################################################################
## install and load all packages

install_load <- function(pkgList) {
  
  for (i in pkgList) {
    if ((i %in% installed.packages()[,1]) == FALSE) {install.packages(i)}
    if (paste0("package:",i) %in% search() == FALSE) {library(i,character.only = T,quietly = T)}
  }  
  
  message("*** All necessary packages are now installed and loaded ***")
}

###################################################################################################
# classify entire image

classifyImage <- function(img,model,output,type,startClust=T,endClust=F,trData=NA,best_k=NA,best_l=NA,naPixels=NA) {
 
   if (startClust) {
    tryCatch(getCluster(),error = function(e) {beginCluster()})
  } else {
    endClust = F
  }
  
  if (type == "KNN") {
    modelKNN <- knn(trData[,names(trData) !='class'],values(img)[!naPixels,],trData[,'class'],k=best_k,l=best_l)
    preds <- raster(inputRaster,layer=1)
    vals <- rep(0,ncell(preds))
    vals[!naPixels] <- modelKNN
    values(preds) <- vals
    NAvalue(preds) <- 0 ; names(preds) <- NA
  } else {
    preds <- clusterR(img, raster::predict, args = list(model = model))
  }
 
   writeRaster(preds,output,"GTiff", overwrite=TRUE,datatypeCharacter='int1U',NAflag=0)
  
  if (endClust) {endCluster()}
}

###################################################################################################
# extract training data

extractTrainingData <- function(img,sf) {
  trainData <- shapefile(sf)
  
  responseCol <- "water" #here it is important to set the correct column of the training data shapefile
  
  #creates a dataframe for extracting the information from the images
  dfAll <- data.frame(matrix(vector(), nrow = 0, ncol = length(names(img)) + 1))
  
  #extracts the pixels from the image on the training data
  for (i in 1:length(unique(trainData[[responseCol]]))){                          
    category <- unique(trainData[[responseCol]])[i]
    categorymap <- trainData[trainData[[responseCol]] == category,]
    dataSet <- extract(img, categorymap)
    dataSet <- cbind(dataSet,categorymap$water)
    dfAll <- rbind(dfAll,dataSet)
  }
  
  names(dfAll) <- c(names(img),"class")
  dfAll$class <- as.factor(dfAll$class)
  
  return(dfAll)
}

###################################################################################################
# adds error bars to interact plots


addErrorBars <- function(x.factor,trace.factor,response,alpha=0.05,epsilon=0.02,colors) {
  
  x.factor.levels <- levels(x.factor)
  trace.factor.levels <- levels(trace.factor)
  pLower <- alpha/2
  pUpper <- 1-alpha/2
  xValues <- 1:length(x.factor.levels)
  
  iNum <- 1
  for (i in trace.factor.levels){
    col <- colors[iNum] ; iNum <- iNum + 1
    avg <- c() ; qLower <- c() ; qUpper <- c()
    for (ii in x.factor.levels){
      responseValues <- response[x.factor== ii & trace.factor== i]
      currentAvg <- mean(responseValues)
      currentStd <- sd(responseValues)
      
      avg <- c(avg,mean(responseValues))
      qLower <- c(qLower,currentAvg + qt(alpha/2,length(responseValues)-1) * currentStd)
      qUpper <- c(qUpper,currentAvg + qt(1-alpha/2,length(responseValues)-1) * currentStd)
    }  
    segments(xValues, qLower,
             xValues, qUpper,col=col)
    segments(xValues-epsilon, qLower,
             xValues+epsilon, qLower,col=col)
    segments(xValues-epsilon, qUpper,
             xValues+epsilon, qUpper,col=col)
  }
}





