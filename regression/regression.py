import pandas as pd
import numpy as np
import os
from PFA import PFA
from piecewise import PiecewiseRegression
from sklearn import linear_model
import warnings
from tabulate import tabulate
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import os

perfCounterMeasurePeriod = 0.01 #seconds

# Result Folder
resultDir = './results'

# Data Folder
trainDataFolderPath = './PACTData/train'
testDataFolderPath = './PACTData/test'

#  Feature Keys
allKeys = ['duration(s)', 'Total_PSU_Power(W)', 'Nvidia_SMI_Power(W)', 'Intel_RAPL_Power(W)', 'GPU_Util_SM(%)', 'GPU_Util_Memory(%)', 'migrations', 'faults', 'cache-misses', 'cycles']
featureKeys = ['GPU_Util_SM(%)', 'GPU_Util_Memory(%)','migrations', 'faults', 'cache-misses','cycles']
predictionKey = ['Power_Other(W)']
predictionKeyPTotal = ['Total_PSU_Power(W)']  

carbonIntensity = 646 #https://app.electricitymaps.com/map

def readFileToDF(fileName, forceReadAllEntries = False, numEntriesToRead = 200, numEntriesPerRunDict = None):
    df = pd.read_csv(fileName)
    # find the first data with perf stat
    startEntryID = 0
    for i in range(len(df)):
        if not np.isnan(df.iloc[i, :]['cache-misses']):
            startEntryID = i + 1 #the next one
            break

    if forceReadAllEntries:
        df = df[allKeys]
    else:
        numEntriesToRead = max([numEntriesPerRunDict[key] if key in fileName else numEntriesToRead for key in numEntriesPerRunDict])
        df = df.iloc[startEntryID:startEntryID+numEntriesToRead][allKeys]

    df = df.fillna(0)
    return df


def getNumEntriesInFiles(fileOrder, folderPath, train = True):
    
    assert os.path.exists(folderPath)
    # List all files in the folder
    files = os.listdir(folderPath)
    # Filter files to read from the folder
    filteredFiles = [file for file in files if "PACT" in file]
    filteredFiles = [file for file in filteredFiles if any(file.startswith(name) for name in fileOrder)]
    sortedFiles = sorted(filteredFiles, key=lambda x: fileOrder.index([name for name in fileOrder if x.startswith(name)][0]))
    numEntriesPerRunDict = {}
    
    if train:
        for key in fileOrder:
            filesWithKeyInName = [file for file in sortedFiles if key in file]
            minEntries = 1000000000
            for fileName in filesWithKeyInName:
                with open(os.path.join(folderPath, fileName), 'r') as fp: 
                    numLines = len(fp.readlines())
                    if (numLines - 1) < minEntries:
                        minEntries = numLines - 1
            numEntriesPerRunDict[key] = minEntries
    else:
        for index in range(len(sortedFiles)):  
            fileName = sortedFiles[index]
            with open(os.path.join(folderPath, fileName), 'r') as fp: 
                numEntriesPerRunDict[fileOrder[index]] = len(fp.readlines()) - 1
    return numEntriesPerRunDict
                   
def createOrderOfFiles(folderPath, train = True):
    assert os.path.exists(folderPath)
    files = os.listdir(folderPath)
    files.sort()
    
    fileOrder = []
    dict = {}
    if train:
        for file in files:
            dict[file.split("_")[0]] = 0
    else:
        for file in files:
            dict[file.replace('_' + file.split("_")[-1], '')] = 0

    fileOrder = list(dict.keys())
    return fileOrder

def readFiles(fileOrder, folderPath, delimitPos = -3, numEntriesPerRunDict = numEntriesPerRunDict):
    assert os.path.exists(folderPath)
    testToOrigDFDict = {}
    print("Reading files!!!")
    # List all files in the folder
    files = os.listdir(folderPath)
    # Filter files to read from the folder
    filteredFiles = [file for file in files if "PACT" in file]
    filteredFiles = [file for file in filteredFiles if any(file.startswith(name) for name in fileOrder)]
    sortedFiles = sorted(filteredFiles, key=lambda x: fileOrder.index([name for name in fileOrder if x.startswith(name)][0]))

    for fileName in sortedFiles:
        key = '' 
        for i in fileName.split('_')[:delimitPos]:
            key = key + i +'_'
        key = key[:-1]
        df = readFileToDF(os.path.join(folderPath, fileName), numEntriesPerRunDict = numEntriesPerRunDict)
        if key in testToOrigDFDict:
            testToOrigDFDict[key].append(df)
        else:
            testToOrigDFDict[key] = [df]
    return testToOrigDFDict

def performRuntimeAveraging(testToOrigDFDict, runningAvgWindow = round(20/1)):
    runAvgDFDict = {}
    print("Runtime Averaging!!!")
    for test in testToOrigDFDict:
        avgList = []
        for runDF in testToOrigDFDict[test]:
            dfAvg = np.zeros((round(runDF.shape[0] - runningAvgWindow), runDF.shape[1]))
            for i in range(runDF.shape[0] - runningAvgWindow):
                dfAvg[i,:] = np.mean(runDF.to_numpy()[i:i + runningAvgWindow,:], axis = 0)
            dfAvg = pd.DataFrame(data=dfAvg, columns=runDF.columns)
            avgList.append(dfAvg)
        runAvgDFDict[test] = avgList
    return runAvgDFDict

def normalizeSomeFeatures(dfDict):
    normDFDict = {}
    print("Normalize Stats: ", featureKeys)    
    for test in dfDict:
        normList = []
        for runDF in dfDict[test]:
            normDF = runDF.copy()
            for i in featureKeys:
                normDF[i] = normDF[i]/(normDF['duration(s)']/perfCounterMeasurePeriod) 
            normList.append(normDF)
        normDFDict[test] = normList
    return normDFDict  

def performTrimMean(dfDict, dropTrimNum = 1):
    trimmedDFList = []
    print("Perform Trimmed Mean!", "dropTrimNum:", dropTrimNum)    
    for test in dfDict:
        pTotalColNum = dfDict[test][0].columns.get_loc(predictionKeyPTotal[0])
        averagedSlice = []
        dfsAsNPArray = np.array(dfDict[test])
        for dataID in range(dfsAsNPArray.shape[1]):
            powerSlice = dfsAsNPArray[:, dataID, pTotalColNum]
            allSlices = dfsAsNPArray[:, dataID, :]
            indicesSorted = np.argsort(powerSlice, axis=0)
            toAverageSlices = allSlices[indicesSorted[dropTrimNum:len(indicesSorted) - dropTrimNum]]
            averagedSlice.append(np.mean(toAverageSlices, axis=0))
        trimmedDF = pd.DataFrame(data=averagedSlice, columns=allKeys)
        trimmedDF['Power_Other(W)'] = trimmedDF['Total_PSU_Power(W)'] - (trimmedDF['Nvidia_SMI_Power(W)'] + trimmedDF['Intel_RAPL_Power(W)'])
        trimmedDFList.append(trimmedDF)
        trimmedDFList[-1]['from'] = [test]*dfsAsNPArray.shape[1]
    return trimmedDFList

def trainTestSplit(dfList, splitRatio = 0.7):
    dataDicts = {}
    numTrainSamples = [int(len(df) * splitRatio) for df in dfList]
    dataDicts['trainDFs'] = [df.iloc[:num] for df, num in zip(dfList, numTrainSamples)]
    dataDicts['testDFs'] = [df.iloc[num:] for df, num in zip(dfList, numTrainSamples)]
    concatTrainDFs = pd.concat(dataDicts['trainDFs'])
    concatTestDFs = pd.concat(dataDicts['testDFs'])
    dataDicts['allDataX'] = pd.concat([concatTrainDFs, concatTestDFs])[featureKeys].to_numpy()
    dataDicts['allDataY'] = pd.concat([concatTrainDFs, concatTestDFs])[predictionKey].to_numpy()    
    dataDicts['trainX'] = concatTrainDFs[featureKeys].to_numpy()
    dataDicts['trainY'] = concatTrainDFs[predictionKey].to_numpy()
    dataDicts['trainYPTotal'] = concatTrainDFs[predictionKeyPTotal].to_numpy()    
    dataDicts['testX'] = concatTestDFs[featureKeys].to_numpy()
    dataDicts['testY'] = concatTestDFs[predictionKey].to_numpy()
    dataDicts['testYPTotal'] = concatTestDFs[predictionKeyPTotal].to_numpy() 
    dataDicts['numTestSamples'] = [df.shape[0] for df in dataDicts['testDFs']] 
    return dataDicts

def performPFA(trainTestDicts):
    pfa = PFA(diff_n_features=0, q=3, explained_var= 0.95)
    pfa.fit_transform(trainTestDicts['allDataX'])

    columns = trainTestDicts['trainDFs'][0][featureKeys].columns
    clusteringKeys = [columns[index] for index in pfa.indices_]

    trainTestDicts['clusteringKeys'] = clusteringKeys
    trainTestDicts['pfaIndices'] = pfa.indices_

def performRegression(trainTestDicts):
    piecewiseReg = PiecewiseRegression(numClusters = 5, featureKeys = featureKeys, clusteringKeys = trainTestDicts['clusteringKeys'])
    piecewiseReg.fit(trainTestDicts['trainX'], trainTestDicts['trainY'])

    linearReg = linear_model.LinearRegression()
    linearReg.fit(trainTestDicts['trainX'], trainTestDicts['trainY'])

    trainTestDicts['piecewiseReg'] = piecewiseReg     
    trainTestDicts['linearReg'] = linearReg 

def wmape(actual, predicted):
    wmape = abs(actual - predicted).sum()/actual.sum()
    return wmape

def printStats(trainTestDicts):
    classifierNames = ['linearReg', 'piecewiseReg']
    classifiers = [trainTestDicts[key] for key in classifierNames]    
    inputs = [trainTestDicts['trainX']]    
    outputs = [trainTestDicts['trainY']] 
    tableNames = ["pOtherTrain"]
    
    for id in range(len(tableNames)):
        header = [tableNames[id]] + [name for name in classifierNames]
        preds = [classifiers[i].predict(inputs[id]) for i in range(len(classifiers))]
        pOtherTrain = [["r2"] + [r2_score(outputs[id], preds[i]) for i in range(len(classifiers))]]
        table = tabulate(pOtherTrain, header, tablefmt="grid")
        print(table, "\n")    

    concatTrainDF = pd.concat(trainTestDicts['trainDFs'], ignore_index=True)
    concatTestDF = pd.concat(trainTestDicts['testDFs'], ignore_index=True)
    powerGPUs = [np.expand_dims(concatTrainDF['Nvidia_SMI_Power(W)'].to_numpy(), -1), np.expand_dims(concatTestDF['Nvidia_SMI_Power(W)'].to_numpy(), -1)]
    powerCPUs = [np.expand_dims(concatTrainDF['Intel_RAPL_Power(W)'].to_numpy(), -1), np.expand_dims(concatTestDF['Intel_RAPL_Power(W)'].to_numpy(), -1)]
    inputs = [trainTestDicts['trainX']]    
    outputs = [trainTestDicts['trainYPTotal']] 
    tableNames = ["pTotalTrain"]
    
    for id in range(len(tableNames)):
        header = [tableNames[id]] + [name for name in classifierNames]
        preds = [classifiers[i].predict(inputs[id]) + powerGPUs[id] + powerCPUs[id] for i in range(len(classifiers))]
        pTotalTrain = [["r2"] + [r2_score(outputs[id], preds[i]) for i in range(len(classifiers))]]
        table = tabulate(pTotalTrain, header, tablefmt="grid")
        print(table, "\n")         
    return

def printTestDataStats(dataDicts, fileOrder):

    if not os.path.exists(resultDir):
        os.makedirs(resultDir)    

    classifierNames = ['piecewiseReg', 'linearReg']
    classifierToDFDict = {}
    for classifierName in classifierNames:
        testDataAllStatsDF = pd.DataFrame(columns=['TestRun', 'POtherTest(mape)', 'POtherTest(wmape)', 'PTotalTest(mape)', 'PTotalTest(wmape)', 'actualEnergy(kWh)', 'actualCarbonEmission(gCO₂eq)', 'predEnergy(kWh)', 'predCarbonEmission(gCO₂eq)'])
        for testID in range(len(dataDicts['testDFs'])):
            allStatsDict = {}
            allStatsDict['TestRun'] = fileOrder[testID]
            durations = dataDicts['testDFs'][testID]['duration(s)'].to_numpy()
            inputs = dataDicts['testDFs'][testID][featureKeys].to_numpy()
            outputs = dataDicts['testDFs'][testID][predictionKey].to_numpy()
            outputsPTotal = dataDicts['testDFs'][testID][predictionKeyPTotal].to_numpy()            
            preds = dataDicts[classifierName].predict(inputs)
            
            allStatsDict['POtherTest(mape)'] = mean_absolute_percentage_error(outputs, preds)
            allStatsDict['POtherTest(wmape)'] = wmape(outputs, preds)

            powerGPUs = np.expand_dims(dataDicts['testDFs'][testID]['Nvidia_SMI_Power(W)'].to_numpy(), -1)
            powerCPUs = np.expand_dims(dataDicts['testDFs'][testID]['Intel_RAPL_Power(W)'].to_numpy(), -1)
            predsPTotal = preds + powerCPUs + powerGPUs

            allStatsDict['PTotalTest(mape)'] = mean_absolute_percentage_error(outputsPTotal, predsPTotal)
            allStatsDict['PTotalTest(wmape)'] = wmape(outputsPTotal, predsPTotal)

            allStatsDict['actualEnergy(kWh)'] = sum((1/0.92)*durations[i]*outputsPTotal[i] for i in range(len(durations)))[0] * 0.00000027778
            allStatsDict['actualCC(gCO₂eq)'] = allStatsDict['actualEnergy(kWh)'] * carbonIntensity

            allStatsDict['predEnergy(kWh)'] = sum((1/0.92)*durations[i]*predsPTotal[i] for i in range(len(durations)))[0] * 0.00000027778
            allStatsDict['predCC(gCO₂eq)'] = allStatsDict['predEnergy(kWh)'] * carbonIntensity            

            testDataAllStatsDF = testDataAllStatsDF.append(allStatsDict, ignore_index=True)
        testDataAllStatsDF.to_csv(resultDir + '/' + classifierName + 'TestAllStats.csv', index=False)
        classifierToDFDict[classifierName] = testDataAllStatsDF
    
    energyStatsDict = {}
    energyStatsDict['TestRun'] = classifierToDFDict['linearReg']['TestRun']
    energyStatsDict['actualEnergy(kWh)'] = classifierToDFDict['linearReg']['actualEnergy(kWh)']
    energyStatsDict['actualCC(gCO₂eq)'] = classifierToDFDict['linearReg']['actualCC(gCO₂eq)']
    energyStatsDict['predEnergyLinear(kWh)'] = classifierToDFDict['linearReg']['predEnergy(kWh)']
    energyStatsDict['predCarbonEmissionLinear(gCO₂eq)'] = classifierToDFDict['linearReg']['predCC(gCO₂eq)']
    energyStatsDict['predEnergyPieceWise(kWh)'] = classifierToDFDict['piecewiseReg']['predEnergy(kWh)']
    energyStatsDict['predCarbonEmissionPieceWise(gCO₂eq)'] = classifierToDFDict['piecewiseReg']['predCC(gCO₂eq)']

    if not os.path.exists(resultDir):
        os.makedirs(resultDir)    

    pd.DataFrame(energyStatsDict).to_csv(resultDir + '/EnergyStats.csv', index=False)
    return 

def main():
    warnings.filterwarnings("ignore")
    # Pre-Processing
    trainFileOrder = createOrderOfFiles(trainDataFolderPath, train = True)
    numEntriesPerRunDict = getNumEntriesInFiles(trainFileOrder, trainDataFolderPath, train = True)     
    trainDataToOrigDFDict = readFiles(trainFileOrder, trainDataFolderPath, numEntriesPerRunDict = numEntriesPerRunDict)
    trainDataToRunAvgDFDict = performRuntimeAveraging(trainDataToOrigDFDict)   
    trainDataToRunAvgNormDFDict = normalizeSomeFeatures(trainDataToRunAvgDFDict)
    trimmedDFList = performTrimMean(trainDataToRunAvgNormDFDict)
    trainDataDicts = trainTestSplit(trimmedDFList)

    performPFA(trainDataDicts)
    print("featureKeys:", featureKeys)
    print("clusteringKeys:", trainDataDicts['clusteringKeys'])

    performRegression(trainDataDicts)

    # Stats
    printStats(trainDataDicts)

    # Test Dataset
    testFileOrder = createOrderOfFiles(trainDataFolderPath, train = False)    
    numEntriesPerRunDict = getNumEntriesInFiles(testFileOrder, testDataFolderPath, train = False)                      
    testDataToOrigDFDict = readFiles(testFileOrder, testDataFolderPath, delimitPos = -1, numEntriesPerRunDict = numEntriesPerRunDict)
    testDataToRunAvgDFDict = performRuntimeAveraging(testDataToOrigDFDict)
    testDataToRunAvgNormDFDict = normalizeSomeFeatures(testDataToRunAvgDFDict)
    testDataDFList = performTrimMean(testDataToRunAvgNormDFDict, dropTrimNum = 0)
    testDataDicts = trainTestSplit(testDataDFList, splitRatio = 0)        
    testDataDicts['clusteringKeys'] = trainDataDicts['clusteringKeys']
    testDataDicts['pfaIndices'] = trainDataDicts['pfaIndices']
    testDataDicts['piecewiseReg'] = trainDataDicts['piecewiseReg']     
    testDataDicts['linearReg'] = trainDataDicts['linearReg'] 
    printTestDataStats(testDataDicts, testFileOrder)

if __name__ == "__main__":
    main()