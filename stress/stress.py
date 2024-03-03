from threading import Timer
import time
import sys
import subprocess
import os

currentDir = os.path.dirname(os.path.realpath(__file__))
parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
sys.path.append(os.path.join(parentDir, "src"))
from trackerPACT import PACT

inputFile = "input.txt"
resultDir = "PACTData"
tracker_file_name = "PACT.csv"

measure_period = float(sys.argv[1])
perf_measure_period = float(sys.argv[2])

events_groups = [['MIGRATIONS'],['FAULTS'],['CACHE-MISSES'],['CYCLES']]

@PACT(measure_period=measure_period, perf_measure_period = perf_measure_period, events_groups = events_groups, tracker_file_name = resultDir + "/" + tracker_file_name)
def my_function(perfFileName, stressArgs):

    if "base" in perfFileName:
        sleepTime = stressArgs[0]
        stressArgs = ["sleep", sleepTime]
    
    with open(perfFileName, "w") as logfile:							 
        p = subprocess.Popen(stressArgs, stdout=logfile, stderr=logfile)
        ret_code = p.wait()
        logfile.flush()    

def main():
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    with open(inputFile) as file:
        lines = [line.rstrip() for line in file]
        for line in lines:
            if "trial#" not in line:
                config = line.split(",")[0]
                trial = line.split(",")[1]
                stressArgs = line.split(",")[2].split(" ")
                
                configName = config + "_" + trial
                start = time.time()
                my_function(resultDir + "/" + configName + "_out.csv", stressArgs)
                end = time.time()

                os.rename(resultDir + "/" + tracker_file_name, resultDir + "/" + configName + "_" + tracker_file_name)

                print("Time: ", "{:.2f}".format(end-start))
  
if __name__=="__main__":
    main()