### How to track carbon emissions of a function in your Python code

```python
currentDir = os.path.dirname(os.path.realpath(__file__))
parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
sys.path.append(os.path.join(parentDir, "src"))
from trackerPACT import PACT

events_groups = [['MIGRATIONS'],['FAULTS'],['CACHE-MISSES'],['CYCLES']]

@PACT(measure_period=1, perf_measure_period = 0.01, events_groups = events_groups, tracker_file_name = "./PACT.csv", PSU = "Corsair_1500i")
def your_function():
  # your code
  ```


## Configuration Options

- `measure_period`: Specifies the duration (in seconds) for which power consumption will be measured.

- `perf_measure_period`: Specifies the interval (in seconds) for collecting performance counter data.

- `events_groups`: Defines the performance events to monitor during execution.

- `tracker_file_name`: Specifies the name of the file to store tracking data.

- `PSU`: Specifies the power supply unit being used for measurement. Current support for 'Corsair_1500i' and 'NZXT_850'.


The package monitors power consumption and performance counters during the execution of your function.