![banner](src/Banner1.jpg)

- [About PACT ](#about-codecarbon-)
- [Quickstart ](#quickstart-)
    - [Installation ](#installation-)
    - [Estimate Carbon Emission ](#start-to-estimate-your-impact-)
- [Workloads ](#workloads-)
    - [Stress Tests ](#stress-tests-)
    - [Computer Vision Benchmarks ](#cv-bm-)
    - [Natural Language Processing Benchmarks ](#nlp-bm-)
    - [Reinforcement Learning Benchmarks ](#rl-bm-)

# About PACT

**PACT** offers a methodology for accurately analyzing power consumption and tracking carbon emissions of a specific hardware setup. By utilizing PACT, we can make informed decisions aimed at optimizing energy efficiency, minimizing carbon footprints, and advancing toward a greener future.


# Quickstart

## Installation

Add PACT tracker to your python script:

```python
currentDir = os.path.dirname(os.path.realpath(__file__))
parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
sys.path.append(os.path.join(parentDir, "src"))
from trackerPACT import PACT
```

## Estimate Carbon Emission

```python
events_groups = [['MIGRATIONS'],['FAULTS'],['CACHE-MISSES'],['CYCLES']]

@PACT(measure_period = 1, perf_measure_period = 0.01, events_groups = events_groups, tracker_file_name = "./PACT.csv", PSU = "Corsair_1500i")
def your_function():
  # your code
  ```

- `measure_period`: Specify the sampling period (in seconds) to measure power.
- `perf_measure_period`: Specify the sampling period (in seconds) to collect performance counter data.
- `events_groups`: Specify the performance events to monitor during execution.
- `tracker_file_name`: Specify the name of the file to store tracking data.
- `PSU`: Specify the power supply unit being used for measurement. Current support for 'Corsair_1500i' and 'NZXT_850'.

PACT monitors emissions generated and performance metrics during the execution of your function.

# Workloads

PACT was evaluated by subjecting it to various workloads, including stress tests designed to pressure different hardware components. Additionally, state-of-the-art machine learning models from computer vision, natural language processing, and reinforcement learning were used to assess PACT's performance under real-life workload scenarios.

## Stress Tests

We stressed various hardware components using the following tools:

#### Tools Required:
1) **sleep**: https://man7.org/linux/man-pages/man3/sleep.3.html
```
Idle Test
```
2) **stress**: https://linux.die.net/man/1/stress
```
CPU, File IO, Virtual Memory (VM), and HDD Tests
```

3) **gpu-burn** https://github.com/wilicc/gpu-burn
```
GPU Tests
```

## Computer Vision Benchmarks

**Source Repository**: https://github.com/kuangliu/pytorch-cifar

## Natural Language Processing Benchmarks
## Reinforcement Learning Benchmarks
