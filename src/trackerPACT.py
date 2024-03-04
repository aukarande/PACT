import os
import time
import pandas as pd
import uuid
import psutil
import tzlocal
from typing import Callable, Optional
from functools import wraps
from apscheduler.schedulers.background import BackgroundScheduler
from liquidctl import find_liquidctl_devices
import pynvml
import sys
from typing import Dict
from rapl import RAPLFile
from units import Time
import re
import perfmon
import time, struct, os, fcntl
import multiprocessing

FROM_mWATTS_TO_kWATTH = 1000*1000*3600
FROM_kWATTH_TO_MWATTH = 1000

_sentinel = object()

def is_file_opened(
    needed_file
):
    result = False
    needed_file = os.path.abspath(needed_file)
    python_processes = []
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=["name", "cpu_percent", "pid"])
            if "python" in pinfo["name"].lower() or "jupyter" in pinfo["name"].lower():
                python_processes.append(pinfo["pid"])
                flist = proc.open_files()
                if flist:
                    for nt in flist:
                        if needed_file in nt.path:
                            result = True
        except:
            pass
    return result

class LIQUIDCTL():
    """
        This class is interface for tracking gpu consumption.
        All methods are done here on the assumption that all gpu devices are of equal model.
        The GPU class is not intended for separate usage, outside the Tracker class
    """
    def __init__(
        self,
        measure_period = 0.5,
        events_groups = [['MIGRATIONS'],['FAULTS'],['CACHE-MISSES'],['CYCLES']],
        PSU = "",
        PSUDictLiquidCTL = {},
        perf_measure_period = 0.01
        ):
        """
            This class method initializes GPU object.
            Creates fields of class object. All the fields are private variables
            Parameters
            ----------
            ignore_warnings: bool
                If true, then user will be notified of all the warnings. If False, there won't be any warnings.
                The default is False.
            Returns
            -------
            GPU: GPU
                Object of class GPU
        """
        self._measure_period = measure_period
        self._perf_measure_period = perf_measure_period
        self._start = time.time()
        self._rapl_time = time.time()
        self._PSU = PSU
        self._PSUDictLiquidCTL = PSUDictLiquidCTL
        devices = find_liquidctl_devices()
        for dev in devices:

            # Connect to the device. In this example we use a context manager, but
            # the connection can also be manually managed. The context manager
            # automatically calls `disconnect`; when managing the connection
            # manually, `disconnect` must eventually be called, even if an
            # exception is raised.
            with dev.connect():
                print(f'{dev.description} at {dev.bus}:{dev.address}:')

                # Devices should be initialized after every boot. In this example
                # we assume that this has not been done before.
                print('- initialize')
                init_status = dev.initialize()

                # Print all data returned by `initialize`.
                if init_status:
                    for key, value, unit in init_status:
                        print(f'- {key}: {value} {unit}')        

                # status = dev.get_status()

                # # Print all data returned by `get_status`.
                # print('- get status')
                # for key, value, unit in status:
                #     print(f'- {key}: {value} {unit}')        
            self._device = dev
            self._device.connect()

        self._psu_power = 0.0
        self._nvml_power = 0.0
        self._rapl_power = 0.0
        self._gpu_util_sm = 0.0
        self._gpu_util_memory = 0.0

        self._rapl_dir = "/sys/class/powercap/intel-rapl"
        self._intel_interface = IntelRAPL(rapl_dir=self._rapl_dir)
        self.event_groups_names = events_groups
        self.event_groups = []
        self.fd_groups = []
        self.__check_paranoid()
        self.__encode_events() 
        self.__initialize()       
        self.reset_events()
        self.enable_events()

    def destruct(self):
        self._device.disconnect()
        self.__destroy_events()
        #return

    def get_gpu_power(self):
        """
            This class method returns GPU power consumption. Pynvml library is used.
            Parameters
            ----------
            No parameters
            Returns
            -------
            gpus_powers: list
                list of GPU power consumption per every GPU
        """
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        gpus_powers = []
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpus_powers.append(float(pynvml.nvmlDeviceGetPowerUsage(handle)))
        pynvml.nvmlShutdown()
        return gpus_powers

    def get_gpu_utils(self):
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        gpus_util= []
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpus_util.append(pynvml.nvmlDeviceGetUtilizationRates(handle))
        pynvml.nvmlShutdown()        
        return {"gpus_util": gpus_util}
    
    def cpu_power(self):
        '''
        This class method returns CPU power consumption from reading RAPL files

        Parameters
        ----------
        No parameters

        Returns
        -------
        power: float
            Averge power of cpu between each call
        '''
        old_time = self._rapl_time
        now_time = time.time()
        duration = now_time - old_time
        self._rapl_time = time.time()
        all_cpu_details:Dict = self._intel_interface.get_cpu_details(duration=Time(duration))
        power = 0
        for metric, value in all_cpu_details.items():
            # "^Processor Power_\d+\(Watt\)$" for Inter Power Gadget
            if re.match(r"^Processor Power", metric):
                power += value
                # logger.debug(f"_get_power_from_cpus - MATCH {metric} : {value}")

            else:
                # logger.debug(f"_get_power_from_cpus - DONT MATCH {metric} : {value}")
                pass
        return power                                

    def get_instant_power(self):
        self._perf_data = [0 for i in range(len(self.event_groups_names))]
        for k in range(int(self._measure_period/self._perf_measure_period)):
            self._perf_data = [x + y for x, y in zip(self._perf_data, self.read_events())]
            time.sleep(self._perf_measure_period)
        
        self.reset_events()
        self.enable_events()

        pre_cpu_power = float(self.cpu_power())
        pre_gpu_power = sum([gpuPower / 1000 for gpuPower in self.get_gpu_power()])
        status = self._device.get_status()
        post_cpu_power = float(self.cpu_power())
        post_gpu_power = sum([gpuPower / 1000 for gpuPower in self.get_gpu_power()])   
        if self._PSU == "NZXT_850":
            self._psu_power = float(status[self._PSUDictLiquidCTL['_12V_peripherals_power']][1]) + float(status[self._PSUDictLiquidCTL['_12V_EPS_ATX12V_power']][1]) + float(status[self._PSUDictLiquidCTL['_12V_motherboard_PCIE_power']][1]) + float(status[self._PSUDictLiquidCTL['_5V_combined_power']][1]) + float(status[self._PSUDictLiquidCTL['_3_3V_combined_power']][1])
        elif self._PSU == "Corsair_1500i":
            self._psu_power = float(status[self._PSUDictLiquidCTL['_total_power']][1])
        self._rapl_power = (pre_cpu_power + post_cpu_power)/2
        self._nvml_power = (pre_gpu_power + post_gpu_power)/2
        
        gpu_stats_dict = self.get_gpu_utils()
        for gpuID in range(len(gpu_stats_dict["gpus_util"])):
            self._gpu_util_sm += float(gpu_stats_dict["gpus_util"][gpuID].gpu)
            self._gpu_util_memory += float(gpu_stats_dict["gpus_util"][gpuID].memory)
        
        self._gpu_util_sm /= len(gpu_stats_dict["gpus_util"])
        self._gpu_util_memory /= len(gpu_stats_dict["gpus_util"])          
        return 

    def calculate_consumption(self):
        old = self._start
        duration = time.time() - old
        self._start = time.time()
        self.get_instant_power()

        statusDict = {  "start_time": old,
                        "duration": duration,
                        "psu_power": self._psu_power,
                        "nvml_power": self._nvml_power, 
                        "rapl_power": self._rapl_power,
                        "gpu_util_sm": self._gpu_util_sm, 
                        "gpu_util_memory": self._gpu_util_memory, 
                        "migrations": self._perf_data[0],
                        "faults": self._perf_data[1],
                        "cache-misses": self._perf_data[2],
                        "cycles": self._perf_data[3]}
       
        self._psu_power = 0.0
        self._nvml_power = 0.0
        self._rapl_power = 0.0
        self._gpu_util_sm = 0.0
        self._gpu_util_memory = 0.0       
        return statusDict
    
    def __check_paranoid(self):
        """
        Check perf_event_paranoid wich Controls use of the performance events
        system by unprivileged users (without CAP_SYS_ADMIN).
        The default value is 2.

        -1: Allow use of (almost) all events by all users
            Ignore mlock limit after perf_event_mlock_kb without CAP_IPC_LOCK
        >=0: Disallow ftrace function tracepoint by users without CAP_SYS_ADMIN
            Disallow raw tracepoint access by users without CAP_SYS_ADMIN
        >=1: Disallow CPU event access by users without CAP_SYS_ADMIN
        >=2: Disallow kernel profiling by users without CAP_SYS_ADMIN
        """
        with open("/proc/sys/kernel/perf_event_paranoid", "r") as f:
            val = int(f.read())
            if val >= 2:
                raise Exception("Paranoid enable")

    def __encode_events(self):
        """
        Find the configuration perf_event_attr for each event name
        """
        for group in self.event_groups_names:
            ev_list = []
            for e in group:
                # TODO check err
                if "SYSTEMWIDE" in e:
                    e = e.split(":")[1]
                try:
                    err, encoding = perfmon.pfm_get_perf_event_encoding(
                        e, perfmon.PFM_PLM0 | perfmon.PFM_PLM3, None, None
                    )
                except:
                    # print("Error encoding : "+e)
                    raise
                ev_list.append(encoding)
            self.event_groups.append(ev_list)

    def __create_events(self):
        """
        Create the events from the perf_event_attr groups
        """
        numCPUs = multiprocessing.cpu_count()
        for group, group_name in zip(self.event_groups, self.event_groups_names):
            fd_list = []
            for e, e_name in zip(group, group_name):
                # e.exclude_kernel = 1
                # e.exclude_hv = 1
                e.inherit = 1
                e.disabled = 1
                for k in range(numCPUs):
                    fd = perfmon.perf_event_open(e, -1, k, -1, 0)
                    if fd < 0:
                        raise Exception("Erro creating fd " + e_name)
                    fd_list.append(fd)
            self.fd_groups.append(fd_list)

    def __destroy_events(self):
        """
        Close all file descriptors destroying the events
        """
        for group in self.fd_groups:
            for fd in group:
                os.close(fd)
        self.fd_groups = []

    def __initialize(self):
        """
        Prepare to run the workload
        """
        # TODO solve the race condition
        try:
            self.__destroy_events()
            self.__create_events()
        except Exception as e:
            raise
        finally:
            pass  # something really bad happen

    def __format_data(self, data):
        """
        Format the data

        Event groups reading format
            id
            time
            counter 1
            ...
            counter n
        Event single format
            counter

        output:
            [
                [counter 1 ... counter n]
                ...
                [counter 1 ... counter n]
            ]
        """
        all_data = []
        for s in data:
            only_s = []
            s = list(s)
            c = 0
            for g in self.event_groups_names:
                if len(g) > 1:
                    only_s += s[c + 2 : c + 2 + len(g)]
                    c = c + 2 + len(g)
                else:
                    only_s += [s[c]]
                    c += 1
            all_data.append(only_s)
        return all_data

    def enable_events(self):
        """
        Enable the events
        """
        for fd in self.fd_groups:
            for fdx in fd:
                fcntl.ioctl(fdx, Tracker.PERF_EVENT_IOC_ENABLE, 0)

    def disable_events(self):
        """
        Disable the events
        """
        for fd in self.fd_groups:
            for fdx in fd:
                fcntl.ioctl(fdx, Tracker.PERF_EVENT_IOC_DISABLE, 0)

    def reset_events(self):
        """
        Reset the events
        """
        for fd in self.fd_groups:
            for fdx in fd:
                fcntl.ioctl(fdx, Tracker.PERF_EVENT_IOC_RESET, 0)

    def read_events(self):
        """
        Read from the events
        """
        data = []
        for group in self.fd_groups:
            raw_all_cpus = 0
            for cpu_num in range(len(group)):
                raw = os.read(group[cpu_num], 4096)
                to_read = int(len(raw) / 8)
                raw = struct.unpack("q" * to_read, raw)
                raw_all_cpus += raw[0]
            data += [raw_all_cpus]
        return data


class IntelRAPL:
    '''
    From codeCarbon core.cpu
    '''
    def __init__(self, rapl_dir="/sys/class/powercap/intel-rapl"):
        self._lin_rapl_dir = rapl_dir
        self._system = sys.platform.lower()
        self._rapl_files = list()
        self._setup_rapl()
        self._cpu_details: Dict = dict()

        self._last_mesure = 0

    def _is_platform_supported(self) -> bool:
        return self._system.startswith("lin")

    def _setup_rapl(self):
        if self._is_platform_supported():
            if os.path.exists(self._lin_rapl_dir):
                self._fetch_rapl_files()
            else:
                raise FileNotFoundError(
                    f"Intel RAPL files not found at {self._lin_rapl_dir} "
                    + f"on {self._system}"
                )
        else:
            raise SystemError("Platform not supported by Intel RAPL Interface")
        return

    def _fetch_rapl_files(self):
        """
        Fetches RAPL files from the RAPL directory
        """

        # consider files like `intel-rapl:$i`
        files = list(filter(lambda x: ":" in x, os.listdir(self._lin_rapl_dir)))

        i = 0
        for file in files:
            path = os.path.join(self._lin_rapl_dir, file, "name")
            with open(path) as f:
                name = f.read().strip()
                # Fake the name used by Power Gadget
                if "package" in name:
                    name = f"Processor Energy Delta_{i}(kWh)"
                    i += 1
                # RAPL file to take measurement from
                rapl_file = os.path.join(self._lin_rapl_dir, file, "energy_uj")
                # RAPL file containing maximum possible value of energy_uj above which it wraps
                rapl_file_max = os.path.join(
                    self._lin_rapl_dir, file, "max_energy_range_uj"
                )
                try:
                    # Try to read the file to be sure we can
                    with open(rapl_file, "r") as f:
                        _ = float(f.read())
                    self._rapl_files.append(
                        RAPLFile(name=name, path=rapl_file, max_path=rapl_file_max)
                    )
                    #logger.debug(f"We will read Intel RAPL files at {rapl_file}")
                except PermissionError as e:
                    raise PermissionError(
                        "Unable to read Intel RAPL files for CPU power, we will use a constant for your CPU power."
                        + " Please view https://github.com/mlco2/codecarbon/issues/244"
                        + f" for workarounds : {e}"
                    )
        return
    
    def get_cpu_details(self, duration: Time, **kwargs) -> Dict:
        """
        Fetches the CPU Energy Deltas by fetching values from RAPL files
        """
        cpu_details = dict()
        try:
            list(map(lambda rapl_file: rapl_file.delta(duration), self._rapl_files))

            for rapl_file in self._rapl_files:
                # logger.debug(rapl_file)
                cpu_details[rapl_file.name] = rapl_file.energy_delta.kWh
                # We fake the name used by Power Gadget when using RAPL
                if "Energy" in rapl_file.name:
                    cpu_details[
                        rapl_file.name.replace("Energy", "Power")
                    ] = rapl_file.power.W
        except Exception as e:
            pass
            # logger.info(
            #     f"Unable to read Intel RAPL files at {self._rapl_files}\n \
            #     Exception occurred {e}",
            #     exc_info=True,
            # )
        self.cpu_details = cpu_details
        # logger.debug(f"get_cpu_details {self.cpu_details}")
        return cpu_details

    def get_static_cpu_details(self) -> Dict:
        """
        Return CPU details without computing them.
        """
        # logger.debug(f"get_static_cpu_details {self.cpu_details}")

        return self.cpu_details

    def start(self):
        for rapl_file in self._rapl_files:
            rapl_file.start()

class Tracker:
    PERF_EVENT_IOC_ENABLE = 0x2400
    PERF_EVENT_IOC_DISABLE = 0x2401
    PERF_EVENT_IOC_ID = 0x80082407
    PERF_EVENT_IOC_RESET = 0x2403    
    def __init__(
        self,
        tracker_file_name = "PACT.csv",
        measure_period=10,
        perf_measure_period = 0.01,
        events_groups = [['MIGRATIONS'],['FAULTS'],['CACHE-MISSES'],['CYCLES']],
        PSU = "Corsair_1500i",
        ):
        if (type(measure_period) == int or type(measure_period) == float) and measure_period <= 0:
            raise ValueError("\'measure_period\' should be positive number")

        self.tracker_file_name = tracker_file_name
        self._measure_period = measure_period
        self._perf_measure_period = perf_measure_period
        self._events_groups = events_groups
        self._scheduler = BackgroundScheduler(job_defaults={'max_instances': 5000}, timezone=str(tzlocal.get_localzone()),misfire_grace_time=None)
        self._start_time = None
        self._liquidCTL = None
        self._id = None
        self.data_dicts = []
        self.data_dict = {}
        self._mode = "first_time"
        self._PSU = PSU
        if self._PSU == "NZXT_850":
            self._PSUDictLiquidCTL = {"_12V_peripherals_power" : 5, "_12V_EPS_ATX12V_power" : 8, "_12V_motherboard_PCIE_power" : 11, "_5V_combined_power" : 14, "_3_3V_combined_power" : 17}
        elif self._PSU == "Corsair_1500i":
            self._PSUDictLiquidCTL = {"_total_power" : -3}

    def consumption(self):
        return self._consumption

    def measure_period(self):
        return self._measure_period

    def _construct_inst_power_attributes_dict(self,data_dict, index):
        attributes_dict = dict()
        attributes_dict["index"] = [index]
        attributes_dict["start_time"] = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data_dict["start_time"]))]
        attributes_dict["duration(s)"] = [data_dict["duration"]]    
        attributes_dict["Total_PSU_Power(W)"] = [data_dict["psu_power"]]        
        attributes_dict["Nvidia_SMI_Power(W)"] = [data_dict["nvml_power"]]
        attributes_dict["Intel_RAPL_Power(W)"] = [data_dict["rapl_power"]]
        attributes_dict["GPU_Util_SM(%)"] = [data_dict["gpu_util_sm"]]
        attributes_dict["GPU_Util_Memory(%)"] = [data_dict["gpu_util_memory"]]
        attributes_dict["migrations"] = [data_dict["migrations"]]
        attributes_dict["faults"] = [data_dict["faults"]]
        attributes_dict["cache-misses"] = [data_dict["cache-misses"]]
        attributes_dict["cycles"] = [data_dict["cycles"]]        
        return attributes_dict
    
    def _write_to_csv(
        self,
        add_new=False,
        ):
        if os.path.exists(self.tracker_file_name):
            os.remove(self.tracker_file_name)        
        
        directory = os.path.dirname(self.tracker_file_name)
        os.makedirs(directory, exist_ok=True)

        file_exists = False
        with open(self.tracker_file_name, "a") as file:
            for i in range(len(self.data_dicts)):
                data_dict = self.data_dicts[i]
                inst_attributes_dict = self._construct_inst_power_attributes_dict(data_dict,i)
                df = pd.DataFrame(inst_attributes_dict)

                # Write the header only if the file doesn't exist or it's the first entry
                header = not file_exists or i == 0
                df.to_csv(file, index=False, header=header, mode='a')

                # Update file_exists variable after the first entry is written
                file_exists = True

        self._mode = "run time" if self._mode != "training" else "training"
        # return inst_attributes_dict


    def _func_for_sched(self, add_new=False): 
        self._id = str(uuid.uuid4())

        self.data_dicts.append(self._liquidCTL.calculate_consumption())

        if self._mode == "shut down":
            self._scheduler.remove_job("job")
            self._scheduler.shutdown()
        
        # self._write_to_csv returns attributes_dict        
        # return self._write_to_csv(add_new)

    
    def start(self):
        if self._start_time is not None:
            try:
                self._scheduler.remove_job("job")
                self._scheduler.shutdown()
            except:
                pass
            self._scheduler = BackgroundScheduler(job_defaults={'max_instances': 10}, misfire_grace_time=None)
        self._liquidCTL = LIQUIDCTL(measure_period=self._measure_period, perf_measure_period=self._perf_measure_period, events_groups = self._events_groups, PSU = self._PSU, PSUDictLiquidCTL = self._PSUDictLiquidCTL)
        self._id = str(uuid.uuid4())
        self._start_time = time.time()
        self._scheduler.add_job(self._func_for_sched, "interval", seconds=self._measure_period, id="job")
        self._scheduler.start()

    def stop(self):
        if self._start_time is None:
            raise Exception("Need to first start the tracker by running tracker.start() or tracker.start_training()")
        self._scheduler.remove_job("job")
        self._scheduler.shutdown()
        self._func_for_sched() 
        self._write_to_csv()
        self._start_time = None
        self._consumption = 0
        self._liquidCTL.destruct()
        self._mode = "shut down"

def PACT(
    fn: Callable = None,
    measure_period: Optional[float] = 1,
    perf_measure_period: Optional[float] = 0.01,
    events_groups:  Optional[list] = [['MIGRATIONS'],['FAULTS'],['CACHE-MISSES'],['CYCLES']],
    tracker_file_name: Optional[str] = "PACT.csv",
    PSU: Optional[str] = "Corsair_1500i", # (NZXT_850), (Corsair_1500i)
):    
    def _decorate(fn: Callable):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            tracker = Tracker(measure_period = measure_period, perf_measure_period = perf_measure_period, events_groups=events_groups, tracker_file_name = tracker_file_name, PSU = PSU)
            tracker.start()
            try:
                returned = fn(*args, **kwargs)
            except Exception:
                tracker.stop()
                del tracker
                raise Exception
            tracker.stop()
            del tracker
            return returned
        return wrapped_fn

    if fn:
        return _decorate(fn)
    return _decorate