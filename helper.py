import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import json
import numpy as np
import av
import av.io
from scipy.signal import find_peaks
from typing import Union, List, Tuple, Optional

# Basic folder, need to be set from the class using it
FIRED_BASE_FOLDER = None

# Divider for info in files
DIVIDER = "__"
# Time format used in file names
STORE_TIME_FORMAT = '%Y_%m_%d__%H_%M_%S'
# Delimiter used for csv files
DELIMITER = "\t"

# Annotation types prefix
LIGHT_ANNOTATION = "light"
SENSOR_ANNOTATION = "sensor"
DEVICE_ANNOTATION = "device"
ANNOTATION_TYPES = [LIGHT_ANNOTATION, SENSOR_ANNOTATION, DEVICE_ANNOTATION]
# Folder names
ANNOTATION_FOLDER_NAME = "annotation"
INFO_FOLDER_NAME = "info"
RAW_FOLDER_NAME = "raw"
SUMMARY_FOLDER_NAME = "summary"
ONE_HZ_FOLDER_NAME = "1Hz"
FIFTY_HZ_FOLDER_NAME = "50Hz"
# Name of the combined file
COMBINED_FILE_NAME = "combined.mkv"

# Identifier for powermeters whose devices change during recording
CHANGING_DEVICE = "changing"
# The filename where changing device information is stored. 
# NOTE: currently only one changing device is supported
CHANGING_DEVICE_INFO_FILENAME = "changingDevice.csv"

# Filenames for mapping and appliance info
DEVICE_MAPPING_FILENAME = "deviceMapping.json"
DEVICE_INFO_FILENAME = "deviceInfo.json"
LIGHTS_POWER_INFO_FILENAME = "lightsPower.json"

# What is seen as actual NIGHT hour 
# Within this time period (hours am), base power is extracted
BASE_NIGHT_RANGE = [1,5]


def __checkBase():
    """Check if base folder is set"""
    if FIRED_BASE_FOLDER is None: sys.exit("\033[91mNeed to set FIRED basefolder Folder\033[0m")


def time_format_ymdhms(dt:Union[datetime, float]) -> str:
    """
    Return time format as y.m.d h:m:s.

    :param dt: The timestamp or date to convert to string
    :type  dt: datetime object or float

    :return: Timeformat as \"y.m.d h:m:s\"
    :rtype: str
    """
    if dt is None: return "(None)"
    if (isinstance(dt, datetime) is False
            and isinstance(dt, timedelta) is False):
        dt = datetime.fromtimestamp(dt)
    return "%s.%s" % (
        dt.strftime('%Y.%m.%d %H:%M:%S'),
        str("%03i" % (int(dt.microsecond/1000)))
    )


def filenameToTimestamp(filename: str, format: str="%Y_%m_%d__%H_%M_%S") -> Optional[float]:
    r"""
    Return time stamp of a given file

    :param filename: filename or filepath
    :type  filename: str 
    :param format: format of time in filename, default: \"%Y_%m_%d__%H_%M_%S\"
    :type  format: str

    :return: Timestamp or None if it can not be extracted
    :rtype: float or None
    """
    timestr = os.path.basename(filename).split(".")[0]
    timestr = "_".join(timestr.split("_")[1:])
    try: d = datetime.strptime(timestr, format)
    except ValueError: d = None
    if d is not None: return d.timestamp()
    return None


def prettyfyApplianceName(string:str) -> str:
    return " ".join([s[0].upper() + s[1:] for s in string.split(" ")])

def loadCSV(filepath: str, delimiter: str=DELIMITER) -> List[dict]:
    """
    Load CSV data from given file.
    First row in file determines dictionary keys

    :param filepath: filepath
    :type  filepath: str 
    :param delimiter: column seperator in file.
    :type  delimiter: str

    :return: Data in csv 
    :rtype: list
    """
    def dateparse(timestamp:float):
        return datetime.fromtimestamp(float(timestamp))
    data = pd.read_csv(filepath, delimiter=delimiter, parse_dates=True, date_parser=dateparse).to_dict('r')
    return data

  
def writeCSV(filepath: str, dataList: list, keys: List[str]=[], delimiter: str=DELIMITER):
    """
    Write data to given CSV file.
    If keys are not given, all keys of first entry in datalist are used.
    All list entries should have the same dictionary keys

    :param filepath: filepath
    :type  filepath: str 
    :param dataList: Data as list of dictionaries
    :type  dataList: list 
    :param keys: Keys of dictionary as list. If not given explicitly, all keys in list[0] entry are use.
    :type  keys: List of str 
    :param delimiter: column seperator in file.
    :type  delimiter: str

    :return: Data in csv 
    :rtype: dict
    """
    if len(dataList) == 0: return
    # Data must be a list of dictionaries
    if len(keys) == 0: keys = list(dataList[0].keys())
    try:
        file = open(filepath, 'w+')
        file.write(delimiter.join(keys) + "\n")
        for event in dataList:
            file.write(delimiter.join([str(event[key]) for key in keys]) + "\n")
    except Exception as e:
        print(e)


def __openJson(file:str) -> Union[dict, None]:
    """
    Open given json file.

    :param file: filepath
    :type  file: str 

    :return: Data in json file 
    :rtype: dict or None
    """
    mapping = None
    with open(file) as json_file:
        mapping = json.load(json_file) 
    return mapping


def getUnmonitoredDevices() -> List[str]:
    """Return list of all appliances in dataset that have no annotation and no dedicated meter."""
    unmetered = getUnmeteredDevices()
    lights = [l["name"] for l in loadAnnotations(LIGHT_ANNOTATION, loadData=False)]
    unmonitored = [m for m in unmetered if m not in lights]
    return unmonitored


def getUnmeteredDevices() -> List[str]:
    """Return list of all appliances in dataset that no dedicated meter attached."""
    allDevices = getDeviceInfo()
    deviceMapping = getDeviceMapping()
    # All directly metered appliances
    meteredAppliances = []
    for k in deviceMapping:
        meteredAppliances.extend(deviceMapping[k]["appliances"])
    meteredAppliances.extend(getChangingDevices())
    unmetered = [m for m in allDevices if m not in meteredAppliances]
    return unmetered


def getSmartMeter() -> Optional[str]:
    """Return smartmeter name used in recording."""
    mapping = getDeviceMapping()
    # Identifier for smartmeter is meter with phase 0
    try: return next(key for key in mapping if mapping[key]["phase"] == 0)
    except StopIteration: return None


def getMeterList() -> Optional[List[str]]:
    """Return smartmeter name used in recording."""
    mapping = getDeviceMapping()
    try: return [key for key in mapping if mapping[key]["phase"] != 0]
    except StopIteration: return None


def getChangingMeter() -> Optional[str]:
    """Return name for the meter for which the connected appliance changes."""
    mapping = getDeviceMapping()
    try: return next(key for key in mapping if CHANGING_DEVICE in mapping[key]["appliances"])
    except StopIteration: return None


def getChangingDevices() -> list:
    """Return all appliances connected to the changing meter."""
    info = getChangingDeviceInfo()
    return list(set(i["name"].lower() for i in info))


def getChangingDeviceInfo() -> List[dict]:
    """Return info for appliances connected to the changing meter."""
    __checkBase()
    return loadCSV(os.path.join(FIRED_BASE_FOLDER, ANNOTATION_FOLDER_NAME, CHANGING_DEVICE_INFO_FILENAME))


def getDeviceMapping() -> dict:
    """Return mapping from recording meter to connected appliances."""
    __checkBase()
    # devInfo = getDeviceInfo()
    # meters = set([devInfo[d]["submeter"] for d in devInfo])
    # devMapping = {}
    # for meter in meters:
    #     phase = next(devInfo[a]["phase"] for a in devInfo)
    #     appliances = [a for a in devInfo if devInfo[a]["submeter"] == meter]
    #     if len(appliances) > 0:
    #         if devInfo[appliances[0]]["timedMeter"]: appliances = ["changing"]
    #     devMapping[meter] = {"phase":phase,"appliances":appliances}
    return __openJson(os.path.join(FIRED_BASE_FOLDER, INFO_FOLDER_NAME, DEVICE_MAPPING_FILENAME))


def getDeviceInfo() -> dict:
    """Return info of all appliances."""
    __checkBase()
    return __openJson(os.path.join(FIRED_BASE_FOLDER, INFO_FOLDER_NAME, DEVICE_INFO_FILENAME))


def getLightsPowerInfo() -> dict:
    """Return power info of all lights."""
    __checkBase()
    return __openJson(os.path.join(FIRED_BASE_FOLDER, INFO_FOLDER_NAME, LIGHTS_POWER_INFO_FILENAME))


def get50HzSummaryPath() -> str:
    """Return folder where 50 Hz is stored."""
    __checkBase()
    return os.path.join(FIRED_BASE_FOLDER, SUMMARY_FOLDER_NAME, FIFTY_HZ_FOLDER_NAME)


def get1HzSummaryPath() -> str:
    """Return folder where 1 Hz is stored."""
    __checkBase()
    return os.path.join(FIRED_BASE_FOLDER, SUMMARY_FOLDER_NAME, ONE_HZ_FOLDER_NAME)


def getSummaryFilePath() -> str:
    """Return filepath of combined 1 Hz summary data."""
    __checkBase()
    return os.path.join(FIRED_BASE_FOLDER, SUMMARY_FOLDER_NAME, ONE_HZ_FOLDER_NAME, COMBINED_FILE_NAME)


def getAnnotationPath() -> str:
    """Return folder where annotation data is stored."""
    __checkBase()
    return os.path.join(FIRED_BASE_FOLDER, ANNOTATION_FOLDER_NAME)


def getRecordingRange(startStr: Optional[str]=None, endStr: Optional[str]=None) -> Tuple[float, float]:
    """
    Return start and stop timestamp of recording.
    If start and/or end is given, max(recordingStart, start) and min(recordingStop, end) is given.

    :param startStr: start timestamp in string representation that is checked for validity or None
    :type  startStr: str or None
    :param stopStr: start timestamp in string representation  that is checked for validity or None
    :type  startStr: str or None

    :return: start and end timestamp 
    :rtype: Tuple(float, float)
    """
    summaryPath = get50HzSummaryPath()
    firstFolder = os.path.join(summaryPath, next(os.path.join(summaryPath, p) for p in os.listdir(summaryPath) if os.path.isdir(os.path.join(summaryPath, p))))
    
    allFiles = sorted([os.path.join(firstFolder, p) for p in os.listdir(firstFolder) if os.path.isfile(os.path.join(firstFolder, p))])
    
    start = filenameToTimestamp(allFiles[0])
    durLast = info(allFiles[-1])["stream0"]["duration"]
    end = filenameToTimestamp(allFiles[-1]) + durLast
    # if end is less than 2 seconds away from full day, use full day 
    endDate = datetime.fromtimestamp(end)
    nextDay = endDate.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    if (nextDay.timestamp() - endDate.timestamp()) < 2: end = nextDay.timestamp()

    if startStr is not None:
        if len(startStr.split(" ")) > 1: startTs = datetime.strptime(startStr, "%Y.%m.%d %H:%M:%S").timestamp()
        else: startTs = datetime.strptime(startStr, "%Y.%m.%d").timestamp()
        start = max(startTs, start)
    if endStr is not None:
        if len(endStr.split(" ")) > 1: stopTs = datetime.strptime(endStr, "%Y.%m.%d %H:%M:%S").timestamp()
        else: stopTs = datetime.strptime(endStr, "%Y.%m.%d").timestamp()
        end = min(stopTs, end)
    
    return start, end
    

def loadAnnotationInfo(filepath: str) -> Optional[dict]:
    """
    Extract info from annotation filename.

    :param filepath: filepath of annotation file
    :type  filepath: str

    :return: Dictionary with extracted annotation info or None 
    :rtype: None or dict
    """
    splits = os.path.basename(filepath).split(".")[0].split(DIVIDER)
    type = splits[0].lower()
    if type not in ANNOTATION_TYPES: return None
    room = splits[1].replace("_"," ")    
    if type in [LIGHT_ANNOTATION, DEVICE_ANNOTATION]:
        name = splits[2].replace("_"," ")
    elif type == SENSOR_ANNOTATION:
        name = room + " " + splits[2].replace("hum","humidity").replace("temp","temperature")
    return {"type":type, "room":room, "name":name, "file":filepath}


def loadAnnotationFile(filepath: str) -> Optional[dict]:
    """
    Extract info and load data from annotation filename.

    :param filepath: filepath of annotation file
    :type  filepath: str

    :return: Dictionary with extracted annotation info and data or None 
    :rtype: None or dict
    """
    info = loadAnnotationInfo(filepath)
    if info is None: return None
    info["data"] = loadCSV(filepath)
    return info


def loadAnnotations(type: str, loadData: bool=True) -> Optional[list]:
    """
    Load all annotations of given type. 

    :param type: Annotation type, must be in ANNOTATION_TYPES
    :type  type: str
    :param loadData: Load only info to speed up things or also data 
    :type  loadData: bool

    :return: List of Dictionary with extracted annotation info and data or None 
    :rtype: None or list
    """
    if type not in ANNOTATION_TYPES: return None
    annotationFolder = os.path.join(FIRED_BASE_FOLDER, ANNOTATION_FOLDER_NAME)
    files = sorted([os.path.join(annotationFolder, o) for o in os.listdir(annotationFolder) if type+"__" in o])
    if loadData: getter = loadAnnotationFile
    else: getter = loadAnnotationInfo
    annos = [getter(file) for file in files]
    return annos


def getPhase(meterOrAppliance: str) -> int:
    """
    Return the live wire the device is connected to. 

    :param meterOrAppliance: meter name or appliance name
    :type  meterOrAppliance: str

    :return: live wire of connected device, -1 if unknown
    :rtype: int
    """
    deviceMapping = getDeviceMapping()
    deviceInfo = getDeviceInfo()
    if meterOrAppliance in deviceMapping:
        return deviceMapping[meterOrAppliance]["phase"]
    elif meterOrAppliance in deviceInfo:
        return deviceInfo[meterOrAppliance]["phase"]
    else: 
        return -1


def convertToTimeRange(data: list, clipLonger: Optional[float]=None, clipTo: float=10*60) -> List[dict]:
    r"""
    Convert given annotation data to time range. 
    Range is determined between two entries.
    e.g. TS1 off data + TS2 on data -> [off startTs=TS1 stopTs=TS2]

    .. code-block:: python3

        data = [
                { "timestamp": <TS1>, <data> }, 
                { "timestamp": <TS2>, <data> }
            ]

    gets converted to:

    .. code-block:: python3

        data = [
                { startTs: <TS1>, stopTs: <TS2>, <data> }, 
                { startTs: <TS2>, stopTs: <TS2>+clipTo, <data> }
            ]

    :param data: List of annotation entries
    :type  data: list
    :param clipLonger: States longer that are clipped to parameter clipTo 
    :type  clipLonger: None or float
    :param clipTo: States longer than clipLonger are clipped to the given value 
    :type  clipTo: float, default: 10 minutes

    :return: List of Dictionary with converted info
    :rtype: list
    """
    rangeData = []
    for i, entry in enumerate(data):
        start = entry["timestamp"]
        if i == len(data)-1: 
            end = start + clipTo
        else: 
            end = data[i+1]["timestamp"]
        # State longer than 24 hours?
        if clipLonger is not None and end-start > clipLonger: end = start + clipTo
        # Copy over old entries
        newEntry = {k:v for k,v in entry.items()}
        newEntry["startTs"] = start 
        newEntry["stopTs"] = end
        rangeData.append(newEntry)
    return rangeData


def getApplianceList(startTs: Optional[float]=None, stopTs: Optional[float]=None) -> list:
    """
    Return list of appliance names active in between the given time range.
    Active is defined as metered or connected to changing device or light turned on.
    NOTE: What about stove?
    
    :param startTs: Time range start
    :type  startTs: None or float
    :param stopTs: Time range stop
    :type  stopTs: None or float´

    :return: List of appliance names
    :rtype: list
    """
    __checkBase()

    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]
    deviceMapping = getDeviceMapping()
    devices = [] 
    for key in deviceMapping: devices.extend(deviceMapping[key]["appliances"])
    devices = [d for d in devices if d not in ["changing","L1","L2","L3"]]

    cdInfo = getChangingDeviceInfo()
    changingDevices = list(set([cdI["name"] for cdI in cdInfo if cdI["startTs"] < stopTs and cdI["stopTs"] > startTs]))

    lightsInfo = loadAnnotations(LIGHT_ANNOTATION, loadData=False)
    lights = [l["name"] for l in lightsInfo]
    appliances = sorted(list(set(devices + changingDevices + lights)))
    
    return appliances


def resampleRecord(data: np.recarray, inRate: float, outRate: float) -> np.recarray:
    """
    Resample a given numpy record array
    
    :param startTs: Time range start
    :type  startTs: None or float
    :param stopTs: Time range stop
    :type  stopTs: None or float

    :return: List of appliance names
    :rtype: list
    """
    if inRate == outRate: return data
    resampleFac = inRate/outRate
    # NOTE: This is done for each measure
    # TODO: Maybe we can make this quicker somehow
    oldX = np.arange(0, len(data))
    newX = np.arange(0, len(data), resampleFac)
    data2 = np.zeros(len(newX), dtype=data.dtype)
    for measure in data.dtype.names:
        data2[measure] = np.interp(newX, oldX, data[measure])
    data = data2
    return data


def bestBasePowerTimeRange(startTs: float, stopTs: float) -> List[dict]:
    """
    Return time ranges to extract base power from given a super time range.
    If no night lies between the time range, the time before the given night is used.
    NOTE: Will cause problems for datasets that start within a day and only this day given
    
    :param startTs: Time range start
    :type  startTs: float
    :param stopTs: Time range stop
    :type  stopTs: float

    :return: List of dict with startTs and stopTs keys as time ranges
    :rtype: list
    """
    # More than a day between data
    startDate = datetime.fromtimestamp(startTs)
    stopDate = datetime.fromtimestamp(stopTs)
    
    timeranges = []
    date = startDate
    # if we cannot use startnight
    if  startDate.hour >= BASE_NIGHT_RANGE[0]: date = date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    while date <= stopDate:
        tsStart = date.replace(hour=BASE_NIGHT_RANGE[0], minute=0, second=0, microsecond=0).timestamp()
        tsStop = date.replace(hour=BASE_NIGHT_RANGE[1], minute=0, second=0, microsecond=0).timestamp()
        timeranges.append({"startTs":tsStart, "stopTs":tsStop})
        date += timedelta(days=1)
    # if we cannot use stopnight
    if stopDate.hour < BASE_NIGHT_RANGE[1]: del timeranges[-1]
    
    if len(timeranges) == 0:
        # use night before then
        tsStart = startDate.replace(hour=BASE_NIGHT_RANGE[0], minute=0, second=0, microsecond=0).timestamp()
        tsStop = startDate.replace(hour=BASE_NIGHT_RANGE[1], minute=0, second=0, microsecond=0).timestamp()
        timeranges.append({"startTs":tsStart, "stopTs":tsStop})
    return timeranges


def getBasePower(samplingrate: int, startTs: Optional[float]=None, stopTs: Optional[float]=None, phase: Union[None,int,List[int]]=None) -> List[dict]:
    """
    Return base power dict with given samplingrate for given time and phase. 
    
    :param samplingrate: samplingrate of returned power
    :type  samplingrate: int
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None
    :param phase: Given phase (grid line L<phase>). Either 1,2,3 or a combination as list or None
    :type  phase: list(int), int or None

    :return: List of dict with power data
    :rtype: list(dict)
    """
    __checkBase()
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]
    if phase is None: phase = [1,2,3]
    if not isinstance(phase, list): phase = [phase]
    # get ranges from where to compute base powers
    ranges = bestBasePowerTimeRange(startTs, stopTs)
    smartmeterName = getSmartMeter()
    meters = getMeterList()
    deviceMapping = getDeviceMapping()

    # Construct yet empty base power list
    basePowers = {}
    newSize = int((stopTs-startTs)*samplingrate)
    rangeSize = sum(int((r["stopTs"]-r["startTs"])*samplingrate) for r in ranges)
    powers = ["p","q","s"]
    dt = [(m, '<f4') for m in powers]
    for p in phase:
        data = np.recarray((rangeSize,), dtype=dt).view(np.recarray)
        for m in powers: data[m] = 0
        dataDict = {"title":"basepower", "name":"basepower l" + str(p), "phase":p, 
                    "data": data, "timestamp":startTs, "type":"audio",
                    "samplingrate":samplingrate, "measures":powers}
        basePowers[p] = dataDict
    
    # Loop over all ranges
    index = 0
    for r in ranges:
        # get smartmeter data
        smData = getMeterPower(smartmeterName, samplingrate, startTs=r["startTs"], stopTs=r["stopTs"])["data"]
        # add it for each phase
        for p in phase:
            for m in powers: 
                basePowers[p]["data"][m][index:index+len(smData)] = smData[m + "_l" + str(p)]
        # load meter data
        for meter in meters:
            p = deviceMapping[meter]["phase"] 
            if p not in phase: continue
            mData = getMeterPower(meter, samplingrate, startTs=r["startTs"], stopTs=r["stopTs"])["data"]
            # Substract it from each phase
            for m in powers: basePowers[p]["data"][m][index:index+len(mData)] -= mData[m]
        index += len(smData)

    # Calculate base power
    for p in phase:
        data = np.recarray((newSize,), dtype=dt).view(np.recarray)
        for m in powers:
            hist, bin_edges = np.histogram(basePowers[p]["data"][m])
            # Sort histogram based on bins with most entries
            idx = list(reversed(np.argsort(hist)))[:2]
            # Mean of 2 bins with most entries in historgram
            mean = np.sum([bin_edges[i]*hist[i] for i in idx])/np.sum([hist[i] for i in idx])
            data[m][:] = mean
        basePowers[p]["data"] = data
    
    return [basePowers[p] for p in phase]


def getPowerStove(samplingrate: int, startTs: Optional[float]=None, stopTs: Optional[float]=None, phase: Union[None,int,List[int]]=None) -> List[dict]:
    """
    Reconstruct power of stove, if it is not directly monitored, it might be reconstructable form smartmeter data.

    :param samplingrate: samplingrate of returned power
    :type  samplingrate: int
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None
    :param phase: Given phase (grid line L<phase>). Either 1,2,3 or a combination as list or None
    :type  phase: list(int), int or None

    :return: List of dict with power data
    :rtype: list(dict)
    """
    __checkBase()
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]
    if phase is None: phase = [1,2,3]
    if not isinstance(phase, list): phase = [phase]
    # Init return dict
    data = {}
    for p in phase: data[p] = {"phase":p,"samplingrate":samplingrate,"title":"stove","name":"stove l" + str(p),"data":None}
    # Get smartmeter
    smartmeterName = getSmartMeter()
    # Calculate base power consumption
    base = getBasePower(samplingrate, startTs=startTs, stopTs=stopTs, phase=phase)
    # Get total power consumption
    smartmeterData = getMeterPower(smartmeterName, samplingrate, startTs=startTs, stopTs=stopTs)
    # Get individual meter data
    deviceMapping = getDeviceMapping()
    # All power meter within that phase
    powerMeters = [m for m in getMeterList() if deviceMapping[m]["phase"] in phase]
    # load their power
    allMeterPower = [getMeterPower(name, samplingrate, startTs=startTs, stopTs=stopTs) for name in powerMeters]
    for meter in allMeterPower:
        meterName = meter["title"]
        p = deviceMapping[meterName]["phase"]
        if data[p]["data"] is None: 
            data[p]["data"] = meter["data"]
            data[p]["measures"] = meter["measures"]
            data[p]["timestamp"] = meter["timestamp"]
        else: 
            for m in data[p]["measures"]: data[p]["data"][m] += meter["data"][m]

    # Lights are neglected, as oven consumes way more power
    for p in data:
        b = next(b for b in base if b["phase"] == p)
        for m in data[p]["measures"]: data[p]["data"][m] = smartmeterData["data"][m + "_l" + str(p)] - data[p]["data"][m] - b["data"][m]
        for m in data[p]["measures"]: data[p]["data"][m][data[p]["data"]["s"] < 800] = 0
        # peaks, props = find_peaks(data[p]["data"]["s"], threshold=800, width=1)
        # Filter peaks which are smaller than 2s as this cannot be the stove
        peaks, props = find_peaks(data[p]["data"]["s"], threshold=800, width=(1, int(1.0*samplingrate)))
        # There may be remaining peaks from slightly misaligned data at higher samplingrates, we want to remove them
        for m in data[p]["measures"]:
            # Force them to be zero
            data[p]["data"][m][peaks] = 0
    return [data[p] for p in data]


def getReconstructibleDevices() -> dict:
    """Return dict for reconstructible devices with handler function"""
    rec = {"stove":getPowerStove}
    return rec


def getMeterPower(meter: str, samplingrate: float, startTs: Optional[float]=None, stopTs: Optional[float]=None, verbose: int=0) -> dict:
    """
    Return power of given meter. 

    :param meter: Name of meter; must be in getMeterList().
    :type  meter: str
    :param samplingrate: samplingrate of returned power
    :type  samplingrate: int
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None
    :param verbose: Enable some verbose output while loading data, 0 for no output
    :type  verbose: int, default: 0

    :return: power data
    :rtype: dict
    """
    __checkBase()
    
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]
    # Use data depending on goal samplingrate
    if samplingrate > 1: folder = os.path.join(get50HzSummaryPath(), meter.lower())
    else: folder = os.path.join(get1HzSummaryPath(), meter.lower())
    startDate = datetime.fromtimestamp(startTs).replace(hour=0, minute=0, second=0, microsecond=0)
    stopDate = datetime.fromtimestamp(stopTs)
    # Special case of data until midnight (next day wont be loaded)
    if stopDate == stopDate.replace(hour=0, minute=0, second=0, microsecond=0): stopDate = stopDate - timedelta(days=1)
    else: stopDate = stopDate.replace(hour=0, minute=0, second=0, microsecond=0)
    
    dates = [startDate]
    while startDate < stopDate:
        startDate += timedelta(days=1)
        dates.append(startDate)
    files = [os.path.join(folder, meter + "_" + date.strftime(STORE_TIME_FORMAT) + ".mkv") for date in dates]
    for file in files:
        if verbose: print(file)
        if not os.path.exists(file): sys.exit("\033[91mFile {} does not exist\033[0m".format(file))
    if verbose: print("{}: Loading MKV...".format(meter), end="", flush=True)
    data = [loadAudio(file)[0] for file in files]
    if verbose: print("Done")
    if len(data) < 1: return None
    dataNice = data[0]
    # Concat data of several files
    if len(data) > 1:
        for d in data[1:]:
            dataNice["data"] = np.concatenate((dataNice["data"], d["data"]))
    data = dataNice
    # Using summary file
    # else:
    #     # This is using the summary file, maybe we can use individual files to boost things
    #     file = getSummaryFilePath()
    #     if verbose: print("{}: Loading 1Hz combined...".format(meter), end="", flush=True)
    #     dataList = loadAudio(file)
    #     if verbose: print("Done")
    #     try: data = next(d for d in dataList if d["title"] == meter)
    #     except StopIteration: return None
        
    fromSample = int((startTs - data["timestamp"])*data["samplingrate"])
    toSample = int((stopTs - data["timestamp"])*data["samplingrate"])
    data["data"] = data["data"][fromSample:toSample]
    data["timestamp"] = startTs
    if verbose: 
        print("{}->{}: len({})".format(time_format_ymdhms(startTs), time_format_ymdhms(stopTs), len(data["data"])))
    if samplingrate != 1 and samplingrate != 50:
        if verbose: print("resampling")
        data["data"] = resampleRecord(data["data"], data["samplingrate"], samplingrate)
        data["samplingrate"] = samplingrate
    goalSamples = int(data["samplingrate"]*(stopTs - startTs))
    if abs(goalSamples - len(data["data"])) > data["samplingrate"]:
        print("\033[91mError loading data for {}. Requested samples: {}, actual samples: {}\033[0m".format(meter, goalSamples, len(data["data"])))
    if goalSamples > len(data["data"]):
        new = np.recarray((goalSamples-len(data["data"]),), dtype=data["data"].dtype).view(np.recarray)
        if abs(goalSamples - len(data["data"])) > data["samplingrate"]: new[:] = 0
        else: new[:] = data["data"][-1]
        data["data"] = np.concatenate((data["data"], new))
    elif goalSamples < len(data["data"]):
        data["data"] = data["data"][:goalSamples]
    deviceMapping = getDeviceMapping()
    data["phase"] = deviceMapping[meter]["phase"]
    return data


def getPower(appliance: str, samplingrate: int, startTs: Optional[float]=None, stopTs: Optional[float]=None) -> Union[list, dict]:
    """
    Return power of given appliance. 

    :param appliance: Name of appliance; must be in getApplianceList().
    :type  appliance: str
    :param samplingrate: samplingrate of returned power
    :type  samplingrate: int
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None

    :return: dict of power data or list of dict for devices connected to multiple phases
    :rtype: dict or list
    """
    __checkBase()
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]

    # See if data is in one of the meters
    deviceMapping = getDeviceMapping()
    try: 
        meter = next(dev for dev in deviceMapping if appliance in deviceMapping[dev]["appliances"])
        data = getMeterPower(meter, samplingrate, startTs=startTs, stopTs=stopTs)
        data["name"] = appliance
        return data
    except StopIteration:pass
    
    # See if data is in the changing device meter
    changingDeviceInfo = [cdI for cdI in getChangingDeviceInfo() if cdI["startTs"] < stopTs and cdI["stopTs"] > startTs and cdI["name"].lower() == appliance]
    if len(changingDeviceInfo) > 0:
        data = getMeterPower(getChangingMeter(), samplingrate, startTs=startTs, stopTs=stopTs)
        # This is a zero array
        cleaned = np.recarray((len(data["data"]),), dtype=data["data"].dtype).view(np.recarray)
        cleaned[:] = 0
        # Fill with entries from changing file
        for info in changingDeviceInfo:
            startSample = int(((info["startTs"] - 60) - data["timestamp"])*data["samplingrate"])
            stopSample = int(((info["stopTs"] + 60) - data["timestamp"])*data["samplingrate"])
            cleaned[startSample:stopSample] = data["data"][startSample:stopSample]
        data["data"] = cleaned
        data["name"] = appliance
        return data
    
    # See if in lights
    lights = [l["name"] for l in loadAnnotations(LIGHT_ANNOTATION, loadData=False)]
    if appliance in lights:
        return getPowerForLight(appliance, samplingrate, startTs=startTs, stopTs=stopTs)

    unmonitored = getUnmonitoredDevices()
    if appliance in unmonitored:
        print("Unmonitored device found: " + str(appliance))
        rec = getReconstructibleDevices()
        if appliance in rec:
            return rec[appliance](samplingrate, startTs=startTs, stopTs=stopTs)
    return None


def getPowerForAppliances(samplingrate: int, startTs: Optional[float]=None, stopTs: Optional[float]=None, appliances: Union[None,str,List[str]]=None) -> List[dict]:
    """
    Return power of all appliance or appliances in list. 

    :param samplingrate: samplingrate of returned power
    :type  samplingrate: int
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None
    :param appliances: List with appliance name; must be in getApplianceList().
    :type  appliances: list(str)

    :return: List of power dictionaries
    :rtype: List(dict)
    """
    __checkBase()
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]
    # Convert simple string to list of strings
    if appliances is not None:
        if not isinstance(appliances, list): appliances = [appliances]
    deviceInfo = getDeviceInfo()
    applianceList = sorted(list(set(deviceInfo.keys())))
    # Union
    applianceList = [ap for ap in applianceList if ap in appliances]
    pows = []
    for ap in applianceList:
        power = getPower(ap, samplingrate, startTs=startTs, stopTs=stopTs)
        if isinstance(power, list): pows.extend(power)
        else: pows.append(power)
    return pows


def getPowerForLight(light: str, samplingrate: int, startTs: Optional[float]=None, stopTs: Optional[float]=None) -> dict:
    """
    Return power for given light. 

    :param light: name of light
    :type  light: str
    :param samplingrate: samplingrate of returned power
    :type  samplingrate: int
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None

    :return: power dictionary
    :rtype: dict
    """
    __checkBase()
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]
    # Load light info
    try: lightData = next(l for l in loadAnnotations(LIGHT_ANNOTATION, loadData=False) if l["name"] == light)
    except StopIteration: return None

    deviceInfo = getDeviceInfo()
    # Load mapping from light model to power consumption
    powerInfo = getLightsPowerInfo()
    # matching is again lowercase
    powerInfo = {k.lower(): v for k, v in powerInfo.items()}

    if lightData["name"] not in deviceInfo:
        print("Light {} not in device info, this should not happen".format(lightData["name"]))
        return None
    lightModel = str(deviceInfo[lightData["name"]]["brand"] + " " + deviceInfo[lightData["name"]]["model"]).lower()
    if lightModel not in powerInfo:
        print("Light {} not in mapping, this should not happen".format(lightModel))
        return None
    lightPower = powerInfo[lightModel]
    # Load the data as time range
    lightData["data"] = convertToTimeRange(loadCSV(lightData["file"]), clipLonger=12*60*60, clipTo=10*60)

    # Power measures to compute
    dt = [(key, np.float32) for key in ["p","q","s"]]
    # Total duration and size calc
    duration = stopTs - startTs
    newSize = int(duration*samplingrate)
    total = np.recarray((newSize,), dtype=dt).view(np.recarray)

    pBase = lightPower["activePowerBase"]
    qBase = lightPower["reactivePowerBase"]
    sBase = np.sqrt(pBase**2+qBase**2)

    pFull = lightPower["activePowerFull"] - pBase
    qFull = lightPower["reactivePowerFull"] - qBase
    sFull = np.sqrt(pFull**2+qFull**2) - sBase

    total[:]['p'] = pBase
    total[:]['q'] = qBase
    total[:]['s'] = sBase
    for entry in lightData["data"]:
        if entry["startTs"] > stopTs or entry["stopTs"] < startTs: continue
        if "state" not in entry or entry["state"].lower() == "off": continue
        start = max(entry["startTs"], startTs)
        end = min(entry["stopTs"], stopTs)

        startSample = int((start-startTs)*samplingrate)
        stopSample = int((end-startTs)*samplingrate)

        dimm = 1.0
        if "dimm" in entry: dimm = entry["dimm"]/100.0
        total[startSample:stopSample]['p'] = pBase + dimm*pFull
        total[startSample:stopSample]['q'] = qBase + dimm*qFull
        total[startSample:stopSample]['s'] = sBase + dimm*sFull
    lightData["data"] = total
    lightData["samplingrate"] = samplingrate
    lightData["timestamp"] = startTs
    lightData["phase"] = deviceInfo[lightData["name"]]
    return lightData


def getPowerForLights(samplingrate: int, startTs: Optional[float]=None, stopTs: Optional[float]=None, lights: Union[None,str,List[str]]=None) -> List[dict]:
    """
    Return power for all or given lights. 

    :param samplingrate: samplingrate of returned power
    :type  samplingrate: int
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None
    :param lights: name of lights 
    :type  lights: None, list(str) or str

    :return: list of power dictionaries
    :rtype: list(dict)
    """
    __checkBase()
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]
    # Convert simple string to list of strings
    if lights is not None:
        if not isinstance(lights, list): lights = [lights]
    # Load light info
    lightsData = loadAnnotations(LIGHT_ANNOTATION, loadData=False)
    # Load device info to get model of light
    deviceInfo = getDeviceInfo()
    # matching is lowercase
    deviceInfo =  {k.lower(): v for k, v in deviceInfo.items()}
    # Load mapping from light model to power consumption
    powerInfo = getLightsPowerInfo()
    # matching is again lowercase
    powerInfo = {k.lower(): v for k, v in powerInfo.items()}

    # Power measures to compute
    dt = [(key, np.float32) for key in ["p","q","s"]]
    # Total duration and size calc
    duration = stopTs - startTs
    newSize = int(duration*samplingrate)
    # Check if we would generate too much data
    if newSize > 60*60*24*30*50: input("getPowerForLights: {} bytes? Thats a lot of data! sure? ctr-c to cancel.".format(newSize*3*len(lights)))
    # zeros when lights are off
    # zeros = np.recarray((newSize,), dtype=dt).view(np.recarray)

    returnedLight = []
    for light in lightsData:
        if lights is not None and light["name"] not in lights: continue
        if light["name"] not in deviceInfo:
            print("Light {} not in device info, this should not happen".format(light["name"]))
            continue
        device = str(deviceInfo[light["name"]]["brand"] + " " + deviceInfo[light["name"]]["model"]).lower()
        if device not in powerInfo:
            print("Light {} not in mapping, this should not happen".format(device))
            continue
        lightPower = powerInfo[device]
        # print("{}: {}".format(light["name"], lightPower))
        # Load the data as time range
        light["data"] = convertToTimeRange(loadCSV(light["file"]), clipLonger=12*60*60, clipTo=10*60)

        total = np.recarray((newSize,), dtype=dt).view(np.recarray)

        pBase = lightPower["activePowerBase"]
        qBase = lightPower["reactivePowerBase"]
        sBase = np.sqrt(pBase**2+qBase**2)

        pFull = lightPower["activePowerFull"] - pBase
        qFull = lightPower["reactivePowerFull"] - qBase
        sFull = np.sqrt(pFull**2+qFull**2) - sBase

        total[:]['p'] = pBase
        total[:]['q'] = qBase
        total[:]['s'] = sBase
        for entry in light["data"]:
            if entry["startTs"] > stopTs or entry["stopTs"] < startTs: continue
            if "state" not in entry or entry["state"].lower() == "off": continue
            start = max(entry["startTs"], startTs)
            end = min(entry["stopTs"], stopTs)

            startSample = int((start-startTs)*samplingrate)
            stopSample = int((end-startTs)*samplingrate)

            dimm = 1.0
            if "dimm" in entry: dimm = entry["dimm"]/100.0
            total[startSample:stopSample]['p'] = pBase + dimm*pFull
            total[startSample:stopSample]['q'] = qBase + dimm*qFull
            total[startSample:stopSample]['s'] = sBase + dimm*sFull
        light["csvData"] = list(light["data"])
        light["data"] = total
        light["samplingrate"] = samplingrate
        light["timestamp"] = startTs
        light["phase"] = deviceInfo[light["name"]]["phase"]
        returnedLight.append(light)
    return returnedLight


def getTsDurRateMeas(file: str) -> Tuple[float, float, int, List[str]]:
    """
    Return simple info tuple for mkv file. 
    Tuple is:
    - timestamp of first sample in file
    - duration of file in seconds
    - samplingrate
    - list of measure names

    :param file: mkv filename
    :type  file: str

    :return: Information tuple
    :rtype: Tuple(float, float, int, list)
    """
    nfo = info(file)
    if nfo is None:
        return (filenameToTimestamp(file), 0, 0, [])
    iinfo = nfo["stream0"]
    if "TIMESTAMP" in iinfo["metadata"]: startTs = float(iinfo["metadata"]["TIMESTAMP"])
    else: startTs = filenameToTimestamp(file)
    return (startTs, iinfo["duration"], iinfo["rate"], iinfo["measures"])


def loadAudio(filepath: str, verbose: bool=False, streamsToLoad: Optional[List[int]]=None, titles: Optional[List[str]]=None) -> List[dict]:
    r"""
    Call to load audio from an MKV file to an numpy array the fast way.
    Note: All streams MUST have the same samplingrate!

    As filepath, give the mkv file to load into an array of dictionaries.
    Each dictionary holds information such as metadata, title, name
    dataList has the following structure:

    .. code-block:: python3

        dataList = [
            {
                title:        < streamTitle >,
                streamIndex:  < index of the stream >,
                metadata:     < metadata of the stream >,
                type:         < type of the stream >,
                samplingrate: < samplingrate >,                       # only for audio
                measures:     < list of measures >,                   # only for audio
                data:         < data as recarray with fieldnames >,   # only for audio
                subs:         < subtitles as pysubs2 file >,          # only for subtitles
            },
            ...
        ]

    :param filepath:       filepath to the mkv to open
    :type  filepath:       str
    :param verbose:        increase output verbosity
    :type  verbose:        Bool
    :param streamsToLoad:  List of streams to load from the file, either list of numbers or list of stream titles
                           default: None -> all streams should be loaded
    :type  streamsToLoad:  list
    :param titles:         List or str of streams titles which should be loaded
                           default: None -> no filter based on title names
    :type  titles:         list or str

    :return: List of dictionaries holding data and info
    :rtype: list
    """
    dataList = []
    # Open the container
    try:
        container = av.open(filepath)
    except av.AVError:
        return []
    # Get available stream indices from file
    streams = [s for i, s in enumerate(container.streams) if ( (s.type == 'audio') )]
    # Look if it is a stream to be loaded
    if streamsToLoad is not None: streams = [stream for stream in streams if stream.index in streamsToLoad]
    # Filter for title names
    if titles is not None:
        # Hande title not list case
        if isinstance(titles, str): titles = [titles]
        # Loop over stream metadata and look for title match
        newStreams = []
        for stream in streams:
            key =  set(["Title", "title", "TITLE", "NAME", "Name", "name"]).intersection(set(stream.metadata.keys()))
            if len(key) > 0:
                title = stream.metadata[next(iter(key))]
                if title in titles: newStreams.append(stream)
        streams = newStreams
    indices = [stream.index for stream in streams]
    # load only the data we want
    fn = lambda bc: [stream for stream in bc if stream.index in indices]
    rawdata = av.io.read(fn, file=filepath)
    for data in rawdata:
        stream = data.info
        dataDict = {}
        dataDict["streamIndex"] = stream.index
        dataDict["metadata"] = stream.metadata
        dataDict["type"] = stream.type
        # Try to extract the tile of the stream
        key = set(["Title", "title", "TITLE", "NAME", "Name", "name"]).intersection(set(stream.metadata.keys()))
        if len(key) > 0: title = stream.metadata[next(iter(key))]
        else: title = "Stream " + str(stream.index)
        dataDict["title"] = title
        if stream.type == "audio":
            dataDict["samplingrate"] = stream.sample_rate
            # Try to extract the name of the measures
            for key in ["TIMESTAMP", "Timestamp", "timestamp"]:
                if key in stream.metadata:
                    dataDict["timestamp"] = float(stream.metadata[key])
                    break

            channelTags = channelTags = ["C" + str(i) for i in range(stream.channels)]
            for key in ["CHANNEL_TAGS", "Channel_tags"]:
                if key in stream.metadata:
                    channelTags = stream.metadata[key].split(",")
                    break;
            
            if len(channelTags) != stream.channels:
                print("Maybe wrong meta, as #tags does not match #channels")
                if len(channelTags) > stream.channels: channelTags = channelTags[0:stream.channels]
                else: channelTags = ["C" + str(i) for i in range(stream.channels)]
            dataDict["measures"] = channelTags
            # Audio data will be stored here
            dataDict["data"] = data.transpose()
        # Append to list
        dataList.append(dataDict)

    # Convert data to record array
    for i in range(len(dataList)):
        if dataList[i]["type"] == 'audio':
            # Convert data into structured array
            dataList[i]["data"] = np.core.records.fromarrays(dataList[i]["data"], dtype={'names': dataList[i]["measures"], 'formats': ['f4']*len(dataList[i]["measures"])})
            # Does not work... maybe there is a better way
            # dataList[i]["data"] = np.array(dataList[i]["data"].reshape((2, -1)), dtype=[(measure, np.float32) for measure in dataList[i]["measures"]])
            # print(dataList[i]["data"].shape)
            # print(dataList[i]["data"].dtype)
            # dataList[i]["data"] = data
            if verbose:
                print("Stream : " + str(dataList[i]["streamIndex"]))
                if verbose:
                    print("Title: " + str(dataList[i]["title"]))
                print("Metadata: " + str(dataList[i]["metadata"]))
                if dataList[i]["type"] == "audio":
                    print("Samplingrate: " + str(dataList[i]["samplingrate"]) + "Hz")
                    print("Channels: " + str(dataList[i]["measures"]))
                    print("Samples: "  + str(len(dataList[i]["data"])))
                    print("Duration: " + str(len(dataList[i]["data"])/dataList[i]["samplingrate"]) + "s")
    return dataList


def info(path: str, format: Optional[str]=None, option: list=[]) -> dict:
    """
    Return info of given mkv file.

    :param path:   Filepath of mkv
    :type  path:   str
    :param format: Format of the file, default=None: guessed by file ending
    :type  format: str
    :param option: Options passed to av.open(), default=[]
    :type  option: av options parameter
    """
    options = dict(x.split('=') for x in option)
    try:
        container = av.open(path, format=format, options=options)
    except av.AVError:
        return None
    info = {}
    info["format"] = container.format
    info["duration"] = float(container.duration) / av.time_base
    info["metadata"] = container.metadata
    info["#streams"] = len(container.streams)
    info["streams"] = []
    samples = None
    if container.duration < 0 or container.duration / av.time_base > 24*60*60*100: # this is 100 days
        # Unfortunately duration estimation of ffmpeg is broken for some files, as the files have not been closed correctly.
        # For later days during recording this is fixed. 
        samples = getSamples(path)
    for i, stream in enumerate(container.streams):
        streamInfo = {}
        streamInfo["rate"] = stream.rate
        streamInfo["type"] = stream.type
        if stream.duration is None: streamInfo["duration"] = float(container.duration) / av.time_base
        # Does not seem to work
        else: streamInfo["duration"] = float(stream.duration) / stream.time_base
        if samples is not None:
            streamInfo["duration"] = samples[i]/streamInfo["rate"]
        streamInfo["start_time"] = stream.start_time
        # print(stream.metadata)
        streamInfo["metadata"] = stream.metadata
        key = set(["Title", "title", "TITLE", "NAME", "Name", "name"]).intersection(set(stream.metadata.keys()))
        if len(key) > 0: title = stream.metadata[next(iter(key))]
        else: title = "Stream " + str(stream.index)
        streamInfo["title"] = title
        if stream.type == 'audio':
            streamInfo["format"] = stream.format
            streamInfo["#channels"] = stream.channels
        elif stream.type == 'video':
            streamInfo["format"] = stream.format
        streamInfo["samples"] = int(streamInfo["duration"]*streamInfo["rate"])
        if samples is not None:
            streamInfo["samples"] = samples[i]

        channelTags = channelTags = ["C" + str(i) for i in range(stream.channels)]
        for key in ["CHANNEL_TAGS", "Channel_tags"]:
            if key in stream.metadata:
                channelTags = stream.metadata[key].split(",")
                break;
        streamInfo["measures"] = channelTags
        info["streams"].append(streamInfo)
        info["stream" + str(i)] = streamInfo
    return info


def getSamples(path:str, format:Union[None,str]=None, option:list=[]) -> int:
    """
    Return number of samples in audio file.

    :param path:   Filepath of mkv
    :type  path:   str
    :param format: Format of the file, default=None: guessed by file ending
    :type  format: str
    :param option: Options passed to av.open(), default=[]
    :type  option: av options parameter
    """
    options = dict(x.split('=') for x in option)
    try:
        container = av.open(path, format=format, options=options)
    except av.AVError:
        return 0
    # all streams to be extracted
    streams = [s for s in container.streams]
    samples = [0 for _ in range(len(streams))]
    for i, stream in enumerate(streams):
        try:
            container = av.open(path, format=format, options=options)
        except av.AVError:
            return 0
        # Seek to the last frame in the container
        container.seek(sys.maxsize, whence='time', any_frame=False, stream=stream)
        for frame in container.decode(streams=stream.index):
            samples[i] = int(frame.pts / 1000.0*frame.rate + frame.samples)
    return samples