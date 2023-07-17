import os
import sys
from datetime import datetime, timedelta, timezone
import pytz
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import json
import numpy as np
import av
import math
import subprocess
from scipy.signal import find_peaks, medfilt

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
RAW_FOLDER_NAME = "highFreq"
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
RSYNC_PWD_FILE = "rsync_pass.txt"

RSYNC_ALLOWED = True
RSYNC_ADDR = "rsync://FIRED@clu.informatik.uni-freiburg.de/FIRED/"
VERBOSE = False

# What is seen as actual NIGHT hour 
# Within this time period (hours am), base power is extracted
BASE_NIGHT_RANGE = [1,5]
# These will be deleted on del
_loadedFiles = []


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

def getRSYNCPwdFile() -> str:
    __checkBase()
    return os.path.join(FIRED_BASE_FOLDER, INFO_FOLDER_NAME, RSYNC_PWD_FILE)


def getChangingDevices() -> list:
    """Return all appliances connected to the changing meter."""
    info = getChangingDeviceInfo()
    return list(set(i["name"].lower() for i in info))


__changingInfo = None
def getChangingDeviceInfo(startTs: Optional[float]=None, stopTs: Optional[float]=None) -> List[dict]:
    """Return info for appliances connected to the changing meter."""
    __checkBase()
    global __changingInfo
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]
    if __changingInfo is None:
        __changingInfo = loadCSV(os.path.join(FIRED_BASE_FOLDER, ANNOTATION_FOLDER_NAME, CHANGING_DEVICE_INFO_FILENAME))
        # Add safe margin for changing device
        for row in __changingInfo:
            row["startTs"] -= 60
            row["stopTs"] += 60
    returnInfo = [e for e in __changingInfo if e["stopTs"] > startTs and e["startTs"] < stopTs]
    return returnInfo


__deviceMapping = None
def getDeviceMapping() -> dict:
    """Return mapping from recording meter to connected appliances."""
    __checkBase()
    global __deviceMapping
    if __deviceMapping is None:
        __deviceMapping = __openJson(os.path.join(FIRED_BASE_FOLDER, INFO_FOLDER_NAME, DEVICE_MAPPING_FILENAME))

    # devInfo = getDeviceInfo()
    # meters = set([devInfo[d]["submeter"] for d in devInfo])
    # devMapping = {}
    # for meter in meters:
    #     phase = next(devInfo[a]["phase"] for a in devInfo)
    #     appliances = [a for a in devInfo if devInfo[a]["submeter"] == meter]
    #     if len(appliances) > 0:
    #         if devInfo[appliances[0]]["timedMeter"]: appliances = ["changing"]
    #     devMapping[meter] = {"phase":phase,"appliances":appliances}
    return __deviceMapping


__deviceInfo = None
def getDeviceInfo() -> dict:
    """Return info of all appliances."""
    __checkBase()
    global __deviceInfo
    if __deviceInfo is None:
        __deviceInfo = __openJson(os.path.join(FIRED_BASE_FOLDER, INFO_FOLDER_NAME, DEVICE_INFO_FILENAME))
    return __deviceInfo


__lightsPower = None
def getLightsPowerInfo() -> dict:
    """Return power info of all lights."""
    __checkBase()
    global __lightsPower
    if __lightsPower is None:
        __lightsPower = __openJson(os.path.join(FIRED_BASE_FOLDER, INFO_FOLDER_NAME, LIGHTS_POWER_INFO_FILENAME))
    return __lightsPower

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

def getHighFreqPath() -> str:
    """Return folder where annotation data is stored."""
    __checkBase()
    return os.path.join(FIRED_BASE_FOLDER, RAW_FOLDER_NAME)


def getRecordingRange(startStr: Optional[str]=None, endStr: Optional[str]=None) -> Tuple[float, float]:
    r"""
    Return start and stop timestamp of recording.
    If start and/or end is given, max(recordingStart, start) and min(recordingStop, end) is given.

    :param startStr: start timestamp in string representation that is checked for validity or None.
                     Format is: \"%Y.%m.%d\" or \"%Y.%m.%d %H:%M:%S\".
    :type  startStr: str or None
    :param stopStr:  start timestamp in string representation  that is checked for validity or None.
                     Format is: \"%Y.%m.%d\" or \"%Y.%m.%d %H:%M:%S\".
    :type  startStr: str or None

    :return: start and end timestamp
    :rtype: Tuple(float, float)
    """
    summaryPath = get50HzSummaryPath()
    firstFolder = min(os.path.join(summaryPath, p) for p in os.listdir(summaryPath)
                      if os.path.isdir(os.path.join(summaryPath, p)))
    allFiles = sorted([os.path.join(firstFolder, p) for p in os.listdir(firstFolder)
                       if os.path.isfile(os.path.join(firstFolder, p)) and "mkv" in p.split(".")])

    if len(allFiles) < 1: return [None, None]
    start = filenameToTimestamp(allFiles[0])
    durLast = info(allFiles[-1])["streams"][0]["duration"]
    end = filenameToTimestamp(allFiles[-1]) + durLast

    import pytz, time
    tzOFRec = pytz.timezone('Europe/Berlin')

    def localTz():
        if time.daylight:
            offsetHour = time.altzone / 3600
        else:
            offsetHour = time.timezone / 3600
        return pytz.timezone('Etc/GMT%+d' % offsetHour)

    thisTz = localTz()

    start = datetime.fromtimestamp(start).astimezone(tzOFRec).astimezone(thisTz).timestamp()
    end = datetime.fromtimestamp(end).astimezone(tzOFRec).astimezone(thisTz).timestamp()

    endDate = datetime.fromtimestamp(end)
    nextDay = endDate.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    if (nextDay.timestamp() - endDate.timestamp()) < 2: end = nextDay.timestamp()

    if startStr is not None:
        if len(startStr.split(" ")) > 1:
            startTs = datetime.strptime(startStr, "%Y.%m.%d %H:%M:%S").timestamp()
        else:
            startTs = datetime.strptime(startStr, "%Y.%m.%d").timestamp()
        start = max(startTs, start)
    if endStr is not None:
        if len(endStr.split(" ")) > 1:
            stopTs = datetime.strptime(endStr, "%Y.%m.%d %H:%M:%S").timestamp()
        else:
            stopTs = datetime.strptime(endStr, "%Y.%m.%d").timestamp()
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


def _getFlip(meterOrAppliance: str) -> bool:
    deviceMapping = getDeviceMapping()
    if meterOrAppliance in deviceMapping:
        # Devices not known, flip by default, as we measure N
        if deviceMapping[meterOrAppliance]["flip"] == "unknown": return True
        return deviceMapping[meterOrAppliance]["flip"]
    else:
        return False

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

def typeFromApplianceName(name):
    types = {"laptop":"laptop",
             " pc":"pc", "light":"light", "grinder":"grinder",
             "charger":"charger",
             "router":"router",
             "access point":"router",
             "display":"monitor",}
    for t in types:
        if t in name: return types[t]
    return name

def getApplianceList(meter: Optional[str]=None, startTs: Optional[float]=None, stopTs: Optional[float]=None) -> list:
    """
    Return list of appliance names active in between the given time range.
    Active is defined as metered or connected to changing device or light turned on.
    NOTE: What about stove?

    :param meter: one meter
    :type  meter: None or str
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

    if meter is None:
        for key in deviceMapping: devices.extend(deviceMapping[key]["appliances"])
        devices = [d for d in devices if d not in ["changing", "L1,L2,L3"]] + ["stove"]
        cdInfo = getChangingDeviceInfo()
        changingDevices = list(set([cdI["name"] for cdI in cdInfo if cdI["startTs"] < stopTs and cdI["stopTs"] > startTs]))
        lightsInfo = loadAnnotations(LIGHT_ANNOTATION, loadData=False)
        lights = [l["name"] for l in lightsInfo]
        appliances = sorted(list(set(devices + changingDevices + lights)))
    else:
        if meter in deviceMapping:
           devices.extend(deviceMapping[meter]["appliances"])
           if "changing" in devices:
                cdInfo = getChangingDeviceInfo()
                devices = list(set([cdI["name"] for cdI in cdInfo if cdI["startTs"] < stopTs and cdI["stopTs"] > startTs]))
        appliances = sorted(list(set(devices)))
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

def UTCfromLocalTs(ts):
    tz = getTimeZone()
    date = datetime.fromtimestamp(ts)
    date = tz.localize(date)
    return date.timestamp()

def getTimeZone():
    return pytz.timezone("Europe/Berlin")


def getBasePower(samplingrate: int, startTs: Optional[float] = None, stopTs: Optional[float] = None,
                 phase: Union[None, int, List[int]] = None) -> List[dict]:
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
    if startTs is None:
        startTs = getRecordingRange()[0]
    if stopTs is None:
        stopTs = getRecordingRange()[1]
    if phase is None:
        phase = [1, 2, 3]
    if not isinstance(phase, list):
        phase = [phase]

    # get ranges from where to compute base powers
    ranges = bestBasePowerTimeRange(startTs, stopTs)
    smartmeterName = getSmartMeter()
    meters = getMeterList()
    deviceMapping = getDeviceMapping()

    # Construct yet empty base power list
    basePowers = {}
    newSize = int((stopTs - startTs) * samplingrate)
    rangeSize = sum(int((r["stopTs"] - r["startTs"]) * samplingrate) for r in ranges)
    powers = ["p", "q", "s"]
    dt = [(m, '<f4') for m in powers]

    for p in phase:
        data = np.recarray((rangeSize,), dtype=dt).view(np.recarray)

        # Loop over all ranges
        index = 0
        for r in ranges:
            # get smartmeter data
            smData = getMeterPower(smartmeterName, samplingrate, startTs=r["startTs"], stopTs=r["stopTs"], phase=p)["data"]

            # add it for each phase
            for m in powers:
                data[m] = 0
                dataDict = {"title": "basepower", "name": "basepower l" + str(p), "phase": p, "data": data,
                            "timestamp": startTs, "type": "audio", "samplingrate": samplingrate, "measures": powers}
                basePowers[p] = dataDict
                basePowers[p]["data"][m][index:index + len(smData)] = smData[m]

            # load meter data
            for meter in meters:
                meter_phase = deviceMapping[meter]["phase"]
                if meter_phase not in p:
                    continue
                mData = getMeterPower(meter, samplingrate, startTs=r["startTs"], stopTs=r["stopTs"])["data"]

                # Subtract it from each phase
                for m in powers:
                    basePowers[p]["data"][m][index:index + len(mData)] -= mData[m]
            index += len(smData)

        # Prevent that base power can be negative
        for m in powers:
            indices = np.where(basePowers[p]["data"][m] < 0)
            basePowers[p]["data"][m][indices] = 0

    # Calculate base power
    for p in phase:
        data = np.recarray((newSize,), dtype=dt).view(np.recarray)
        for m in powers:
            hist, bin_edges = np.histogram(basePowers[p]["data"][m])
            # Sort histogram based on bins with most entries
            idx = list(reversed(np.argsort(hist)))[:2]
            # Mean of 2 bins with most entries in histogram
            mean = np.sum([bin_edges[i] * hist[i] for i in idx]) / np.sum([hist[i] for i in idx])
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
    if startTs is None:
        startTs = getRecordingRange()[0]
    if stopTs is None:
        stopTs = getRecordingRange()[1]
    if phase is None:
        phase = [1, 2, 3]
    if not isinstance(phase, list):
        phase = [phase]
    # Init return dict
    data = {}
    for p in phase:
        data[p] = {"phase": p, "samplingrate": samplingrate, "title": "stove", "name": "stove l" + str(p), "data": None}

    # Get smartmeter name
    smartmeterName = getSmartMeter()

    # Calculate base power consumption
    base = getBasePower(samplingrate, startTs=startTs, stopTs=stopTs, phase=phase)

    # Get total power consumption
    smartmeterData = {}
    for p in phase:
        smartmeterData[p] = getMeterPower(smartmeterName, samplingrate, startTs=startTs, stopTs=stopTs, phase=p)

    # Get individual meter data
    deviceMapping = getDeviceMapping()

    # Load power for all meters within that phase
    powerMeters = [m for m in getMeterList() if deviceMapping[m]["phase"] in phase]
    allMeterPower = [getMeterPower(name, samplingrate, startTs=startTs, stopTs=stopTs) for name in powerMeters]

    for meter in allMeterPower:
        meterName = meter["title"]
        p = deviceMapping[meterName]["phase"]
        if data[p]["data"] is None:
            data[p]["data"] = meter["data"]
            data[p]["measures"] = meter["measures"]
            data[p]["timestamp"] = meter["timestamp"]
        else:
            for m in data[p]["measures"]:
                data[p]["data"][m] += meter["data"][m]

    # Lights are neglected, as stove consumes way more power
    for p in data:
        b = next(b for b in base if b["phase"] == p)
        for m in data[p]["measures"]:
            data[p]["data"][m] = smartmeterData[p]["data"][m] - data[p]["data"][m] - b["data"][m]
            data[p]["data"][m][data[p]["data"]["s"] < 800] = 0
            data[p]["data"][m][data[p]["data"]["p"] < 800] = 0
        pass
        # peaks, props = find_peaks(data[p]["data"]["s"], threshold=800, width=1)
        # Filter peaks which are smaller than 2s as this cannot be the stove
        peaks, props = find_peaks(data[p]["data"]["s"], threshold=800, width=(1, int(1.0*samplingrate)))
        # There may be remaining peaks from slightly misaligned data at higher samplingrates, we want to remove them
        for m in data[p]["measures"]:
            # Force them to be zero
            data[p]["data"][m][peaks] = 0

        # median filter data
        N = max(1, int(5.0 * samplingrate))
        if (N % 2) == 0:
            N += 1  # filter has to be odd
        if N > 1:
            for m in data[p]["measures"]:
                # Force them to be zero
                data[p]["data"][m] = medfilt(data[p]["data"][m], N)

    return [data[p] for p in data]


def getReconstructibleDevices() -> dict:
    """Return dict for reconstructible devices with handler function"""
    rec = {"stove":getPowerStove}
    return rec

def delDownloads():
    """Delete the files loaded via rsync."""
    global _loadedFiles
    for f in _loadedFiles:
        try: subprocess.check_output(["rm", f])
        # Errors like not exist
        except subprocess.CalledProcessError: pass
    _loadedFiles = []

def getMeterChunk(meter: str, timeslice: float, data: str="VI", samplingrate: Optional[float]=None, startTs: Optional[float]=None, stopTs: Optional[float]=None, channel: Optional[int]=0) -> dict:
    """
    Return data of given meter in a as chunks of given size.

    :param meter: Name of meter; must be in getMeterList().
    :type  meter: str
    :param timeslice: time slice over which to iterate
    :type  timeslice: float
    :param samplingrate: samplingrate of returned power, None for default
    :type  samplingrate: int
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None

    :return: power data
    :rtype: dict
    """
    __checkBase()
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]

    # This will download it over rsync if this is required
    files = getMeterFiles(meter, samplingrate, data=data, startTs=startTs, stopTs=stopTs)

    if VERBOSE: print(files)
    current = startTs
    if len(files) == 0: return

    deviceMapping = getDeviceMapping()
    finish = False
    timestamp = startTs
    chunkI = 0
    fileI = 0
    fileIndex = 0
    chunk = None
    eof = None
    sendSamples = 0
    samplesToSend = 1
    while fileIndex < len(files) and sendSamples < samplesToSend:
        # Not inited at all
        if chunk is None:
            try:
                inf = info(files[fileIndex])["streams"][0]
            except Exception as e:
                print(e)
                print(fileIndex)
                print(files[fileIndex])
                print(info(files[fileIndex]))
                return None
            start = timestamp-inf["timestamp"]
            end = inf["timestamp"]+inf["duration"]
            dur = -1
            if stopTs < end: dur = stopTs-(inf["timestamp"]+start)
            audio = loadAudio(files[fileIndex], streamsToLoad=[channel], start=start, duration=dur)
            if audio is None or len(audio) == 0: return
            data = audio[0]
            data["phase"] = deviceMapping[meter]["phase"]
            if VERBOSE and samplingrate is not None and samplingrate != data["samplingrate"]:
                print("Have to resmaple data, this ist typically slow")
            chunkSize = int(timeslice*data["samplingrate"])
            samplesToSend = (stopTs-startTs)*data["samplingrate"]
            chunk = np.recarray((chunkSize,), dtype=data["data"].dtype).view(np.recarray)
            eof = data["timestamp"] + data["duration"]
            fileI = int((timestamp - data["timestamp"])*data["samplingrate"])
            if VERBOSE:
                print(time_format_ymdhms(data["timestamp"]) + "->" + time_format_ymdhms(eof))

        while fileI < len(data["data"]):
            # Fill chunk with available data
            fileJ = min(len(data["data"]), fileI + (chunkSize - chunkI))
            samples = fileJ - fileI
            # print("{}->{}: {}".format(fileI, fileJ, len(data["data"])))
            # print("{}->{}: {}".format(chunkI, chunkI+samples, chunkSize))
            # print("_"*50)
            chunk[chunkI:chunkI+samples] = data["data"][fileI:fileJ]
            chunkI += samples
            fileI += samples
            # Total chunk is written
            if chunkI >= chunkSize:
                # Copy over dict entries
                dataReturn = {k:i for k,i in data.items() if k != "data"}
                dataReturn["data"] = chunk

                if samplingrate is not None and samplingrate != data["samplingrate"]:
                    dataReturn["data"] = resampleRecord(dataReturn["data"], data["samplingrate"], samplingrate)
                    dataReturn["samplingrate"] = samplingrate

                dataReturn["timestamp"] = timestamp
                distanceToStop = stopTs - (dataReturn["timestamp"] + len(dataReturn["data"])/data["samplingrate"])

                if distanceToStop <= 0:
                    sampleEnd = int((stopTs - dataReturn["timestamp"])*data["samplingrate"])
                    dataReturn["data"] = dataReturn["data"][:sampleEnd]
                    finish = True
                dataReturn["samples"] = len(dataReturn["data"])
                dataReturn["duration"] = dataReturn["samples"]/data["samplingrate"]
                sendSamples = sendSamples+len(dataReturn["data"])
                yield dataReturn
                chunk = np.recarray((chunkSize,), dtype=data["data"].dtype).view(np.recarray)
                timestamp += chunkSize/data["samplingrate"]
                chunkI = 0
            if finish: return

        missingSamples = int((stopTs - eof)*data["samplingrate"])

        if VERBOSE:
            print("Missing:{}".format(missingSamples))
            print("Samples Send: {}/{}".format(sendSamples, samplesToSend))
        # We reached end of file
        # Load next and check distance
        fileIndex += 1
        if fileIndex < len(files):
            dur = stopTs-eof
            # Strange things if we use exact dur, so use an extra second, it is cut later
            # NOTE: Maybe we are missing samples sometimes due to milliseconds resolution of pyav?
            audio = loadAudio(files[fileIndex], streamsToLoad=[channel], duration=dur+1)
            if audio is None or len(audio) == 0: return
            data = audio[0]
            fileI = 0
            missingSamples = int((data["timestamp"] - eof)*data["samplingrate"])
            if VERBOSE and missingSamples > 0:
                print("Gap between: {}->{}, {}samples".format(time_format_ymdhms(eof), time_format_ymdhms(data["timestamp"]), missingSamples))
            eof = data["timestamp"] + data["duration"]

        # If file has missing samples or missing samples when no file is left
        if missingSamples > 0:
            while missingSamples > 0:
                samples = min(missingSamples, (chunkSize - chunkI))
                # Fill with zeros
                for m in chunk.dtype.names: chunk[m][chunkI:chunkI+samples] = 0
                chunkI += samples
                missingSamples -= samples
                # Total chunk is written
                if chunkI >= chunkSize:
                    # Copy over dict entries
                    dataReturn = {k:i for k,i in data.items() if k != "data"}
                    dataReturn["data"] = chunk

                    if samplingrate is not None and samplingrate != data["samplingrate"]:
                        dataReturn["data"] = resampleRecord(dataReturn["data"], data["samplingrate"], samplingrate)
                        dataReturn["samplingrate"] = samplingrate

                    dataReturn["timestamp"] = timestamp
                    distanceToStop = stopTs - (dataReturn["timestamp"] + len(dataReturn["data"])/data["samplingrate"])
                    if distanceToStop <= 0:
                        sampleEnd = int((stopTs - dataReturn["timestamp"])*data["samplingrate"])
                        dataReturn["data"] = dataReturn["data"][:sampleEnd]
                        finish = True
                    dataReturn["samples"] = len(dataReturn["data"])
                    dataReturn["duration"] = dataReturn["samples"]/data["samplingrate"]
                    sendSamples += int(dataReturn["samples"])
                    yield dataReturn
                    chunk = np.recarray((chunkSize,), dtype=data["data"].dtype).view(np.recarray)
                    timestamp += chunkSize/data["samplingrate"]
                    chunkI = 0
        if finish: return


def getMeterVI(meter: str, samplingrate: Optional[float]=None, startTs: Optional[float]=None, stopTs: Optional[float]=None, phase: Optional[int]=None, smartmeterMergePhases: Optional[bool] = False) -> dict:
    """
    Return vi of given meter.

    :param meter: Name of meter; must be in getMeterList().
    :type  meter: str
    :param samplingrate: samplingrate of returned power, None for default
    :type  samplingrate: int
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None

    :return: power data
    :rtype: dict
    """
    __checkBase()
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]
     # Standard is first stream in the data
    stream = [0]
    # For smartmeter we have 3 streams
    if meter == getSmartMeter():
        stream = [0,1,2]
        # if specific phase is requested
        if phase is not None:
            stream = [phase - 1]
    # Can only request specific stream for smartmeters
    else:
        assert phase == None, 'Can only return specific phase for recording of smartmeter'
    
    allData = []
    for s in stream:
        # Use generator object
        data = [c for c in getMeterChunk(meter, (stopTs - startTs), data="VI", startTs=startTs, stopTs=stopTs, samplingrate=samplingrate, channel=s)]
        # On flip, flip all measurements
        if _getFlip(meter):
            for i in range(len(data)):
                for m in data[i]["measures"]: data[i]["data"][m] *= -1
        allData.append(data[0])

    if meter == getSmartMeter() and smartmeterMergePhases:
        allData = [mergeSmartmeterChannels(allData)]

    if len(allData) == 1: return allData[0]
    elif len(allData) > 0: return allData
    else: return None


def getMeterFiles(meter: str, samplingrate: float, data: Optional[str]="PQ", startTs: Optional[float]=None, stopTs: Optional[float]=None) -> list:
    """
    Return files required for given meter and timestamp.

    Will use rsync if active to get missing files from remote location.
    Files are downloaded into the corresponding BASE_FOLDER.
    You can delete them (only the ones loaded with this function) using delDownloads().

    :param meter: Name of meter; must be in getMeterList().
    :type  meter: str
    :param samplingrate: samplingrate of returned power
    :type  samplingrate: int
    :param data: Type of the data, either PQ or VI
    :type  data: str
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None

    :return: power data
    :rtype: dict
    """
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]

    if data == "VI": directory = os.path.join(getHighFreqPath(), meter)
    elif data == "PQ" and samplingrate is not None and samplingrate > 1.0: directory = os.path.join(get50HzSummaryPath(), meter)
    else: directory = os.path.join(get1HzSummaryPath(), meter)

    files = []
    if data == "VI":
        ts = int(int(startTs)/(10*60))*10*60
        # NOTE: This will not work for files other than full 10 min files
        while ts < stopTs:
            f = os.path.join(directory, meter + "_" + datetime.fromtimestamp(ts).strftime(STORE_TIME_FORMAT) + ".mkv")
            files.append(f)
            # get next full 10 minutes
            ts = int((int(ts)+10*60)/(10*60))*10*60
    else:
        startDate = datetime.fromtimestamp(startTs).replace(hour=0, minute=0, second=0, microsecond=0)
        stopDate = datetime.fromtimestamp(stopTs)
        # Special case of data until midnight (next day wont be loaded)
        if stopDate == stopDate.replace(hour=0, minute=0, second=0, microsecond=0): stopDate = stopDate - timedelta(days=1)
        else: stopDate = stopDate.replace(hour=0, minute=0, second=0, microsecond=0)

        dates = [startDate]
        while startDate < stopDate:
            startDate += timedelta(days=1)
            dates.append(startDate)
        files = [os.path.join(directory, meter + "_" + date.strftime(STORE_TIME_FORMAT) + ".mkv") for date in dates]
    for file in files:
        if VERBOSE: print(file)
        if not os.path.exists(file) and not RSYNC_ALLOWED: sys.exit("\033[91mFile {} does not exist\033[0m".format(file))
        if not os.path.exists(file) and RSYNC_ALLOWED:
            if VERBOSE:
                print("File: {} does not exist, using rsync to download".format(file))
            dest = os.path.dirname(file)
            os.makedirs(dest, exist_ok=True)
            rsyncSubPath = file.split(FIRED_BASE_FOLDER)[-1].replace(os.path.sep, "/")
            cmd = "rsync --password-file={} {}{} {}".format(getRSYNCPwdFile(), RSYNC_ADDR, rsyncSubPath, dest)
            if VERBOSE: print(cmd)
            OUT = subprocess.PIPE
            process = subprocess.Popen(cmd, stdout=OUT, stderr=OUT, shell=True)
            process.wait()
            # In case of failure, notify
            if not os.path.exists(file):
                print("error retreiving file:\n{}".format(OUT))
            if os.path.exists(file): _loadedFiles.append(file)
    return files


def getMeterFiles2(meter: str, samplingrate: float, data: Optional[str]="PQ", startTs: Optional[float]=None, stopTs: Optional[float]=None) -> list:
    """
    Return files required for given meter and timestamp.

    :param meter: Name of meter; must be in getMeterList().
    :type  meter: str
    :param samplingrate: samplingrate of returned power
    :type  samplingrate: int
    :param data: Type of the data, either PQ or VI
    :type  data: str
    :param startTs: Time range start
    :type  startTs: float or None
    :param stopTs: Time range stop
    :type  stopTs: float or None

    :return: power data
    :rtype: dict
    """
    if startTs is None: startTs = getRecordingRange()[0]
    if stopTs is None: stopTs = getRecordingRange()[1]

    # Get correct data path depending on data and sr
    if data =="VI": directory = os.path.join(getHighFreqPath(), meter)
    if data =="PQ" and samplingrate is not None and samplingrate > 1.0: directory = os.path.join(get50HzSummaryPath(), meter)
    else: directory = os.path.join(get1HzSummaryPath(), meter)

    files = []
    ct = None
    for file in sorted(os.listdir(directory)):
        # not a matroska file
        if ".mkv" not in file: continue
        ts = filenameToTimestamp(file)
        # could not extract timestamp
        if ts is None: continue
        # This is much much quicker than opening all files
        if data =="VI":
            # get full 10 minutes
            ts_end = int((int(ts)+10*60)/(10*60))*10*60
        else: ts_end = ts+24*60*60
        if ts_end <= startTs: continue
        if ts >= stopTs: break # only works if files are sorted
        files.append(os.path.join(directory, file))
    return files


def getMeterPower(meter: str, samplingrate: float, startTs: Optional[float] = None, stopTs: Optional[float] = None,
                  phase: Optional[int] = None, smartmeterMergePhases: Optional[bool] = False) -> dict:
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
    :param phase: phase number
    :type phase: int or None

    :return: power data
    :rtype: dict
    """
    __checkBase()
    if startTs is None:
        startTs = getRecordingRange()[0]
    if stopTs is None:
        stopTs = getRecordingRange()[1]
    files = getMeterFiles(meter, samplingrate, startTs=startTs, stopTs=stopTs)

    # Standard is first stream in the data
    stream = [0]
    # For smartmeter we have 3 streams
    if meter == getSmartMeter():
        stream = [0,1,2]
        # if specific phase is requested
        if phase is not None:
            stream = [phase - 1]
    # Can only request specific stream for smartmeters
    else:
        assert phase == None, 'Can only return specific phase for recording of smartmeter'
    
    # Return data is either dict for individual meters or list of 3 dicts for smartmeter 
    returnData = []
    for s in stream:
        if VERBOSE:
            print("{}: Loading MKV stream {}...".format(meter, s), end="", flush=True)
        data = [loadAudio(file, streamsToLoad=[s])[0] for file in files]
        if VERBOSE:
            print("Done")
        if len(data) < 1:
            return None
        dataNice = data[0]
        # Concat data of several files
        if len(data) > 1:
            for d in data[1:]:
                dataNice["data"] = np.concatenate((dataNice["data"], d["data"]))

        data = dataNice

        fromSample = int((startTs - data["timestamp"]) * data["samplingrate"])
        toSample = int((stopTs - data["timestamp"]) * data["samplingrate"])

        data["data"] = data["data"][fromSample:toSample]
        data["timestamp"] = startTs

        if VERBOSE:
            print("{}->{}: len({})".format(time_format_ymdhms(startTs), time_format_ymdhms(stopTs), len(data["data"])))

        if samplingrate != 1 and samplingrate != 50:
            if VERBOSE:
                print("resampling")
            data["data"] = resampleRecord(data["data"], data["samplingrate"], samplingrate)
            data["samplingrate"] = samplingrate

        goalSamples = int(data["samplingrate"] * (stopTs - startTs))

        if abs(goalSamples - len(data["data"])) > data["samplingrate"]:
            print(f"\033[91mError loading data for {meter}. Requested samples: {goalSamples}, "
                f"actual samples: {len(data['data'])}\033[0m")

        if goalSamples > len(data["data"]):
            new = np.recarray((goalSamples - len(data["data"]),), dtype=data["data"].dtype).view(np.recarray)
            if abs(goalSamples - len(data["data"])) > data["samplingrate"]:
                new[:] = 0
            else:
                new[:] = data["data"][-1]
            data["data"] = np.concatenate((data["data"], new))
        elif goalSamples < len(data["data"]):
            data["data"] = data["data"][:goalSamples]

        deviceMapping = getDeviceMapping()
        data["phase"] = deviceMapping[meter]["phase"]
        data["samples"] = len(data["data"])
        data["duration"] = data["samples"] / data["samplingrate"]

        # prevent memory leak by copying over and delete larger one
        new = np.recarray((data["samples"],), dtype=data["data"].dtype).view(np.recarray)
        new[:] = data["data"][:]
        del data["data"]
        data["data"] = new
        returnData.append(data)
    if meter == getSmartMeter() and smartmeterMergePhases:
        returnData = [mergeSmartmeterChannels(returnData)]
        
    if len(returnData) == 1: return returnData[0]
    elif len(returnData) > 0: return returnData
    else: return None

def mergeSmartmeterChannels(dataList):
    """
    Return smartmeter data merged into one dictionary.

    :param dataList: List of dictionaries
    :type  dataList: str

    :return: dict of power data or list of dict for devices connected to multiple phases
    :rtype: dict or list
    """
    assert len(dataList) == 3, "This is no valid smartmeter data"
    oldMeasures = dataList[0]["data"].dtype.names
    newMeasures = [m + "_l" + str(i) for i in [1,2,3] for m in oldMeasures]
    print(newMeasures)
    dt = [(m, '<f4') for m in newMeasures]
    newData = np.recarray((len(dataList[0]["data"]),), dtype=dt).view(np.recarray)
    for i in [1,2,3]:
        for m in oldMeasures: newData[m + "_l" + str(i)] = dataList[i-1]["data"][m]
    dataDic = dataList[0]
    dataDic["data"] = newData
    dataDic["title"] = dataDic["title"].split(" ")[0]
    dataDic["measures"] = newMeasures
    dataDic["channels"] = len(newMeasures)
    return dataDic

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
    iinfo = nfo["streams"][0]
    if "TIMESTAMP" in iinfo["metadata"]: startTs = float(iinfo["metadata"]["TIMESTAMP"])
    else: startTs = filenameToTimestamp(file)
    return (startTs, iinfo["duration"], iinfo["samplingrate"], iinfo["measures"])



def streamsForTitle(streams, titles:Union[str, List[str]]):
    """Return only streams that match the given title"""
    # Handle title not list case
    if not isinstance(titles, list): titles = [titles]
    # Loop over stream metadata and look for title match
    newStreams = []
    for stream in streams:
        key =  set(["Title", "title", "TITLE", "NAME", "Name", "name"]).intersection(set(stream.metadata.keys()))
        if len(key) > 0:
            title = stream.metadata[next(iter(key))]
            if title in titles: newStreams.append(stream)
    return newStreams


def chunkLoads(fileList: List[str], timeslice:float, start: Optional[float]=0, stop: Optional[float]=-1,
               streamsToLoad: Optional[List[int]]=None, titles: Optional[List[str]]=None) -> List[dict]:
    r"""
    Load multiple files in chunks to keep memory usage small.
    The purpose is to have a dataset split into multiple files.
    fileList must be sorted!!!

    Yields a list of dictionaries.
    Each dictionary holds information such as metadata, title, name and data.
    The list has the following structure:

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
            },
            ...
        ]

    :param fileList:       List of filepath to the mkvs to open. Must be sorted and continues
    :type  fileList:       list of str
    :param timeslice:      timeslice that is returned
    :type  timeslice:      float
    :param start:          start time in the file (seconds from first file start)
                           default: 0
    :type  start:          float
    :param stop:           stop time in the file (seconds from file start), stop muts be larger than start,
                           default: -1 (no stop)
    :type  stop:           float
    :param streamsToLoad:  List of streams to load from the file, either list of numbers or list of stream titles
                           default: None -> all streams should be loaded
    :type  streamsToLoad:  list
    :param titles:         List or str of streams titles which should be loaded
                           default: None -> no filter based on title names
    :type  titles:         list or str

    :yield: List of dictionaries holding data and info
    :rtype: list
    """
    assert stop == -1 or stop > start, "Stop seconds must be larger than start seconds"
    finished = False

    # TODO: This does not work if timeslice is larger than filesize... make it work
    # NOTE: Should be fixed with latest commit, need to further test it

    started = start
    duration = stop-start
    currentDuration = 0
    missingSeconds = 0
    import time
    start = time.time()

    j = 0
    while not finished:
        if j >= len(fileList): return
        file = fileList[j]
        # Skip files we do not need
        inf = info(file)["streams"][0]
        if inf["duration"] < started:
            started -= inf["duration"]
            continue
        for dataListChunk in chunkLoad(file, timeslice, start=started, stop=stop, streamsToLoad=streamsToLoad, titles=titles):
            start = time.time()
            # We started from starttime frist, now we have to set
            started = missingSeconds
            # Check if chunk length matches
            addNext = any([int(timeslice*s["samplingrate"]) > s["samples"] for s in dataListChunk])
            # If not, load chunk from next file if there is one left
            while addNext and file != fileList[-1]:
                file = fileList[j+1]
                # Calculate missing seconds
                missingSeconds = timeslice - (dataListChunk[0]["samples"]/dataListChunk[0]["samplingrate"])
                started = missingSeconds
                # load one chunk from nextfile
                addedChunk = next(chunkLoad(fileList[j+1], missingSeconds, start=0, stop=stop, streamsToLoad=streamsToLoad, titles=titles))
                # Copy data over and set samples and duration
                for i in range(len(dataListChunk)):
                    dataListChunk[i]["data"] = np.concatenate((dataListChunk[i]["data"], addedChunk[i]["data"]))
                    dataListChunk[i]["samples"] += addedChunk[i]["samples"]
                    dataListChunk[i]["duration"] += addedChunk[i]["duration"]
                # Look if we still need to add the next slice
                addNext = any([int(timeslice*s["samplingrate"]) > s["samples"] for s in dataListChunk])
                # if so increase j
                if addNext: j += 1
            # If we have a stop time, clean data at the end, if chunk was too much
            if stop != -1:
                dis = (currentDuration + dataListChunk[0]["duration"]) - duration
                # Look if stop has been reached
                if dis >= 0:
                    if VERBOSE and dis != 0.0:
                        print("Removing: {}s or {} samples".format(dis, len(dataListChunk[0]["data"])))
                    # Indicate finish
                    finished = True
                    # Clean data
                    for i in range(len(dataListChunk)):
                        endSample = len(dataListChunk[i]["data"])-int(dis*dataListChunk[i]["samplingrate"])
                        # dataListChunk[i]["data"] = dataListChunk[i]["data"][:endSample]
                        dataListChunk[i]["data"] = np.delete(dataListChunk[i]["data"], np.s_[endSample:], axis = 0)
                        dataListChunk[i]["samples"] = len(dataListChunk[i]["data"])
                        dataListChunk[i]["duration"] = dataListChunk[i]["samples"]/dataListChunk[i]["samplingrate"]

            currentDuration += dataListChunk[0]["duration"]
            yield dataListChunk
            if finished: return
        j += 1 # next file


def chunkLoad(filepath: str, timeslice: float, start: Optional[float] = 0, stop: Optional[float] = -1,
              streamsToLoad: Optional[List[int]] = None, titles: Optional[List[str]] = None) -> List[dict]:
    r"""
    Load a single file in chunks to keep memory usage small.

    Yields a list of dictionaries.
    Each dictionary holds information such as metadata, title, name and data.
    The list has the following structure:

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
            },
            ...
        ]

    :param filepath:       filepath to the mkv to open
    :type  filepath:       str
    :param timeslice:      timeslice that is returned
    :type  timeslice:      float
    :param start:          start time in the file (seconds from file start)
                           default: 0
    :type  start:          float
    :param stop:           stop time in the file (seconds from file start), stop muts be larger than start,
                           default: -1 (no stop)
    :type  stop:           float
    :param streamsToLoad:  List of streams to load from the file, either list of numbers or list of stream titles
                           default: None -> all streams should be loaded
    :type  streamsToLoad:  list
    :param titles:         List or str of streams titles which should be loaded
                           default: None -> no filter based on title names
    :type  titles:         list or str

    :yield: List of dictionaries holding data and info
    :rtype: list
    """
    assert stop == -1 or stop > start, "Stop seconds must be larger than start seconds"
    chunkSizes, dataOnly, loadedFrames, dataLen, dataDict = {}, {}, {}, {}, {}
    # Open the container
    try: container = av.open(filepath)
    except av.AVError: return []
    # NOTE: As for now, this only works for audio data
    # Get available stream indices from file
    streams = [s for i, s in enumerate(container.streams) if ( (s.type == 'audio') )]
    # Get available stream indices from file
    if isinstance(streamsToLoad, int): streamsToLoad = [streamsToLoad]
    if streamsToLoad is not None: streams = [stream for stream in streams if stream.index in streamsToLoad]
    if titles is not None: streams = streamsForTitle(streams, titles)
    if len(streams) == 0: return []
    # Copy over stream infos into datalist
    dataDict = {i["streamIndex"]:i for i in info(filepath)["streams"] if i["streamIndex"] in [s.index for s in streams]}

    # Prepare chunk size dict
    for i in dataDict.keys():
        chunkSizes[i] = int(dataDict[i]["samplingrate"]*timeslice)
        # Audio data will be stored here
        dataOnly[i] = np.empty((chunkSizes[i],len(dataDict[i]["measures"])), dtype=np.float32)
        loadedFrames[i] = []
        dataLen[i] = 0

    chunks = 0
    # Copy over timestamps so the timestamp of each chunk can be calculated
    timestamps = {i:dataDict[i]["timestamp"] for i in dataDict}

    start_pts = int(start*1000)
    # we can seek directly to the point of interest if we only have one stream
    if start != 0 and len(streams) == 1:
        # NOTE: Sometimes it seeks too far (maybe ), so make a safe margin
        container.seek(start_pts, whence='time', any_frame=False, stream=streams[0])

    inited = False
    # De-multiplex the individual packets of the file
    for packet in container.demux(streams):
        i = packet.stream.index

        # Inside the packets, decode the frames
        for frame in packet.decode():
            if not inited:
                inited = True
                if frame.pts > start_pts: raise AssertionError("Seeked too far, should not happen: {}ms - {}ms".format(frame.pts, start_pts))
            # Look if we can seek to next frame
            if frame.pts + int(frame.samples/float(frame.sample_rate)*1000) < start*1000: continue
            if stop != -1 and frame.pts > stop*1000: break
            # If we need to skip data at the beginning, 0 else
            s = max(0, int((start*1000-frame.pts)/1000.0*frame.sample_rate))
            # If we need to skip data at the end
            if stop != -1: e = min(frame.samples, int((stop*1000-frame.pts)/1000.0*frame.sample_rate))
            else: e = frame.samples

            # The frame can be audio, subtitles or video
            if packet.stream.type == 'audio':
                ndarray = frame.to_ndarray().T[s:e]
                loadedFrames[i].append(ndarray)
                dataLen[i] += ndarray.shape[0]

        # If the chunks have been loaded for all streams (sometimes chunksize smaller the framesize -> hence while)
        while all([dataLen[index] >= chunkSizes[index] for index in chunkSizes.keys()]):
            dataReturn = []
            for i in chunkSizes.keys():
                # Copy over metadata of stream
                dataReturn.append(dataDict[i])

                # Copy data of frames into chunks
                currentLen = 0
                while currentLen < chunkSizes[i]:
                    end = min(loadedFrames[i][0].shape[0], chunkSizes[i]-currentLen)
                    dataOnly[i][currentLen:currentLen+end] = loadedFrames[0][i][:end]
                    currentLen += end
                    loadedFrames[i][0] = loadedFrames[i][0][end:]
                    if loadedFrames[i][0].shape[0] == 0: del loadedFrames[i][0]

                dataReturn[-1]["data"] = np.core.records.fromarrays(dataOnly[i][:currentLen].T, dtype={'names': dataReturn[-1]["measures"], 'formats': ['f4']*len(dataReturn[-1]["measures"])})
                dataReturn[-1]["samples"] = len(dataReturn[-1]["data"])
                dataReturn[-1]["duration"] = dataReturn[-1]["samples"]/dataReturn[-1]["samplingrate"]
                dataReturn[-1]["timestamp"] = timestamps[i] + start + chunks*chunkSizes[i]/dataReturn[-1]["samplingrate"]
                dataLen[i] = max(0, dataLen[i]-chunkSizes[i])
            chunks += 1
            yield dataReturn

    # This is for the remaining chunk that is smaller than chunksize
    while not all([dataLen[index] == 0 for index in chunkSizes.keys()]):
        dataReturn = []
        for i in chunkSizes.keys():
            # Copy over metadata of stream
            dataReturn.append(dataDict[i])
            # Copy data of frames into chunks
            currentLen = 0
            while currentLen < min(chunkSizes[i], dataLen[i]):
                end = min(loadedFrames[i][0].shape[0], chunkSizes[i]-currentLen)
                dataOnly[i][currentLen:currentLen+end] = loadedFrames[0][i][:end]
                currentLen += end
                loadedFrames[i][0] = loadedFrames[i][0][end:]
                if loadedFrames[i][0].shape[0] == 0: del loadedFrames[i][0]
            dataReturn[-1]["samples"] = dataOnly[i][:currentLen].shape[0]
            dataReturn[-1]["duration"] = dataReturn[-1]["samples"]/dataReturn[-1]["samplingrate"]
            dataReturn[-1]["timestamp"] = timestamps[i] + start + chunks*chunkSizes[i]/dataReturn[-1]["samplingrate"]
            dataReturn[-1]["data"] = np.core.records.fromarrays(dataOnly[i][:currentLen].T, dtype={'names': dataReturn[-1]["measures"], 'formats': ['f4']*len(dataReturn[-1]["measures"])})
            dataLen[i] = max(0, dataLen[i]-chunkSizes[i])
        yield dataReturn


def loadAudio(filepath: str, streamsToLoad: Optional[List[int]]=None, titles: Optional[List[str]]=None, start: Optional[float]=0, duration: Optional[float]=-1) -> List[dict]:
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
            },
            ...
        ]

    :param filepath:       filepath to the mkv to open
    :type  filepath:       str
    :param start:          start time in the file (seconds from file start)
                           default: 0
    :type  start:          float
    :param duration:       Duration to load data (from start)
                           default: -1 (all data)
    :type  duration:       float
    :param streamsToLoad:  List of streams to load from the file, either list of numbers or list of stream titles
                           default: None -> all streams should be loaded
    :type  streamsToLoad:  list
    :param titles:         List or str of streams titles which should be loaded
                           default: None -> no filter based on title names
    :type  titles:         list or str

    :return: List of dictionaries holding data and info
    :rtype: list
    """
    # Open the container
    try: container = av.open(filepath)
    except av.AVError: return []
    # Get available stream indices from file
    streams = [s for i, s in enumerate(container.streams) if ( (s.type == 'audio') )]
    # Look if it is a stream to be loaded
    if isinstance(streamsToLoad, int): streamsToLoad = [streamsToLoad]
    if streamsToLoad is not None: streams = [stream for stream in streams if stream.index in streamsToLoad]
    if titles is not None: streams = streamsForTitle(streams, titles)
    if len(streams) == 0: return []
    # Copy over stream infos into datalist
    dataDict = {i["streamIndex"]:i for i in info(filepath)["streams"] if i["streamIndex"] in [s.index for s in streams]}
    # VERBOSE print
    if VERBOSE:
        for key, stream in dataDict.items():
            print("Stream : " + str(stream["streamIndex"]))
            for k in ["type", "title", "metadata", "samplingrate", "measures", "duration"]:
                if k in stream: print("{}: {}".format(k, stream[k]))
    indices = [stream.index for stream in streams]
    # Prepare dict for data
    for i in dataDict.keys():
        # Allocate empty array
        dur = dataDict[i]["samples"]/dataDict[i]["samplingrate"]-start
        if duration >= 0: dur = min(dur, duration)
        samples = max(0, int(math.ceil(dur*float(dataDict[i]["samplingrate"]))))
        dataDict[i]["samples"] = samples
        dataDict[i]["duration"] = dur
        dataDict[i]["timestamp"] = dataDict[i]["timestamp"]+start
        dataDict[i]["data"] = np.ones((dataDict[i]["samples"],len(dataDict[i]["measures"])), dtype=np.float32)
        # dataDict[i]["data"] = np.empty((dataDict[i]["samples"],len(dataDict[i]["measures"])), dtype=np.float32)
        dataDict[i]["storeIndex"] = 0

    start_pts = start*1000.0
    end_pts = start_pts + duration*1000
    inited = False

    for stream in streams:
        index = stream.index
        if VERBOSE: print(stream)
        container = av.open(filepath)
        container.seek(math.floor(start_pts), whence='time', any_frame=False, stream=stream)
        initCheck = False
        for frame in container.decode(streams=stream.index):
            # Check seek status
            if not inited:
                if frame.pts > start_pts: raise AssertionError("Seeked too far, should not happen: {}ms - {}ms".format(frame.pts, start_pts))
            # Check start
            if frame.pts + int(frame.samples/float(frame.sample_rate)*1000) < start_pts: continue
            # Check end
            if duration != -1 and frame.pts > end_pts: break

            # If we need to skip data at the beginning, 0 else
            # This is only ms resolution
            # NOTE: Does this cause problems for us?? we need to find out
            s = max(0, math.floor(float(start_pts-float(frame.pts))/1000.0*frame.sample_rate))
            # so use samplecount
            #s = max(0, int(frame.samples - (dataDict[index]["samples"] - dataDict[index]["storeIndex"])))
            # If we need to skip data at the end
            if duration >= 0: e = min(frame.samples, float((end_pts-frame.pts)/1000.0*frame.sample_rate))
            else: e = frame.samples
            # If there were rounding issues
            e = min(e, int(s+dataDict[index]["samples"] - dataDict[index]["storeIndex"]))

            # if not inited:
            #     print("dur: {}, endPts: {}".format(duration, end_pts, ))
            #     print("{}:{} - f:{}, sr: {}, pts:{}, start:{}, {}".format(s,e, frame.samples,frame.sample_rate, frame.pts, start_pts, (start_pts-frame.pts)/1000.0*frame.sample_rate))
            # Get corresponding index in dataList array
            ndarray = frame.to_ndarray().transpose()[int(s):int(e),:]
            # copy over data
            j = dataDict[index]["storeIndex"]
            dataDict[index]["data"][j:j+ndarray.shape[0],:] = ndarray[:,:]
            dataDict[index]["storeIndex"] += ndarray.shape[0]
            if not inited:
                inited = True

    # Demultiplex the individual packets of the file
    # for i, packet in enumerate(container.demux(streams)):

    #     # Inside the packets, decode the frames
    #     for frame in packet.decode(): # codec_ctx.decode(packet):

    #         # TODO:
    #         if not inited:
    #             inited = True
    #             if frame.pts > start_pts: raise AssertionError("Seeked too far, should not happen: {}ms - {}ms".format(frame.pts, start_pts))
    #             # Look if we can seek to next frame
    #         if frame.pts + int(frame.samples/float(frame.sample_rate)*1000) < start*1000: continue
    #         if stop != -1 and frame.pts > stop*1000: break
    #         # If we need to skip data at the beginning, 0 else
    #         s = max(0, int((start*1000-frame.pts)/1000.0*frame.sample_rate))
    #         # If we need to skip data at the end
    #         if stop != -1: e = min(frame.samples, int((stop*1000-frame.pts)/1000.0*frame.sample_rate))
    #         else: e = frame.samples

    #         # Get corresponding index in dataList array
    #         index = packet.stream.index
    #         ndarray = frame.to_ndarray().transpose()
    #         # copy over data
    #         j = dataDict[index]["storeIndex"]
    #         dataDict[index]["data"][j:j+ndarray.shape[0],:] = ndarray[:,:]
    #         dataDict[index]["storeIndex"] += ndarray.shape[0]
    # Make recarray from data
    for i in dataDict.keys():
        if "storeIndex" in dataDict[i]: del dataDict[i]["storeIndex"]
        dataDict[i]["data"] = np.core.records.fromarrays(dataDict[i]["data"].transpose(), dtype={'names': dataDict[i]["measures"], 'formats': ['f4']*len(dataDict[i]["measures"])})
    # RETURN it as a list
    return [dataDict[i] for i in dataDict.keys()]


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
    # This extracts container info
    info["format"] = container.format
    if container.duration is not None:
        info["duration"] = float(container.duration) / av.time_base
    else:
        info["duration"] = -1
    info["metadata"] = container.metadata
    info["#streams"] = len(container.streams)
    info["streams"] = []
    # Getting number of samples for each stream
    samples = getSamples(path)
    # Enumerate all streams and extract stream specific info
    for i, stream in enumerate(container.streams):
        streamInfo = {}
        # Samplingrate
        streamInfo["samplingrate"] = stream.sample_rate
        # Type (audio, video, subs)
        streamInfo["type"] = stream.type
        # index in of stream
        streamInfo["streamIndex"] = stream.index
        # extract # samples and duration
        if samples is not None:
            streamInfo["samples"] = samples[i]
            streamInfo["duration"] = samples[i]/streamInfo["samplingrate"]
        else:
            streamInfo["duration"] = 0
            streamInfo["samples"] = int(streamInfo["duration"]*streamInfo["samplingrate"])
        # Start time (0 for most cases)
        streamInfo["start_time"] = stream.start_time
        # Copy metadata dictionary
        streamInfo["metadata"] = stream.metadata
        # Extract stream title if there is any
        key = set(["Title", "title", "TITLE", "NAME", "Name", "name"]).intersection(set(stream.metadata.keys()))
        if len(key) > 0: title = stream.metadata[next(iter(key))]
        else: title = "Stream " + str(stream.index)
        streamInfo["title"] = title
        # Video and audio have a stream format
        if stream.type in ['audio', 'video']:
            streamInfo["format"] = stream.format
        # Audio has number of channels
        if stream.type == 'audio':
            streamInfo["#channels"] = stream.channels
        # Extract timestamp if there is any
        for key in ["TIMESTAMP", "Timestamp", "timestamp"]:
            if key in stream.metadata:
                streamInfo["timestamp"] = float(stream.metadata[key])
                break
        # Extract the channel tags / measures 
        channelTags = channelTags = ["C" + str(i) for i in range(stream.channels)]
        for key in ["CHANNEL_TAGS", "Channel_tags"]:
            if key in stream.metadata:
                channelTags = stream.metadata[key].split(",")
                break;
        streamInfo["measures"] = channelTags
        info["streams"].append(streamInfo)
    # Duration of container is longest duration of all streams

    info["duration"] = max([info["streams"][i]["duration"] for i in range(len(container.streams))])
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