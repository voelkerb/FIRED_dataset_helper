# %%
import helper as hp
# Set FIRED base Folder (location where you downloaded the dataset)
hp.FIRED_BASE_FOLDER = "~/FIRED"
# load 1Hz power data of the television for complete recording range
television = hp.getPower("television", 1)

# %%
# load 2 hours of 50Hz power data of powermeter09 (Fridge) of day 2020.08.03
startTs,  stopTs = hp.getRecordingRange("2020.08.03 17:25:00", "2020.08.03 19:25:00")
fridge = hp.getMeterPower("powermeter09", 50, startTs=startTs, stopTs=stopTs)

#Plotting the data is straightforward:
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

# Generate timestamps
start = fridge["timestamp"]
end = start+(len(fridge["data"])/fridge["samplingrate"])
timestamps = np.linspace(start, end, len(fridge["data"]))
dates = [datetime.fromtimestamp(ts) for ts in timestamps]
# Plot
fig, ax = plt.subplots()
ax.plot(dates, fridge["data"]["p"], label="active power")
ax.plot(dates, fridge["data"]["q"], label="reactive power")
# format plot  
ax.set_ylim(0,80)
ax.set(xlabel='Time of day', ylabel='Power [W/var]', title='Fridge')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax.get_xticklabels(), ha="center", rotation=35)
plt.legend()
plt.show()

# %%
hp.RSYNC_ALLOWED = True
# load two seconds of high freq data powermeter09 (Fridge)
startTs, stopTs = hp.getRecordingRange("2020.08.03 17:34:02", "2020.08.03 17:34:04")
fridge = hp.getMeterVI("powermeter09", startTs=startTs, stopTs=stopTs)

# Generate timestamps
start = fridge["timestamp"]
end = start+(len(fridge["data"])/fridge["samplingrate"])
timestamps = np.linspace(start, end, len(fridge["data"]))
dates = [datetime.fromtimestamp(ts) for ts in timestamps]
# Plot
fig, ax = plt.subplots()
ax.plot(dates, fridge["data"]["i"], label="current")
# format plot  
ax.set(xlabel='Time of day', ylabel='Current [mA]', title='Fridge')
plt.setp(ax.get_xticklabels(), ha="center", rotation=35)
plt.show()

# %%