### Loading a scientific data file using pandas ###

import pandas as pd
import numpy as np


# GMN data from August 2022
# https://globalmeteornetwork.org/data/plots/monthly/scecliptic_monthly_202208_density.png

# Data path online:
# https://globalmeteornetwork.org/data/traj_summary_data/monthly/traj_summary_monthly_202208.txt

data_file = "traj_summary_monthly_202208.txt"

# Load the data
# The delimiter is a semicolon
# After skipping the first row, the header is counted from zero
# The last row to skip is row 5 (the rows are counted with the header rows for some reason!)
data = pd.read_csv(data_file, sep=";", header=[0, 1], skiprows=[0, 5])

# Print the data
print(data)

# Notice how the header is not properly formatted, we need to fix that!

    
# Merge the two header rows and remove extra space and hashtag
updated_header = []
for head in data.columns:
    h1, h2 = head
    h1 = h1.replace("#", "").strip()
    h2 = h2.replace("#", "").strip()

    updated_header.append(h1 + " " + h2)
    
data.columns = updated_header

# Print the data (looks much better now!)
print(data)

# Print the columns so we know what we're doing and what to extract
print(data.columns)


# ## Do after an initial plot with all meteors was done ##

# # List all unique meteor shower codes
# print("Meteor shower codes:")
# print(data["IAU code"].unique())

# # Only select the Perseids (note the extra 2 spaces in the code)
# data = data[data["IAU code"] == "  PER"]

# ###



# Let's compute the Sun-centered ecliptic longitude and add it as a column
data["SCElon"] = (data["LAMgeo deg"] - data["Sol lon deg"])%360

# Plot the SCE longitude and latitude with the geocentric velocity color-coded
import matplotlib.pyplot as plt


# Wrap coordinates into the -180 to 180 deg range and apply a 270 deg rotation
sce_lon_plot = ((270 - data["SCElon"]) + 180)%360 - 180

# Plot an all-sky map (centre around 270 deg)
plt.subplot(projection='hammer')
plt.grid(True)

# The plot function takes the x and y coordinates in radians
plt.scatter(np.radians(sce_lon_plot), np.radians(data["BETgeo deg"]), c=data["Vgeo km/s"], s=1)
plt.xlabel("SCE longitude [deg]")
plt.ylabel("SCE latitude [deg]")

# Manually map X ticks to they properly reflect the applied coordinate rotation by 270 deg
plt.xticks(np.radians([-180, -90, 0, 90, 180]), ["90", "0", "270", "180", "90"])

plt.colorbar(label="Geocentric velocity [km/s]")
plt.show()


# And now let's only select the Perseids, see the commented code above