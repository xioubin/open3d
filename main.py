import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import cv2


# =============================================================
# Can't use this method since python is too slow
# =============================================================
""" 
interval = 5

# read raw data
raw_data = pd.read_csv('PSA-3.csv', header=None, engine='python', sep='\n')
raw_data = raw_data[0].str.split(',', expand=True)

raw_data = raw_data.drop([0])
column_names = []
for i in range(raw_data.shape[1]):
    column_names.append(i)
column_names[0:6] = ['date', 'time', 'x', 'y', 'z', 'num_depth']
raw_data.columns = column_names
raw_data.x = raw_data.x.astype(float)
raw_data.y = raw_data.y.astype(float)
raw_data.z = raw_data.z.astype(float)
raw_data.num_depth = raw_data.num_depth.astype(int)
raw_data = raw_data.drop(['date', 'time'], axis=1)
raw_data = raw_data.reset_index(drop=True)
# split data

dataframe = pd.DataFrame(columns=['x', 'y', 'z', 'i'])

print(raw_data.head())

for i in range(len(raw_data)):
    x = raw_data['x'][i]
    y = raw_data['y'][i]
    z = raw_data['z'][i]
    num_depth = raw_data.num_depth[i]
    depth_data = []
    intensity_data = []
    for j in range(num_depth):
        depth = float( raw_data.values[i, 4+2*j])
        intensity = float(raw_data.values[i, 5+2*j])
        # fill 0
        if(j == 0):
            depth_data.append(depth)
            intensity_data.append(intensity)
        else:
            if(depth - depth_data[j-1] >= interval):
                times = 1
                while(depth - depth_data[j-1] >= interval * times):
                    depth_data.append(depth_data[j-1] + interval * times)
                    intensity_data.append(0)
                    times += 1
                depth_data.append(depth)
                intensity_data.append(intensity)
    
    # do gaussian filter
    intensity_data = np.array(intensity_data)
    gaussian_kernel = cv2.getGaussianKernel(3, 0)
    intensity_data = cv2.filter2D(intensity_data, -1, gaussian_kernel)
    intensity_data = intensity_data.tolist()

    # adjust depth
    for j in range(len(depth_data)):
        newZ = z + depth_data[j] / 1000.0
        series = pd.Series({'x': x, 'y': y, 'z': newZ, 'i': intensity_data[j]})
        dataframe.append(series, ignore_index=True)
        # print(x, y, z, j)
    
    if(i % 1000 == 0):
        print('now is in {0}', i)

"""

# read data
df = pd.read_csv('TestData/3_in5_k3.txt', sep='\ ', header=None, engine='python')
df.columns = ['x', 'y', 'z', 'i']

# df = df.sort_values(by=[0, 1], ignore_index=True)
 
# df.x = df.x.round(3)
# df.y = df.y.round(3)

# first row 
first_x = df.x.min()
first_y = df.y.min()

row = df.where((df.x == first_x) & (df.y == first_y)).dropna()

# =============================================================
# one dimensional plot
# =============================================================

# plot original
plt.subplot(3, 1, 1)
plt.plot(row.z.to_numpy(), row.i.to_numpy())
plt.title('original one dimensional plot')

# first row and do otsu thresholding
glo_thresh, glo_result = cv2.threshold(df.i.to_numpy(dtype = np.uint16), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("threshold: ", glo_thresh)
thresh, result = cv2.threshold(row.i.to_numpy(dtype = np.uint16), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.subplot(3, 1, 2)
plt.plot(row.z.to_numpy(), result)
plt.title('Otsu Thresholding')

thresh2, result2 = cv2.threshold(row.i.to_numpy(dtype = np.uint8), thresh-10 , 255, cv2.THRESH_BINARY)
plt.subplot(3, 1, 3)
plt.plot(row.z.to_numpy(), result2)
plt.title('Thresholding with threshold-10')

plt.tight_layout()

plt.show()


# =============================================================
# two dimensional plot
# =============================================================

# first plate
first_plate = df.where((df.x == first_x)&(df.i >= (thresh-10))).dropna()


twod_data = np.array([first_plate.z.to_numpy(), first_plate.i.to_numpy()]).T
# print(twod_data)
# cluster = DBSCAN(eps=0.007, min_samples=3).fit(oned_data)
# print(cluster.labels_)

plt.plot(first_plate.y.to_numpy(), first_plate.z.to_numpy(), 'o')
plt.title('original two dimensional plot')
plt.show()