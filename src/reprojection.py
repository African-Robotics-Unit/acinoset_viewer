import matplotlib.pyplot as plt
import pickle
from calib import calib, utils
import build as bd
import cv2
import analyse as an
import numpy as np
import math
import glob
import os

scene_path = "C://Users//user-pc//Documents//Scripts//FYP_tests//GT//scene_sba.json"
#scene_fpath = 'C://Users//user-pc//Documents//Scripts//09//scene_sba.json'
currdir = "C://Users/user-pc/Documents/Scripts/FYP"
skel_name = "cheetah_serious"
skelly_dir = os.path.join(currdir, "skeletons", (skel_name + ".pickle"))
results_dir = os.path.join(currdir, "data", "results", (skel_name + ".pickle"))

project_dir = "C://Users//user-pc//Documents//Scripts//FYP_tests//GT"
df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
points_2d_df = utils.create_dlc_points_2d_file(df_paths)
#print(points_2d_df)
pts1=[]
pts2=[]
pts3=[]
pts4=[]
pts5=[]
pts6=[]
for i in range(100,110):
    pts1.append(points_2d_df[points_2d_df["frame"]==str("labeled-data\\09_03_2019LilyRun1CAM1\\img"+str(i)+".png")][["x", "y"]].values)
    pts2.append(points_2d_df[points_2d_df["frame"]==str("labeled-data\\09_03_2019LilyRun1CAM2\\img"+str(i)+".png")][["x", "y"]].values)
    pts3.append(points_2d_df[points_2d_df["frame"]==str("labeled-data\\09_03_2019LilyRun1CAM3\\img"+str(i)+".png")][["x", "y"]].values)
    pts4.append(points_2d_df[points_2d_df["frame"]==str("labeled-data\\09_03_2019LilyRun1CAM4\\img"+str(i)+".png")][["x", "y"]].values)
    pts5.append(points_2d_df[points_2d_df["frame"]==str("labeled-data\\09_03_2019LilyRun1CAM5\\img"+str(i)+".png")][["x", "y"]].values)
    pts6.append(points_2d_df[points_2d_df["frame"]==str("labeled-data\\09_03_2019LilyRun1CAM6\\img"+str(i)+".png")][["x", "y"]].values)
scene_path = "C://Users//user-pc//Documents//Scripts//FYP_tests//DLC//scene_sba.json"
project_dir = "C://Users//user-pc//Documents//Scripts//FYP_tests//DLC"
df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
points_2d_df = utils.create_dlc_points_2d_file(df_paths)
K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_path)
D_arr = D_arr.reshape((-1,4))
triangulate_func = calib.triangulate_points_fisheye
points_3d_df = calib.get_pairwise_3d_points_from_df(points_2d_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)

pts3d = []
for i in range(100,110):
    pts3d.append(np.array([points_3d_df[points_3d_df["frame"]==i][["x", "y", "z"]].values]))
pts3d = np.array(pts3d)
print(len(pts1[0]))


im = cv2.imread("C://Users//user-pc//Documents//Scripts//FYP//src//img100.png")
RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
fig = plt.figure()
ax = fig.add_subplot(312)
ax.imshow(RGB_img)
ax.title.set_text("Camera 2")

skel_dict = bd.load_skeleton(skelly_dir)
results = an.load_pickle(results_dir)

k, d, r, t, _ = utils.load_scene(scene_path)
d = d.reshape((-1,4))

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

pts_3d_trajopt = results

index_dict = {"0":23, "1":24, "2":6, "3":22, "4":11,
     "5":12, "6":13,"7":14,"8":15,"9":2,
      "10":3, "11":4,"12":17,"13":18, "14":19,
       "15":7,"16":8,"17":9}

pts_2d_traj = []
pts_2d_sba = []
cam = 5
for i in range(10):
    print(pts3d[i])
    pts_2d_sba.append(calib.project_points_fisheye(pts3d[i], k[cam],d[cam],r[cam],t[cam]))
    #pts_2d_ekf.append(calib.project_points_fisheye(pts_3d_ekf['positions'][40], k[cam],d[cam],r[cam],t[cam]))
    pts_2d_traj.append(calib.project_points_fisheye(pts_3d_trajopt['positions'][i+20], k[cam],d[cam],r[cam],t[cam]))

print(len(pts_2d_sba[0]))
print(len(pts_2d_traj[0]))

sba_errs = []
traj_errs = []

for frame in range(10):
    for part in range(18):
        ind = index_dict[str(part)]
        if not np.isnan(pts6[frame][ind][0]):
            sba_errs.append((pts_2d_sba[frame][ind][0] - pts6[frame][ind][0] + pts_2d_sba[frame][ind][1] - pts6[frame][ind][1])/2)
            traj_errs.append((pts_2d_traj[frame][part][0] - pts6[frame][ind][0] + pts_2d_traj[frame][part][1] - pts6[frame][ind][1])/2)

ax = fig.add_subplot(111)
plt.hist(traj_errs, bins=10)
plt.xlabel("Error in pixels")
plt.ylabel("Number of errors")
plt.title("Histogram showing the 2D reprojection errors in pixels for the trajectory optimisation (camera 3)")
plt.xlim(-600,600)
plt.grid()
plt.show()

std = np.std(traj_errs)
rmse = np.sqrt(np.mean(np.square(traj_errs)))
print(std)
print(rmse)

std = np.std(sba_errs)
rmse = np.sqrt(np.mean(np.square(sba_errs)))
print(std)
print(rmse)

"""
for i in range(18):
    plt.scatter(pts_2d_traj[1][i][0], pts_2d_traj[1][i][1], c="g")
for i in range(25):
    plt.scatter(pts2[i][0], pts2[i][1], c="b")
for i in range(25):
    plt.scatter(pts_2d_sba[1][i][0], pts_2d_sba[1][i][1], c="r")
plt.xlim(750,1850)
plt.ylim(1000,600)

im = cv2.imread("C://Users//user-pc//Documents//Scripts//FYP//src//img100cam1.png")
RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
ax = fig.add_subplot(311)
ax.imshow(RGB_img)
ax.title.set_text("Camera 1")

for i in range(18):
    plt.scatter(pts_2d_traj[0][i][0], pts_2d_traj[0][i][1], c="g")
for i in range(25):
    plt.scatter(pts1[i][0], pts1[i][1], c="b")
for i in range(25):
    plt.scatter(pts_2d_sba[0][i][0], pts_2d_sba[0][i][1], c="r")
plt.xlim(750,1850)
plt.ylim(1000,600)

im = cv2.imread("C://Users//user-pc//Documents//Scripts//FYP//src//img100cam3.png")
RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
ax = fig.add_subplot(313)
ax.imshow(RGB_img)
ax.title.set_text("Camera 3")

for i in range(18):
    plt.scatter(pts_2d_traj[2][i][0], pts_2d_traj[2][i][1], c="g")
for i in range(25):
    plt.scatter(pts3[i][0], pts3[i][1], c="b")
for i in range(25):
    plt.scatter(pts_2d_sba[2][i][0], pts_2d_sba[2][i][1], c="r")
plt.xlim(450,1550)
plt.ylim(1000,600)

plt.show()

for cam in range(6):
    print("Cam: ")
    print(cam)

    xs = [row[0] for row in pts_2d_gt[cam]]
    ys = [row[1] for row in pts_2d_gt[cam]]

    print("sba:")
    n=0
    sum = 0
    for i in range(len(xs)):
        if not math.isnan(xs[i]):
            if not math.isnan(pts_2d_sba[cam][:,0][i]):
                sum += (pts_2d_sba[cam][:,0][i] - xs[i])**2
                sum += (pts_2d_sba[cam][:,1][i] - ys[i])**2
            else:
                sum+=1080**2
            n+=2

    rmse = np.sqrt(sum/n-1)/np.sqrt(n)
    stdev = np.sqrt(sum/(n-1))
    if not math.isnan(rmse):
        sba_tot+=rmse
    print(rmse)
    print(stdev)

    print("ekf:")
    n=0
    sum = 0
    for i in range(len(xs)):
        if not math.isnan(xs[i]):
            if not math.isnan(pts_2d_ekf[cam][:,0][i]):
                sum += (pts_2d_ekf[cam][:,0][i] - xs[i])**2
                sum += (pts_2d_ekf[cam][:,1][i] - ys[i])**2
            else:
                sum+=1080**2
            n+=2

    rmse = np.sqrt(sum/n-1)/np.sqrt(n)
    stdev = np.sqrt(sum/(n-1))
    if not math.isnan(rmse):
        ekf_tot+=rmse
    print(rmse)
    print(stdev)

    print("traj:")
    n=0
    sum = 0
    for i in range(len(xs)):
        if not math.isnan(xs[i]):
            if not math.isnan(pts_2d_traj[cam][:,0][i]):
                sum += (pts_2d_traj[cam][:,0][i] - xs[i])**2
                sum += (pts_2d_traj[cam][:,1][i] - ys[i])**2
            else:
                sum+=1080**2
            n+=2

    rmse = np.sqrt(sum/n-1)/np.sqrt(n)
    stdev = np.sqrt(sum/(n-1))
    if not math.isnan(rmse):
        traj_tot+=rmse
    print(rmse)
    print(stdev)

print("SBA:")
print(sba_tot/6)
print("EKF:")
print(ekf_tot/6)
print("TRAJ:")
print(traj_tot/6)
"""

