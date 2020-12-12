from pickle import load
from typing import Dict
import pickle
from pyomo.core.base.constraint import ConstraintList
import sympy as sp
import numpy as np
from scipy import stats
import os
import glob
from calib import utils, calib, plotting, app, extract
from scipy import stats
from pprint import pprint
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.core.base.PyomoModel import ConcreteModel
import matplotlib.pyplot as plt
import build as bd
import analyse as an

def load_pickle(pickle_file):
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    #print(data["x"][1])
    #print(data["positions"][2])
    return(data)

def compare_plots() -> None:

    fig = plt.figure()
    #ax =fig.add_subplot(111, projection='3d')
    pose_dict = {}
    currdir = "C://Users/user-pc/Documents/Scripts/FYP"
    skel_name = "cheetah_serious"
    skelly_dir = os.path.join(currdir, "skeletons", (skel_name + ".pickle"))
    results_dir = os.path.join(currdir, "data", "results", (skel_name + ".pickle"))

    skel_dict = bd.load_skeleton(skelly_dir)
    results = an.load_pickle(results_dir)
    links = skel_dict["links"]
    markers = skel_dict["markers"]
    traj_errs=[]

    # --- Ground Truth Data ---

    #ax.view_init(elev=20,azim=110)
    scene_path = "C://Users//user-pc//Documents//Scripts//FYP_tests//GT//scene_sba.json"
    project_dir = "C://Users//user-pc//Documents//Scripts//FYP_tests//GT"
    df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
    points_2d_df = utils.create_dlc_points_2d_file(df_paths)
    K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_path)
    D_arr = D_arr.reshape((-1,4))
    gt_dict = {}
    triangulate_func = calib.triangulate_points_fisheye
    #points_2d_filtered_df = points_2d_df[points_2d_df['likelihood']>0.5]
    #print(points_2d_df)
    points_3d_df = calib.get_pairwise_3d_points_from_df(points_2d_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)
    for fn in range(105,110):
        pts = points_3d_df[points_3d_df["frame"]==str(fn)][["x", "y", "z", "marker"]].values

        for pt in pts:
            gt_dict[pt[3]] = [pt[0], pt[1], pt[2]]
        
        #for pt in pts:
            #if pt[3] in markers:
                #ax.scatter(pt[0], pt[1], pt[2], c="b")
        
        #ax.xaxis.pane.fill = False
        #ax.yaxis.pane.fill = False
        #ax.zaxis.pane.fill = False

        #for link in links:
            #if len(link)>1:
                #ax.plot3D([gt_dict[link[0]][0], gt_dict[link[1]][0]],
                #[gt_dict[link[0]][1], gt_dict[link[1]][1]],
                #[gt_dict[link[0]][2], gt_dict[link[1]][2]], c="b")
    
    # --- Triangulation ---
    #ax = fig.add_subplot(323, projection='3d')
    #ax.title.set_text('Sparse Bundle Adjustment - Side')
    #ax.view_init(elev=20,azim=10)
        scene_path = "C://Users//user-pc//Documents//Scripts//FYP_tests//DLC//scene_sba.json"
        project_dir = "C://Users//user-pc//Documents//Scripts//FYP_tests//DLC"
        df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
        points_2d_df = utils.create_dlc_points_2d_file(df_paths)
        K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_path)
        D_arr = D_arr.reshape((-1,4))
        triangulate_func = calib.triangulate_points_fisheye
        points_3d_df = calib.get_pairwise_3d_points_from_df(points_2d_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)
        sba_errs = []
        sba_dict = {}
        #for fn in range(100,110):
        pts = points_3d_df[points_3d_df["frame"]==fn][["x", "y", "z", "marker"]].values
        #print(pts)
        #for pt in pts:
            #if pt[3] in markers:
                #ax.scatter(pt[0], pt[1], pt[2], c="r")

        for pt in pts:
            sba_dict[pt[3]] = [pt[0], pt[1], pt[2]]

        #for mark in markers:
           # sba_errs.append((sba_dict[mark][0] - gt_dict[mark][0] + sba_dict[mark][1] - gt_dict[mark][1] + sba_dict[mark][2] - gt_dict[mark][2])/3)
        
        #for link in links:
            #if len(link)>1:
                #ax.plot3D([sba_dict[link[0]][0], sba_dict[link[1]][0]],
                #[sba_dict[link[0]][1], sba_dict[link[1]][1]],
               # [sba_dict[link[0]][2], sba_dict[link[1]][2]], c="r")
        

        #plt.xlim(-2,5)
        #plt.ylim(6,9)   
        #ax.xaxis.pane.fill = False
        #ax.yaxis.pane.fill = False
        #ax.zaxis.pane.fill = False
    
        # --- Full Traj Opt ---
        #ax.view_init(elev=45,azim=130)
        
        #for frame in range(20,30):
        frame=fn-80
        for i in range(len(markers)):
            pose_dict[markers[i]] = [results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2]]
            #ax.scatter(results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2], c="g")
            
        #for link in links:
            #if len(link)>1:
                #ax.plot3D([pose_dict[link[0]][0], pose_dict[link[1]][0]],
                #[pose_dict[link[0]][1], pose_dict[link[1]][1]],
                #[pose_dict[link[0]][2], pose_dict[link[1]][2]], c="g")

        #print(pose_dict)
        #print(gt_dict)
        for mark in markers:
            if not np.isnan(gt_dict[mark][0]):
                traj_errs.append((pose_dict[mark][0] - gt_dict[mark][0] + pose_dict[mark][1] - gt_dict[mark][1] + pose_dict[mark][2] - gt_dict[mark][2])/3)
        #ax.xaxis.pane.fill = False
        #ax.yaxis.pane.fill = False
        #ax.zaxis.pane.fill = False
        #plt.show()

    print(traj_errs)
    std = np.std(traj_errs)
    rmse = np.sqrt(np.mean(np.square(traj_errs)))
    print(std)
    print(rmse)

    
    #ax.view_init(elev=80,azim=60)
    fig.add_subplot(111)
    plt.hist(traj_errs)
    plt.title("Histogram of 3D errors from traj. opt. results with pairwise predictions")
    plt.xlabel("Error in metres")
    plt.ylabel("Number of errors")
    plt.xlim(-2.5,2.5)
    plt.grid()
    plt.show()
    

def plot_states():

    fig = plt.figure()
    ax =fig.add_subplot(111)
    pose_dict = {}
    currdir = "C://Users/user-pc/Documents/Scripts/FYP"
    skel_name = "cheetah_serious"
    skelly_dir = os.path.join(currdir, "skeletons", (skel_name + ".pickle"))
    results_dir = os.path.join(currdir, "data", "results", (skel_name + ".pickle"))

    skel_dict = bd.load_skeleton(skelly_dir)
    results = an.load_pickle(results_dir)
    links = skel_dict["links"]
    markers = skel_dict["markers"]
    print(len(results["dx"]))
    xs=[]
    ys=[]
    zs=[]
    for state in results["x"]:
        xs.append(state[3])
        ys.append(state[21])
        zs.append(state[39])
    ax.plot(xs, label="phi")
    ax.plot(ys, label="theta")
    ax.plot(zs, label="psi")
    plt.xlabel("Frame")
    plt.ylabel("Magnitude of state (rad)")
    plt.title("Tail phi, theta, and psi states for a 100-frame trajectory")
    plt.legend()
    plt.grid()
    plt.show()


def compare_plots_cloud() -> None:

    fig = plt.figure()
    pose_dict = {}
    currdir = "C://Users/user-pc/Documents/Scripts/FYP"
    skel_name = "cheetah_serious"
    skelly_dir = os.path.join(currdir, "skeletons", (skel_name + ".pickle"))
    results_dir = os.path.join(currdir, "data", "results", (skel_name + ".pickle"))

    skel_dict = bd.load_skeleton(skelly_dir)
    results = an.load_pickle(results_dir)
    links = skel_dict["links"]
    markers = skel_dict["markers"]

    # --- Ground Truth Data ---
    
    ax = fig.add_subplot(321, projection='3d')
    ax.title.set_text('Ground Truth - Side')
    #ax.view_init(elev=20,azim=110)
    scene_path = "C://Users//user-pc//Documents//Scripts//FYP_tests//GT//scene_sba.json"
    project_dir = "C://Users//user-pc//Documents//Scripts//FYP_tests//GT"
    df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
    points_2d_df = utils.create_dlc_points_2d_file(df_paths)
    K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_path)
    D_arr = D_arr.reshape((-1,4))
    triangulate_func = calib.triangulate_points_fisheye
    #points_2d_filtered_df = points_2d_df[points_2d_df['likelihood']>0.5]
    #print(points_2d_df)
    points_3d_df = calib.get_pairwise_3d_points_from_df(points_2d_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)

    for fn in range(100,110):
        pts = points_3d_df[points_3d_df["frame"]==str(fn)][["x", "y", "z", "marker"]].values
    #print(pts)

        gt_dict = {}

        for pt in pts:
            gt_dict[pt[3]] = [pt[0], pt[1], pt[2]]
        
        for pt in pts:
            if pt[3] in markers:
                ax.scatter(pt[0], pt[1], pt[2], c="b")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax = fig.add_subplot(322, projection='3d')
    ax.title.set_text('Ground Truth - Top')
    ax.view_init(elev = 80,azim=60)

    for fn in range(100,110):
        pts = points_3d_df[points_3d_df["frame"]==str(fn)][["x", "y", "z", "marker"]].values
    #print(pts)

        gt_dict = {}

        for pt in pts:
            gt_dict[pt[3]] = [pt[0], pt[1], pt[2]]
        
        for pt in pts:
            if pt[3] in markers:
                ax.scatter(pt[0], pt[1], pt[2], c="b")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    #ax.set_zlabel('Z Axis')
    #for link in links:
        #if len(link)>1:
            #ax.plot3D([gt_dict[link[0]][0], gt_dict[link[1]][0]],
            #[gt_dict[link[0]][1], gt_dict[link[1]][1]],
            #[gt_dict[link[0]][2], gt_dict[link[1]][2]], c="b")
    
    # --- Triangulation ---
    ax = fig.add_subplot(323, projection='3d')
    ax.title.set_text('Sparse Bundle Adjustment - Side')
    #ax.view_init(elev=20,azim=10)
    scene_path = "C://Users//user-pc//Documents//Scripts//FYP_tests//DLC//scene_sba.json"
    project_dir = "C://Users//user-pc//Documents//Scripts//FYP_tests//DLC"
    df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
    points_2d_df = utils.create_dlc_points_2d_file(df_paths)
    K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_path)
    D_arr = D_arr.reshape((-1,4))
    triangulate_func = calib.triangulate_points_fisheye
    points_3d_df = calib.get_pairwise_3d_points_from_df(points_2d_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)

    for fn in range(100,110):
        pts = points_3d_df[points_3d_df["frame"]==fn][["x", "y", "z", "marker"]].values
        #print(pts)
        for pt in pts:
            if pt[3] in markers:
                ax.scatter(pt[0], pt[1], pt[2], c="r")

        gt_dict = {}

        for pt in pts:
            gt_dict[pt[3]] = [pt[0], pt[1], pt[2]]     
    plt.xlim(-2,5)
    plt.ylim(6,9)   
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax = fig.add_subplot(324, projection='3d')
    ax.title.set_text('Sparse Bundle Adjustment - Top')
    ax.view_init(elev=80,azim=60)
    for fn in range(100,110):
        pts = points_3d_df[points_3d_df["frame"]==fn][["x", "y", "z", "marker"]].values
        #print(pts)
        for pt in pts:
            if pt[3] in markers:
                ax.scatter(pt[0], pt[1], pt[2], c="r")

        gt_dict = {}

        for pt in pts:
            gt_dict[pt[3]] = [pt[0], pt[1], pt[2]]     
    plt.xlim(-2,5)
    plt.ylim(6,9)  
    """
    for link in links:
        if len(link)>1:
            ax.plot3D([gt_dict[link[0]][0], gt_dict[link[1]][0]],
            [gt_dict[link[0]][1], gt_dict[link[1]][1]],
            [gt_dict[link[0]][2], gt_dict[link[1]][2]], c="r")
    """
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    #ax.set_zlabel('Z Axis')

    # --- Full Traj Opt ---
    
    ax = fig.add_subplot(325, projection='3d')
    ax.title.set_text('Trajectory Optimisation - Side')
    #ax.view_init(elev=45,azim=130)

    for frame in range(20, 30):
    
        for i in range(len(markers)):
            pose_dict[markers[i]] = [results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2]]
            ax.scatter(results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2], c="g")

           
        #for link in links:
            #if len(link)>1:
                #ax.plot3D([pose_dict[link[0]][0], pose_dict[link[1]][0]],
                #[pose_dict[link[0]][1], pose_dict[link[1]][1]],
                #[pose_dict[link[0]][2], pose_dict[link[1]][2]], c="g")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    ax = fig.add_subplot(326, projection='3d')
    ax.title.set_text('Trajectory Optimisation - Top')
    ax.view_init(elev=80,azim=60)

    for frame in range(20, 30):
    
        for i in range(len(markers)):
            pose_dict[markers[i]] = [results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2]]
            ax.scatter(results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2], c="g")

           
        #for link in links:
            #if len(link)>1:
                #ax.plot3D([pose_dict[link[0]][0], pose_dict[link[1]][0]],
                #[pose_dict[link[0]][1], pose_dict[link[1]][1]],
                #[pose_dict[link[0]][2], pose_dict[link[1]][2]], c="g")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    #ax.set_zlabel('Z Axis')
    
    plt.show()
    #print(pts)
    

def get_error(part1, part2) -> float:
    pickle_path = "C://Users//user-pc//Desktop//pw_errors.pickle"
    data = load_pickle(pickle_path)
    xerrs = []
    yerrs = []
    sum=0
    for frame in range(len(data["x"])):
        sum+= (abs(data["x"][frame,part1,part2]))
        xerrs.append(data["x"][frame,part1,part2])
        yerrs.append(data["y"][frame,part1,part2])
    
    #xstd = np.std(xerrs)
    #ystd = np.std(yerrs)
    #return(xstd, ystd)
    #xrmse = np.sqrt(np.mean(xerrs))
    #yrmse = np.sqrt(np.mean(yerrs))
    return(sum/len(data["x"]))

def hist(part1,part2):
    pickle_path = "C://Users//user-pc//Desktop//pw_errors.pickle"
    data = load_pickle(pickle_path)
    xerrs = []
    yerrs = []
    sum=0
    for frame in range(len(data["x"])):
        sum+= np.abs(data["x"][frame,part1,part2])
        xerrs.append(data["x"][frame,part1,part2])
        yerrs.append(data["y"][frame,part1,part2])
    
    print("MAE:")
    print(sum/8582)
    print("St Dev:")
    print(np.std(xerrs))
    plt.hist(xerrs, bins=80)  # `density=False` would make counts
    plt.ylabel('Number of Errors in Bin')
    plt.xlabel('Error in pixels')
    plt.xlim(-1000,1000)
    plt.title("Histogram showing errors of the right rear ankle predicted from the right knee")
    plt.grid()
    plt.show()

def plot_loss():
    r_x = np.arange(-40,40, 1e-1)
    r_y1 = [redescending_loss(i, 3, 10, 20) for i in r_x]
    r_y2 = [redescending_loss(i, 3, 5, 15) for i in r_x]
    plt.figure(figsize=(5,3))
    plt.plot(r_x,r_y1, label="Redescending (a=3, b=10, c=20)")
    plt.plot(r_x,r_y2, label="Redescending (a=3, b=5, c=15)")
    plt.grid()
    plt.xlabel("e")
    plt.ylabel("C(e)")
    plt.legend(loc = 'bottom right')
    plt.show()

def func_step(start, x):
        return 1/(1+np.e**(-1*(x - start)))

def func_piece(start, end, x):
        return func_step(start, x) - func_step(end, x)

def redescending_loss(err, a, b, c):
    e = abs(err)
    cost = 0.0
    cost += (1 - func_step(a, e))/2*e**2
    cost += func_piece(a, b, e)*(a*e - (a**2)/2)
    cost += func_piece(b, c, e)*(a*b - (a**2)/2 + (a*(c-b)/2)*(1-((c-e)/(c-b))**2))
    cost += func_step(c, e)*(a*b - (a**2)/2 + (a*(c-b)/2))
    return cost

if __name__=="__main__":
    pickle_path = "C://Users//user-pc//Desktop//pw_errors.pickle"
    #compare_plots()
    skel_path = "C://Users//user-pc//Desktop//cheetah_serious.pickle"
    #plot_states()
    #data = load_pickle(skel_path)
    #print(data["markers"])
    #print(get_error(14,15))
    #hist(11,12)
    #plot_loss()
    """
    data = load_pickle(pickle_path)
    print(get_error(14,15))
    
    print(data["x"][373,11,12])
    print(data["y"][373,11,12])
    print(get_error(12,11))
    for i in range(25):
        for j in range(25):
            print(get_error(i,j))
    """
