from os import link
import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog, ttk
import os
import time
import glob
from pandas.io.pytables import to_hdf
from pyomo.core.expr import current
from calib import calib, app, extract, utils, plotting
import get_points as pt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np
import analyse as an
import build as bd
import matplotlib.image as img
from matplotlib.animation import FuncAnimation, PillowWriter, Animation

class Application(tk.Tk):

    def __init__(self, *args, **kwargs):
        """
        Initialise the main application
        """

        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Montserrat', size=18)
        self.normal_font = tkfont.Font(family='Montserrat', size=12)
        self.project_dir = "No project folder chosen"
        self.sba_dir = "No SBA file chosen"

        container = tk.Frame(self, width=1130, height=720)
        nav_bar = tk.Frame(self, width=150, height=720, background="#2c3e50")
        container.pack_propagate(False)
        nav_bar.pack_propagate(False)
        nav_bar.place(relx=0, rely=0)
        container.place(x=150,y=0)
        
        self.home_label = tk.Label(nav_bar, text="Home", font=self.title_font, bg="#2c3e50", fg="#ffffff", width=150, cursor="hand2")
        self.home_label.place(relx=0.5, y=30, anchor="center")
        self.build_label = tk.Label(nav_bar, text="Build", font=self.title_font, bg="#2c3e50", fg="#ffffff", width=150, cursor="hand2")
        self.build_label.place(relx=0.5, y=80, anchor="center")
        self.analyse_label = tk.Label(nav_bar, text="Analyse", font=self.title_font, bg="#2c3e50", fg="#ffffff", width=150, cursor="hand2")
        self.analyse_label.place(relx=0.5, y=130, anchor="center")

        self.home_label.bind("<Enter>", self.home_on_enter)
        self.home_label.bind("<Leave>", self.home_on_leave)
        self.home_label.bind("<Button-1>", self.home)

        self.build_label.bind("<Enter>", self.build_on_enter)
        self.build_label.bind("<Leave>", self.build_on_leave)
        self.build_label.bind("<Button-1>", self.build)

        self.analyse_label.bind("<Enter>", self.analyse_on_enter)
        self.analyse_label.bind("<Leave>", self.analyse_on_leave)
        self.analyse_label.bind("<Button-1>", self.analyse)

        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.place(x=0,y=0)

        self.show_frame("StartPage")

    # --- Binding events ---
    
    def home(self, event):
        self.show_frame("StartPage")

    def home_on_enter(self, event):
        self.home_label.configure(bg="#34495e")

    def home_on_leave(self, event):
        self.home_label.configure(bg="#2c3e50")

    def build(self, event):
        self.show_frame("PageOne")

    def build_on_enter(self, event):
        self.build_label.configure(bg="#34495e")

    def build_on_leave(self, event):
        self.build_label.configure(bg="#2c3e50")

    def analyse(self, event):
        self.show_frame("PageTwo")

    def analyse_on_enter(self, event):
        self.analyse_label.configure(bg="#34495e")

    def analyse_on_leave(self, event):
        self.analyse_label.configure(bg="#2c3e50")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        """
        Initialise a frame for the home page
        """
        tk.Frame.__init__(self, parent, height=720, width=1130, bg="#ffffff")
        self.controller = controller
        self.pack_propagate(False)
        combo_vids = ttk.Combobox(self, values=["- Select Video -"])
        
        # --- Define functions to be used by GUI components ---

        def choose_folder():
            currdir = os.getcwd()
            controller.project_dir = filedialog.askdirectory(parent=self, initialdir=currdir, title='Please Select a labeled-data Folder:')
            if len(controller.project_dir) > 0:
                print("You chose %s" % controller.project_dir)
                label_folder.configure(text=controller.project_dir)
            dirs = glob.glob(os.path.join(controller.project_dir, "*"))
            #print(dirs)
            video_names = get_video_names(dirs)

            label_vids = tk.Label(self, text="Choose a video:", font=controller.normal_font, bg="#ffffff")
            label_vids.place(relx=0.5, rely=0.6, anchor="center")

            combo_vids.configure(values=video_names)
            combo_vids.place(relx=0.5,rely=0.65, anchor = "center")

            combo_vids.bind("<<ComboboxSelected>>", set_vid)

        def set_vid(self):
            controller.vid = combo_vids.get()
            print(controller.vid)

        def get_video_names(dir_list):
            vid_names = []
            for dir in dir_list:
                vid_string = os.path.split(dir.split("CAM")[0])[1]
                if vid_string not in vid_names:
                    vid_names.append(vid_string)

            return(vid_names)

        def choose_sba():
            currdir = os.getcwd()
            controller.sba_dir = filedialog.askopenfilename(parent=self, initialdir=currdir, title='Please choose an SBA .json file:')
            label_sba.configure(text=controller.sba_dir)
            print(controller.sba_dir)


        # --- Define and place GUI components ---

        label_folder = tk.Label(self, text=controller.project_dir, font=controller.normal_font, bg="#ffffff")
        label_folder.place(relx=0.5, rely=0.35, anchor="center")
        label = tk.Label(self, text="Setup", font=controller.title_font, bg="#ffffff")
        label.place(relx=0.5, rely=0.1, anchor="center")
        button1 = tk.Button(self, text="Choose labeled-data Folder", command=choose_folder)
        button1.place(relx=0.5, rely=0.3, anchor="center")

        button1 = tk.Button(self, text="Choose SBA File", command=choose_sba)
        button1.place(relx=0.5, rely=0.45, anchor="center")
        label_sba = tk.Label(self, text=controller.sba_dir, font=controller.normal_font, bg="#ffffff")
        label_sba.place(relx=0.5, rely=0.5, anchor="center")
        

class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        """
        Initialise a frame for the build page
        """
        tk.Frame.__init__(self, parent, height=720, width=1130, background="#ffffff")
        self.controller = controller
        self.pack_propagate(False)

        # --- Initialise class-wide variables ---

        f = Figure(figsize=(4,4), dpi=100)
        a = f.add_subplot(111, projection="3d")
        a.view_init(elev=20., azim=60)

        f_2d_left = Figure(figsize=(4,4), dpi=100)
        a_2d_1 = f_2d_left.add_subplot(311)
        a_2d_2 = f_2d_left.add_subplot(312)
        a_2d_3 = f_2d_left.add_subplot(313)

        f_2d_right = Figure(figsize=(4,4), dpi=100)
        a_2d_4 = f_2d_right.add_subplot(311)
        a_2d_5 = f_2d_right.add_subplot(312)
        a_2d_6 = f_2d_right.add_subplot(313)

        axes_list = [a_2d_1, a_2d_2, a_2d_3, a_2d_4, a_2d_5, a_2d_6]

        self.frame = 0
        x_free = tk.IntVar()
        y_free = tk.IntVar()
        z_free = tk.IntVar()

        parts_dict = {}
        points_dict = {}
        dof_dict = {}
        skel_dict = {}
        links_list = []

        self.points_3d_df={}
        self.points_2d_df={}

        # --- Define functions to be used by GUI components ---

        def update_canvas() -> None:
            """
            Replots canvas on the GUI with updated points
            """
            canvas = FigureCanvasTkAgg(f, self)
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            canvas._tkcanvas.place(relx=0.5, rely=0.2, anchor="center")
            
        def update_2d_views() -> None:
            canvas_2d_1 = FigureCanvasTkAgg(f_2d_left, self)
            canvas_2d_1.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            canvas_2d_1._tkcanvas.place(relx=0.2, rely=0.5, anchor="center")

            canvas_2d_2 = FigureCanvasTkAgg(f_2d_right, self)
            canvas_2d_2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            canvas_2d_2._tkcanvas.place(relx=0.8, rely=0.5, anchor="center")

        def load_gt_data() -> None:
            """
            Loads the GT points from a given folder
            """
            data_dir = controller.project_dir
            sba_dir = controller.sba_dir
            vid = controller.vid

            markers = ["r_eye", "l_eye", "nose", "neck_base", "r_shoulder", "r_front_knee", "r_front_ankle", "spine",
             "tail_base", "r_hip", "r_back_knee", "r_back_ankle", "tail1", "tail2", "l_shoulder", "l_front_knee",
              "l_front_ankle", "l_hip", "l_back_knee", "l_back_ankle"]

            K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(sba_dir)
            D_arr = D_arr.reshape((-1,4))

            print(f"\n\n\nLoading data")
            print(data_dir)
            folder_paths = sorted(glob.glob(os.path.join(data_dir, vid+'*')))
            h5_paths=[]
            for folder in folder_paths:
                h5_paths.append(glob.glob(os.path.join(folder, "*.h5"))[0])
            self.points_2d_df = utils.create_dlc_points_2d_file(h5_paths)
            print(self.points_2d_df)
            triangulate_func = calib.triangulate_points_fisheye
            self.points_3d_df = calib.get_pairwise_3d_points_from_df(self.points_2d_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)
            self.points_3d_df = self.points_3d_df[self.points_3d_df["marker"].isin(markers)]
            self.frame = int(np.min(self.points_3d_df["frame"]))
            label_frame.configure(text=self.frame)
            plot_cheetah(self.frame, self.points_3d_df, self.points_2d_df)
        
        def plot_cheetah(frame, df_3d, df_2d) -> None:

            K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(controller.sba_dir)
            D_arr = D_arr.reshape((-1,4))

            label_frame.configure(text=frame)

            markers = ["r_eye", "l_eye", "nose", "neck_base", "r_shoulder", "r_front_knee", "r_front_ankle", "spine",
             "tail_base", "r_hip", "r_back_knee", "r_back_ankle", "tail1", "tail2", "l_shoulder", "l_front_knee",
              "l_front_ankle", "l_hip", "l_back_knee", "l_back_ankle"]
            
            links = [[0,2], [1,2], [2,3], [3,4], [4,5], [5,6], [3,7], [7,8], [8,9], [9,10], [10,11], [8,12], [12,13],
             [14, 3], [14,15], [15,16], [8,17], [17,18], [18,19]]

            #df_3d = df_3d[df_3d["marker"].isin(markers)]
            #print(df_3d)
            pts3d = []
            pts3d.append(np.array([df_3d[df_3d["frame"]==str(frame)][["x", "y", "z"]].values]))
            pts3d = np.array(pts3d)

            for part in markers:
                pts = df_3d[df_3d["marker"]==part][df_3d["frame"]==str(frame)][["x", "y", "z"]].values
                #print(pts)
                parts_dict[part] = [pts[0][0], pts[0][1], pts[0][2]]
                points_dict[part] = a.scatter(parts_dict[part][0],parts_dict[part][1],parts_dict[part][2])
            
            for link in links:
                part1 = markers[link[0]]
                part2 = markers[link[1]]
                a.plot3D([parts_dict[part1][0], parts_dict[part2][0]],
             [parts_dict[part1][1], parts_dict[part2][1]], 
             [parts_dict[part1][2], parts_dict[part2][2]], 'b')
            
            image_tails = list(set(df_2d["frame"].values))
            image_head = os.path.split(controller.project_dir)[0]
            print(image_tails)
            for i, axis in enumerate(axes_list):
                cam=i+1
                #print("CAM"+str(cam))
                pts_2d_cam = []
                for tail in image_tails:
                    if "CAM"+str(cam) in tail and str(frame) in tail:
                        image_path = os.path.join(image_head, tail)
                        image = img.imread(image_path)
                        axis.imshow(image)
                        pts_2d_cam.append(calib.project_points_fisheye(pts3d, K_arr[i],D_arr[i],R_arr[i],t_arr[i]))
                        #print(pts_2d_cam)
                        x_s = []
                        y_s = []
                        for point in pts_2d_cam[0]:
                            if not np.isnan(point[0]):
                                axis.scatter(point[0], point[1], s=10)
                                x_s.append(point[0])
                                y_s.append(point[1])
                        minx = np.min(x_s)*0.9
                        miny = np.min(y_s)*0.9
                        maxx = np.max(x_s)*1.1
                        maxy = np.max(y_s)*1.1
                        axis.set_xlim(minx, maxx)
                        axis.set_ylim(maxy, miny)
            
            combo_move.configure(values=markers)
            
            update_canvas()
            update_2d_views()

        def rotate_right() -> None:
            """
            Rotates the axes right
            """
            azimuth = a.azim
            a.view_init(elev=20., azim=azimuth+10)
            update_canvas()
        
        def rotate_left() -> None:
            """
            Rotates the axes left
            """
            azimuth = a.azim
            a.view_init(elev=20., azim=azimuth-10)
            update_canvas()
        
        def move_point() -> None:
            """
            Moves/places the selected point to the defined x, y, z
            """
            part_to_move = combo_move.get()
            frame = self.frame

            if part_to_move in points_dict:
                point_to_move = points_dict[part_to_move]
                point_to_move.remove()

            new_x = float(x_spin.get())
            new_y = float(y_spin.get())
            new_z = float(z_spin.get())
            print(self.points_3d_df.keys())
            vals = self.points_3d_df[self.points_3d_df["marker"]==part_to_move][self.points_3d_df["frame"]==str(frame)][["x", "y", "z"]].values
            print(vals[0])
            self.points_3d_df = self.points_3d_df.replace(vals[0][0], new_x)
            self.points_3d_df = self.points_3d_df.replace(vals[0][1], new_y)
            self.points_3d_df = self.points_3d_df.replace(vals[0][2], new_z)
            vals = self.points_3d_df[self.points_3d_df["marker"]==part_to_move][self.points_3d_df["frame"]==str(frame)][["x", "y", "z"]].values
            print(vals[0])

            points_dict[part_to_move] = a.scatter(new_x,new_y, new_z)
            parts_dict[part_to_move] = [new_x, new_y, new_z]

            plot_cheetah(self.frame, self.points_3d_df, self.points_2d_df)

        def save_labels() -> None:
            """
            Writes the currently built skeleton to a pickle file
            """
            currdir = os.getcwd()
            results_file = os.path.join(currdir, "results", controller.vid+".pickle")
            file_data = self.points_3d_df
            with open(results_file, 'wb') as f:
                pickle.dump(file_data, f)
    
            print(f'save {results_file}')
        
        def next_frame() -> None:
            """
            Plots the next frame of the results
            """
            self.frame+=1
            a.clear()
            for axis in axes_list:
                axis.clear()
            plot_cheetah(self.frame, self.points_3d_df, self.points_2d_df)

        def prev_frame() -> None:
            """
            Plots the previous frame of the results
            """
            self.frame-=1
            a.clear()
            for axis in axes_list:
                axis.clear()
            plot_cheetah(self.frame, self.points_3d_df, self.points_2d_df)
        
        def update_spins(self) -> None:
            part = combo_move.get()
            x_free.set(parts_dict[part][0])
            y_free.set(parts_dict[part][1])
            z_free.set(parts_dict[part][2])

        # --- Define and place GUI components ---

        update_canvas()

        combo_move = ttk.Combobox(self, values=["Empty"])
        combo_move.place(relx=0.5,rely=0.6, anchor = "center")
        combo_move.bind("<<ComboboxSelected>>", update_spins)

        x_spin = tk.Spinbox(self, from_=-10, to=10, increment=0.01, textvariable=x_free, format = "%.2f")
        x_spin.place(relx=0.5, rely=0.65, anchor="center")
        y_spin = tk.Spinbox(self, from_=-10, to=10, increment=0.01, textvariable=y_free, format = "%.2f")
        y_spin.place(relx=0.5, rely=0.7, anchor="center")
        z_spin = tk.Spinbox(self, from_=-10, to=10, increment=0.01, textvariable=z_free, format = "%.2f")
        z_spin.place(relx=0.5, rely=0.75, anchor="center")

        label_x = tk.Label(self, text="x: ", font=controller.normal_font, background="#ffffff")
        label_x.place(relx=0.43, rely=0.65, anchor="center")
        label_y = tk.Label(self, text="y: ", font=controller.normal_font, background="#ffffff")
        label_y.place(relx=0.43, rely=0.7, anchor="center")
        label_z = tk.Label(self, text="z: ", font=controller.normal_font, background="#ffffff")
        label_z.place(relx=0.43, rely=0.75, anchor="center")

        button_update = tk.Button(self, text="Move", command=move_point)
        button_update.place(relx=0.5, rely=0.8, anchor="center")

        button = tk.Button(self, text="Load Points", command=load_gt_data)
        button.place(relx=0.45, rely=0.95, anchor="center")

        button_right = tk.Button(self, text="-->", command=rotate_right)
        button_right.place(relx=0.55, rely=0.5, anchor="center")
        button_left = tk.Button(self, text="<--", command=rotate_left)
        button_left.place(relx=0.45, rely=0.5, anchor="center")

        button_save = tk.Button(self, text="Save Labels", command=save_labels)
        button_save.place(relx=0.55, rely=0.95, anchor="center")

        button_next = tk.Button(self, text="next", command=next_frame)
        button_next.place(relx=0.57, rely=0.9, anchor="center")

        button_prev = tk.Button(self, text="Prev", command=prev_frame)
        button_prev.place(relx=0.43, rely=0.9, anchor="center")

        label_frame = tk.Label(self, text=self.frame, font=controller.normal_font, background="#ffffff")
        label_frame.place(relx=0.5, rely=0.9, anchor="center")

class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        """
        Initialise a frame for the analyse page
        """
        tk.Frame.__init__(self, parent, height=720, width=1130, background="#ffffff")
        self.controller = controller
        self.pack_propagate(False)
        self.current_frame = 0

        f = Figure(figsize=(4,4), dpi=100)
        a = f.add_subplot(111, projection="3d")
        a.view_init(elev=20., azim=60)

        def rotate_right() -> None:
            """
            Rotates the axes right
            """
            azimuth = a.azim
            a.view_init(elev=20., azim=azimuth+10)
            update_canvas()
        
        def rotate_left() -> None:
            """
            Rotates the axes left
            """
            azimuth = a.azim
            a.view_init(elev=20., azim=azimuth-10)
            update_canvas()

        def update_canvas() -> None:
            """
            Replots canvas on the GUI with updated points
            """
            a.set_xlim3d(3, 7)
            a.set_ylim3d(6, 10)
            a.set_zlim3d(0,4)
            canvas = FigureCanvasTkAgg(f, self)
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            canvas._tkcanvas.place(relx=0.5, rely=0.45, anchor="center")

        def init():
            a.set_xlim3d(-1, 8)
            a.set_ylim3d(6, 10)
            a.set_zlim3d(0,1)
            a.view_init(elev=20., azim=30)
        
        def update(i):
            plot_results(i)

        def plot_results(frame=0) -> None:
            """
            Plots results for the given skeleton (frame 0)
            """
            pose_dict = {}
            currdir = os.getcwd()
            skel_name = (field_name1.get())
            skelly_dir = os.path.join(currdir, "skeletons", (skel_name + ".pickle"))
            results_dir = os.path.join(currdir, "data", "results", (skel_name + ".pickle"))

            skel_dict = bd.load_skeleton(skelly_dir)
            results = an.load_pickle(results_dir)
            links = skel_dict["links"]
            markers = skel_dict["markers"]

            for i in range(len(markers)):
                pose_dict[markers[i]] = [results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2]]
                a.scatter(results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2])
            
            for link in links:
                if len(link)>1:
                    a.plot3D([pose_dict[link[0]][0], pose_dict[link[1]][0]],
                     [pose_dict[link[0]][1], pose_dict[link[1]][1]],
                    [pose_dict[link[0]][2], pose_dict[link[1]][2]])

            update_canvas()
        
        def next_frame() -> None:
            """
            Plots the next frame of the results
            """
            self.current_frame+=1
            a.clear()
            plot_results(self.current_frame)

        def prev_frame() -> None:
            """
            Plots the previous frame of the results
            """
            self.current_frame-=1
            a.clear()
            plot_results(self.current_frame)
        
        def play_animation() -> None:
            """
            Creates an animation or "slide show" of plotted results in the GUI
            """
            ani = FuncAnimation(f, update, 19, 
                               interval=40, blit=True)
            writer = PillowWriter(fps=25)  
            ani.save("test.gif", writer=writer)

        # --- Define and place GUI components ---

        update_canvas()

        label_name = tk.Label(self, text="Enter skeleton name: ", font=controller.normal_font, background="#ffffff")
        label_name.place(relx=0.4, rely=0.15, anchor = "center")

        field_name1 = tk.Entry(self)
        field_name1.place(relx=0.6, rely=0.15, anchor="center")

        button_next = tk.Button(self, text="Next", command=next_frame)
        button_next.place(relx=0.8, rely=0.3, anchor="center")
        button_prev = tk.Button(self, text="Prev", command=prev_frame)
        button_prev.place(relx=0.8, rely=0.4, anchor="center")

        button_right = tk.Button(self, text="-->", command=rotate_right)
        button_right.place(relx=0.3, rely=0.3, anchor="center")
        button_left = tk.Button(self, text="<--", command=rotate_left)
        button_left.place(relx=0.3, rely=0.4, anchor="center")

        button_anim = tk.Button(self, text="Animation", command=play_animation)
        button_anim.place(relx=0.5, rely=0.8, anchor="center")
    
        label = tk.Label(self, text="Analyse", font=controller.title_font, background="#ffffff")
        label.place(relx=0, rely=0)

        button_plot = tk.Button(self, text="Plot",
                           command=plot_results)
        button_plot.pack()

if __name__ == "__main__":
    app = Application()
    app.geometry("1280x720")
    app.title("Final Year Project")
    app.mainloop()