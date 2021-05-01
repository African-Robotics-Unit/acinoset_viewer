import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog, ttk
import os
import glob
from calib import calib, app, extract, utils, plotting
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np
import cv2
from matplotlib.animation import FuncAnimation, PillowWriter, Animation

class Application(tk.Tk):

    def __init__(self, *args, **kwargs):
        """
        Initialise the main application
        """

        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Montserrat', size=18)
        self.normal_font = tkfont.Font(family='Montserrat', size=12)
        self.project_dir = "No video folder selected"
        self.sba_dir = "No SBA file selected"

        #x = tk.Tk()
        #W = x.winfo_screenwidth()*.8
        #H = x.winfo_screenheight()*.8

        container = tk.Frame(self, width=1130, height=720)
        nav_bar = tk.Frame(self, width=150, height=720, background="#2c3e50")

        #container = tk.Frame(self, width=W, height=H)
        #nav_bar = tk.Frame(self, width=W*.1, height=H, background="#2c3e50")
        
        container.pack_propagate(False)
        nav_bar.pack_propagate(False)
        nav_bar.place(relx=0, rely=0)
        container.place(x=150,y=0)

        self.home_label = tk.Label(nav_bar, text="Load", font=self.title_font, bg="#2c3e50", fg="#ffffff", width=150, cursor="hand2")
        self.home_label.place(relx=0.5, y=30, anchor="center")
        self.build_label = tk.Label(nav_bar, text="Create", font=self.title_font, bg="#2c3e50", fg="#ffffff", width=150, cursor="hand2")
        self.build_label.place(relx=0.5, y=80, anchor="center")
        self.analyse_label = tk.Label(nav_bar, text="Analyze", font=self.title_font, bg="#2c3e50", fg="#ffffff", width=150, cursor="hand2")
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
        #combo_vids = ttk.Combobox(self, values=["- Select Video -"])

        # --- Define functions to be used by GUI components ---

        def choose_folder():
            currdir = os.getcwd()
            controller.project_dir = filedialog.askdirectory(parent=self, initialdir=currdir, title='Please Select a Video Folder:')
            if len(controller.project_dir) > 0:
                print("You chose %s" % controller.project_dir)
                label_folder.configure(text=os.path.normpath(controller.project_dir))
            dirs = glob.glob(os.path.join(controller.project_dir, "*.mp4"))
            print(dirs)
            n_vids = len(dirs)
            controller.dirs = dirs

            try:

                sba_folder = controller.project_dir.split("data")[1]
                sba_folder = os.path.normpath(sba_folder)
                sba_filepath = sba_folder.split(os.sep)[1]
                print(sba_filepath)
                full_sba = os.path.join(controller.project_dir.split(sba_filepath)[0], sba_filepath, "extrinsic_calib", str(n_vids)+"_cam_scene_sba.json")
                print(f'Detected SBA file is {full_sba}')

                if os.path.exists(os.path.normpath(full_sba)):
                    controller.sba_dir = os.path.normpath(full_sba)
                    label_sba.configure(text=controller.sba_dir)
                else:
                    print("The associated SBA file could not be located!")
            except:
                print("That folder does not appear to be a valid video folder!")
            #video_names = get_video_names(dirs)

            #label_vids = tk.Label(self, text="Choose a video:", font=controller.normal_font, bg="#ffffff")
            #label_vids.place(relx=0.5, rely=0.6, anchor="center")

            #combo_vids.configure(values=video_names)
            #combo_vids.place(relx=0.5,rely=0.65, anchor = "center")

            #combo_vids.bind("<<ComboboxSelected>>", set_vid)

        #def set_vid(self):
            #controller.vid = combo_vids.get()
            #print(controller.vid)

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
        button1 = tk.Button(self, text="Choose Video Folder", command=choose_folder)
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

        f_2d_left = Figure(figsize=(4,7), dpi=100)
        a_2d_1 = f_2d_left.add_subplot(311)
        a_2d_2 = f_2d_left.add_subplot(312)
        a_2d_3 = f_2d_left.add_subplot(313)

        f_2d_right = Figure(figsize=(4,7), dpi=100)
        a_2d_4 = f_2d_right.add_subplot(311)
        a_2d_5 = f_2d_right.add_subplot(312)
        a_2d_6 = f_2d_right.add_subplot(313)

        axes_list = [a_2d_1, a_2d_2, a_2d_3, a_2d_4, a_2d_5, a_2d_6]

        self.frame = 0
        x_free = tk.IntVar()
        y_free = tk.IntVar()
        z_free = tk.IntVar()

        self.markers = ["l_eye", "r_eye", "nose", "neck_base", "spine",
            "tail_base", "tail1", "tail2",
            "l_shoulder", "l_front_knee","l_front_ankle",
            "r_shoulder", "r_front_knee", "r_front_ankle",
            "l_hip", "l_back_knee", "l_back_ankle",
            "r_hip", "r_back_knee", "r_back_ankle"]

        self.parts_dict = {}
        self.points_dict={}
        self.changes_dict = {}

        self.traj_data={}
        self.points_2d = {}
        self.original_pos = {}
        self.vid_arr = []

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
            canvas_2d_1._tkcanvas.place(relx=0.18, rely=0.45, anchor="center")

            canvas_2d_2 = FigureCanvasTkAgg(f_2d_right, self)
            canvas_2d_2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            canvas_2d_2._tkcanvas.place(relx=0.82, rely=0.45, anchor="center")

        def load_gt_data() -> None:
            """
            Loads the GT points from a given folder
            """
            data_dir = os.path.join(controller.project_dir, "fte", "fte.pickle")
            self.traj_data = load_pickle(data_dir)
            self.original_pos = load_pickle(data_dir)
            sba_dir = controller.sba_dir
            dirs = controller.dirs
            for cam in range(1,7):
                vid_dir = os.path.join(controller.project_dir, "cam"+str(cam)+".mp4")
                cap = cv2.VideoCapture(vid_dir)
                self.vid_arr.append(cap)

            markers = self.markers

            #print(self.traj_data["positions"])
            for i,frame in enumerate(self.traj_data["positions"]):
                if not np.isnan(frame[0][0]):
                    self.frame=i
                    break
            
            print(len(self.traj_data["positions"]))

            print(self.frame)
            label_frame.configure(text=self.frame)
            plot_cheetah(self.frame)

        def load_pickle(pickle_file):
            """
            Loads a dictionary from a saved skeleton .pickle file
            """
            with open(pickle_file, 'rb') as handle:
                data = pickle.load(handle)

            return(data)

        def plot_cheetah(frame) -> None:

            self.parts_dict = {}
            self.points_dict = {}
            a.clear()
            a.set_xlabel('X')
            a.set_ylabel('Y')
            a.set_zlabel('Z')
            label_frame.configure(text=frame)

            markers = self.markers

            K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(controller.sba_dir)
            D_arr = D_arr.reshape((-1,4))

            links = [[0,2], [1,2], [2,3], [0,3], [1,3], [3,4], [4,5], [5,6], [6,7], [3,8], [8,9], [9,10], [3,11], [11,12], [12,13],
             [5, 14], [14,15], [15,16], [5,17], [17,18], [18,19]]

            pts = self.traj_data["positions"][self.frame]

            for i in range(len(markers)):
                self.parts_dict[markers[i]] = [pts[i][0], pts[i][1], pts[i][2]]
                self.points_dict[markers[i]] = a.scatter(pts[i][0], pts[i][1], pts[i][2])

            for link in links:
                part1 = markers[link[0]]
                part2 = markers[link[1]]
                a.plot3D([self.parts_dict[part1][0], self.parts_dict[part2][0]],
                 [self.parts_dict[part1][1], self.parts_dict[part2][1]],
                 [self.parts_dict[part1][2], self.parts_dict[part2][2]], 'b')

            for i, axis in enumerate(axes_list):
                axis.clear()
                cam=i+1

                vid_dir = os.path.join(controller.project_dir, "cam"+str(cam)+".mp4")
                cap = self.vid_arr[i]
                cam_frame = self.frame + self.traj_data["start_frame"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, cam_frame)
                ret, image = cap.read()
                RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axis.imshow(RGB_img)
                pts_2d_cam = calib.project_points_fisheye(pts, K_arr[i],D_arr[i],R_arr[i],t_arr[i])

                width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                #print(f'Video is {width} pixels by {height} pixels!')

                x_s = []
                y_s = []

                for point in pts_2d_cam:
                    if not np.isnan(point[0]):
                        axis.scatter(point[0], point[1], s=10)
                        x_s.append(point[0])
                        y_s.append(point[1])
                self.points_2d[self.frame,cam] = pts_2d_cam
                minx = np.min(x_s)*0.9
                miny = np.min(y_s)*0.9
                maxx = np.max(x_s)*1.1
                maxy = np.max(y_s)*1.1
                if maxx > width:
                    maxx = width
                if minx < 0:
                    minx = 0
                if maxy > height:
                    maxy = height
                if miny < 0:
                    maxy = 0
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
            markers = self.markers

            new_x = float(x_spin.get())
            new_y = float(y_spin.get())
            new_z = float(z_spin.get())

            index = markers.index(part_to_move)

            point_to_move = self.points_dict[part_to_move]
            point_to_move.remove()

            old_x = self.original_pos["positions"][frame][index][0]
            old_y = self.original_pos["positions"][frame][index][1]
            old_z = self.original_pos["positions"][frame][index][2]

            print([old_x, old_y, old_z])

            self.traj_data["positions"][frame][index] = [new_x, new_y, new_z]

            self.changes_dict[part_to_move] = [new_x-old_x, new_y-old_y, new_z-old_z]
            print(self.changes_dict[part_to_move])

            self.points_dict[part_to_move] = a.scatter(new_x, new_y, new_z)
            self.parts_dict[part_to_move] = [new_x, new_y, new_z]

            plot_cheetah(self.frame)

        def save_labels() -> None:
            """
            Writes the gt frames to a pickle file
            """
            project_dir = controller.project_dir
            results_file3d = os.path.join(project_dir, "3D_GT.pickle")
            file_data3d = self.traj_data["positions"]
            results_file2d = os.path.join(project_dir, "2D_reprojected_GT.pickle")
            file_data2d = self.points_2d

            with open(results_file3d, 'wb') as f:
                pickle.dump(file_data3d, f)

            with open(results_file2d, 'wb') as f:
                pickle.dump(file_data2d, f)

            print(f'Saved {results_file3d}')
            print(f'Saved {results_file2d}')

        def next_frame() -> None:
            """
            Plots the next frame of the results
            """
            self.frame+=1
            a.clear()
            for axis in axes_list:
                axis.clear()
            plot_cheetah(self.frame)

        def prev_frame() -> None:
            """
            Plots the previous frame of the results
            """
            self.frame-=1
            a.clear()
            for axis in axes_list:
                axis.clear()
            plot_cheetah(self.frame)

        def goto_frame() -> None:
            frame_to_go = frame_spin.get()
            self.frame = int(frame_to_go)
            a.clear()
            for axis in axes_list:
                axis.clear()
            plot_cheetah(self.frame)

        def update_spins(event) -> None:
            part = combo_move.get()
            x_free.set(self.parts_dict[part][0])
            y_free.set(self.parts_dict[part][1])
            z_free.set(self.parts_dict[part][2])

        # --- Define and place GUI components ---

        update_canvas()

        combo_move = ttk.Combobox(self, values=["Empty"])
        combo_move.place(relx=0.5,rely=0.6, anchor = "center")
        combo_move.bind("<<ComboboxSelected>>", update_spins)

        frame_spin = tk.Spinbox(self, from_=0, to=500, increment=1, format = f"%.0f")
        frame_spin.place(relx=0.2, rely=0.95, anchor="center")

        button_go = tk.Button(self, text="Go", command=goto_frame)
        button_go.place(relx=0.27, rely=0.95, anchor="center")

        x_spin = tk.Spinbox(self, from_=-10, to=10, increment=0.01, textvariable=x_free, format = f"%.2f")
        x_spin.place(relx=0.5, rely=0.65, anchor="center")
        y_spin = tk.Spinbox(self, from_=-10, to=10, increment=0.01, textvariable=y_free, format = f"%.2f")
        y_spin.place(relx=0.5, rely=0.7, anchor="center")
        z_spin = tk.Spinbox(self, from_=-10, to=10, increment=0.01, textvariable=z_free, format = f"%.2f")
        z_spin.place(relx=0.5, rely=0.75, anchor="center")

        label_x = tk.Label(self, text="x: ", font=controller.normal_font, background="#ffffff")
        label_x.place(relx=0.43, rely=0.65, anchor="center")
        label_y = tk.Label(self, text="y: ", font=controller.normal_font, background="#ffffff")
        label_y.place(relx=0.43, rely=0.7, anchor="center")
        label_z = tk.Label(self, text="z: ", font=controller.normal_font, background="#ffffff")
        label_z.place(relx=0.43, rely=0.75, anchor="center")

        button_update = tk.Button(self, text="Move", command=move_point)
        button_update.place(relx=0.5, rely=0.8, anchor="center")

        button = tk.Button(self, text="Load Data", command=load_gt_data)
        button.place(relx=0.45, rely=0.95, anchor="center")

        button_right = tk.Button(self, text="-->", command=rotate_right)
        button_right.place(relx=0.55, rely=0.5, anchor="center")
        button_left = tk.Button(self, text="<--", command=rotate_left)
        button_left.place(relx=0.45, rely=0.5, anchor="center")

        button_save = tk.Button(self, text="Save Data", command=save_labels)
        button_save.place(relx=0.55, rely=0.95, anchor="center")

        button_next = tk.Button(self, text="Next", command=next_frame)
        button_next.place(relx=0.57, rely=0.9, anchor="center")

        button_prev = tk.Button(self, text="Prev", command=prev_frame)
        button_prev.place(relx=0.43, rely=0.9, anchor="center")

        label_frame = tk.Label(self, text=self.frame, font=controller.normal_font, background="#ffffff")
        label_frame.place(relx=0.5, rely=0.9, anchor="center")

class PageTwo(tk.Frame):

    # --- Residual Code from another version, unimportant ---

    def __init__(self, parent, controller):
        """
        Initialise a frame for the analyse page
        """
        tk.Frame.__init__(self, parent, height=720, width=1130, background="#ffffff")
        self.controller = controller
        self.pack_propagate(False)

        # --- Initialise class-wide variables ---

        f_2d_left = Figure(figsize=(4,7), dpi=100)
        a_2d_1 = f_2d_left.add_subplot(311)
        a_2d_2 = f_2d_left.add_subplot(312)
        a_2d_3 = f_2d_left.add_subplot(313)

        f_2d_right = Figure(figsize=(4,7), dpi=100)
        a_2d_4 = f_2d_right.add_subplot(311)
        a_2d_5 = f_2d_right.add_subplot(312)
        a_2d_6 = f_2d_right.add_subplot(313)

        axes_list = [a_2d_1, a_2d_2, a_2d_3, a_2d_4, a_2d_5, a_2d_6]

        self.frame = 1
        
        self.changes_dict = {}

        self.vid_arr = []

        # --- Define functions to be used by GUI components ---

        def update_2d_views() -> None:
            canvas_2d_1 = FigureCanvasTkAgg(f_2d_left, self)
            canvas_2d_1.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            canvas_2d_1._tkcanvas.place(relx=0.18, rely=0.45, anchor="center")

            canvas_2d_2 = FigureCanvasTkAgg(f_2d_right, self)
            canvas_2d_2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            canvas_2d_2._tkcanvas.place(relx=0.82, rely=0.45, anchor="center")

        def load_gt_data() -> None:
            """
            Loads the GT points from a given folder
            """
            for cam in range(1,7):
                vid_dir = os.path.join(controller.project_dir, "cam"+str(cam)+".mp4")
                cap = cv2.VideoCapture(vid_dir)
                self.vid_arr.append(cap)

            print(self.frame)
            label_frame.configure(text=self.frame)
            plot_cheetah(self.frame)

        def plot_cheetah(frame) -> None:

            self.parts_dict = {}
            self.points_dict = {}
            label_frame.configure(text=frame)

            for i, axis in enumerate(axes_list):
                axis.clear()
                cam=i+1
                #print("CAM"+str(cam))
                vid_dir = os.path.join(controller.project_dir, "cam"+str(cam)+".mp4")
                cap = self.vid_arr[i]
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
                ret, image = cap.read()
                if ret:
                    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    axis.imshow(RGB_img)
                else:
                    print("That frame doesn't exist!")
            
            update_2d_views()

        def next_frame() -> None:
            """
            Plots the next frame of the results
            """
            self.frame+=1
            for axis in axes_list:
                axis.clear()
            plot_cheetah(self.frame)

        def prev_frame() -> None:
            """
            Plots the previous frame of the results
            """
            self.frame-=1
            for axis in axes_list:
                axis.clear()
            plot_cheetah(self.frame)

        def goto_frame() -> None:
            frame_to_go = frame_spin.get()
            self.frame = int(frame_to_go)
            for axis in axes_list:
                axis.clear()
            plot_cheetah(self.frame)

        # --- Define and place GUI components ---

        frame_spin = tk.Spinbox(self, from_=0, to=500, increment=1, format = f"%.0f")
        frame_spin.place(relx=0.2, rely=0.95, anchor="center")

        button_go = tk.Button(self, text="Go", command=goto_frame)
        button_go.place(relx=0.27, rely=0.95, anchor="center")

        button = tk.Button(self, text="Load Data", command=load_gt_data)
        button.place(relx=0.5, rely=0.95, anchor="center")

        button_next = tk.Button(self, text="Next", command=next_frame)
        button_next.place(relx=0.57, rely=0.9, anchor="center")

        button_prev = tk.Button(self, text="Prev", command=prev_frame)
        button_prev.place(relx=0.43, rely=0.9, anchor="center")

        label_frame = tk.Label(self, text=self.frame, font=controller.normal_font, background="#ffffff")
        label_frame.place(relx=0.5, rely=0.9, anchor="center")


if __name__ == "__main__":
    app = Application()
    app.geometry("1280x720")
    app.title("AcinoNet Viewer")
    app.mainloop()
