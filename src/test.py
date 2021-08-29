import pickle

def check_data():
    path = "/Users/user/Documents/Scripts/acinoset_viewer/data/fte_run.pickle"
    path2 = "/Users/user/Documents/Scripts/acinoset_viewer/data/2019_03_09/jules/flick1/3D_GT.pickle"
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    with open(path2, 'rb') as handle:
        data2 = pickle.load(handle)
    print(data2[0])
    print(data["positions"][0])

if __name__=="__main__":
    check_data()