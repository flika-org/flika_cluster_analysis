import flika
import sys
import numpy as np
sys.path.append('C:/Users/Kevin/.FLIKA/plugins/cluster_analysis')
from DBScan.Main import *

flika.version = flika.__version__
from flika import global_vars as g
from flika.process.BaseProcess import BaseProcess_noPriorWindow, ComboBox, CheckBox
from flika.process.file_ import open_file_gui
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qtpy.QtWidgets import *
from qtpy.QtCore import *
from matplotlib.widgets import Button

CURRENT_STATE = 'All Points'
CLUSTERS = []

def get_text_file(filename=None):
    if filename is None:
            filetypes = '*.txt'
            prompt = 'Open File'
            filename = open_file_gui(prompt, filetypes=filetypes)
            if filename is None:
                return None
    else:
        filename = g.settings['filename']
        if filename is None:
            g.alert('No filename selected')
            return None
    print("Filename: {}".format(filename))
    g.m.statusBar().showMessage('Loading {}'.format(os.path.basename(filename)))
    return filename


def axis_creator(*args, choices):
    '''Creates the choices for each axis'''
    for item in args:
        for choice in choices:
            item.addItem(choice)


def get_coordinates(x_index, y_index, z_index, text):
    red = text[0].split('\t')[0]
    redx, redy, redz, greenx, greeny, greenz = [], [], [], [], [], []
    for line in text:
        if line.split()[0] == red:
            redx.append(float(line.split()[x_index]))
            redy.append(float(line.split()[y_index]))
            redz.append(float(line.split()[z_index]))
        else:
            greenx.append(float(line.split()[x_index]))
            greeny.append(float(line.split()[y_index]))
            greenz.append(float(line.split()[z_index]))
    return redx,redy,redz,greenx,greeny,greenz


def get_merged_coords(x,y,z):
    merged_coords=[]
    for i in range(len(x)):
        merged_coords.append([x[i], y[i], z[i]])
    return merged_coords


def get_scan_data(x_index, y_index, text):
    x = [line.split('\t')[x_index] for line in text]
    y = [line.split('\t')[y_index] for line in text]
    return np.vstack([x,y]).T


def get_clusters(x_axis_selection, y_axis_selection, z_axis_selection, keys, text):
    global CLUSTERS
    points = get_scan_data(keys.index(x_axis_selection), keys.index(y_axis_selection), text[1:])
    clusts = scan(points, g.settings['epsilon'], g.settings['min_neighbors'], g.settings['min_density'])
    CLUSTERS = []
    for i in range(len(clusts)):
        CLUSTERS.append(Cluster(np.array(clusts[i]).astype(np.float)))
    rx, ry, rz, gx, gy, gz = get_coordinates(keys.index(x_axis_selection),
                                             keys.index(y_axis_selection),
                                             keys.index(z_axis_selection), text[1:])

    merged_coords_r = get_merged_coords(rx, ry, rz)
    merged_coords_g = get_merged_coords(gx, gy, gz)

    for clust in CLUSTERS:
        clust.set_z_points(merged_coords_r, 'red')
        clust.set_z_points(merged_coords_g, 'green')
    save(CLUSTERS)
    return rx,ry,rz,gx,gy,gz

class ClusterAnalysis(BaseProcess_noPriorWindow):

    def __init__(self):
        super().__init__()
        self.__name__ = self.__class__.__name__

    def get_init_settings_dict(self):
        s = dict()
        s['x_axis_selection'] = 'X'
        s['y_axis_selection'] = 'Y'
        s['z_axis_selection'] = 'Z'
        return s

    def gui(self):
        text_file = get_text_file()
        self.text = open(text_file, 'r').readlines()
        self.keys = self.text[0].split('\t')
        self.gui_reset()
        x_axis_selection = ComboBox()
        y_axis_selection = ComboBox()
        z_axis_selection = ComboBox()
        epsilon = QDoubleSpinBox()
        epsilon.setDecimals(2)
        epsilon.setSingleStep(1)
        min_neighbors = QDoubleSpinBox()
        min_neighbors.setDecimals(2)
        min_neighbors.setSingleStep(1)
        min_cluster = QDoubleSpinBox()
        min_cluster.setDecimals(2)
        min_cluster.setSingleStep(1)
        simulate_check = CheckBox()
        axis_creator(x_axis_selection, y_axis_selection, z_axis_selection, choices=self.keys)
        self.items.append({'name': 'x_axis_selection', 'string': 'x axis', 'object': x_axis_selection})
        self.items.append({'name': 'y_axis_selection', 'string': 'y axis', 'object': y_axis_selection})
        self.items.append({'name': 'z_axis_selection', 'string': 'z axis', 'object': z_axis_selection})
        self.items.append({'name': 'epsilon', 'string': 'Epsilon', 'object': epsilon})
        self.items.append({'name': 'min_neighbors', 'string': 'Minimum neighbors to consider a point',
                           'object': min_neighbors})
        self.items.append({'name': 'min_cluster', 'string': 'Minimum Cluster Density', 'object': min_cluster})
        self.items.append({'name': 'simulate_check', 'string': 'Simulate Center Proximities', 'object': simulate_check})
        super().gui()
        self.ui.setGeometry(QRect(150, 50, 150, 130))

    def __call__(self, x_axis_selection, y_axis_selection, z_axis_selection, epsilon=30, min_neighbors=4, min_cluster=5,
                 simulate_check=True):
        global CLUSTERS
        self.start()
        g.settings['epsilon'] = epsilon
        g.settings['min_neighbors'] = min_neighbors
        g.settings['min_density'] = min_cluster
        g.settings['simulate_check'] = simulate_check
        rx,ry,rz,gx,gy,gz = get_clusters(x_axis_selection, y_axis_selection, z_axis_selection, self.keys, self.text)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def show_centroids(event):
            global CURRENT_STATE, CLUSTERS
            CURRENT_STATE = 'Centroids'
            cxr = []
            cyr = []
            czr = []
            cxg = []
            cyg = []
            czg = []
            ax.clear()
            for clust in CLUSTERS:
                if clust.color == 'red':
                    cxr.append(clust.center[0])
                    cyr.append(clust.center[1])
                    czr.append(clust.ZCenter)
                elif clust.color == 'green':
                    cxg.append(clust.center[0])
                    cyg.append(clust.center[1])
                    czg.append(clust.ZCenter)

            ax.scatter(xs=cxr, ys=cyr, zs=czr, color='r', s=20, marker='o')
            ax.scatter(xs=cxg, ys=cyg, zs=czg, color='g', s=20, marker='^')

        def show_clusters(event):
            global CURRENT_STATE, CLUSTERS
            CURRENT_STATE = 'Cluster Points'
            ax.clear()
            cxr = []
            cyr = []
            czr = []
            cxg = []
            cyg = []
            czg = []
            for clust in CLUSTERS:
                if clust.color == 'red':
                    for i in range(len(clust.points)):
                        cxr.append(clust.points[i][0])
                        cyr.append(clust.points[i][1])
                        czr.append(clust.zs[i])
                if clust.color == 'green':
                    for i in range(len(clust.points)):
                        cxg.append(clust.points[i][0])
                        cyg.append(clust.points[i][1])
                        czg.append(clust.zs[i])

            ax.scatter(xs=cxr, ys=cyr, zs=czr, color='r', s=20, marker='o')
            ax.scatter(xs=cxg, ys=cyg, zs=czg, color='g', s=20, marker='^')


        def show_all_points(event):
            global CURRENT_STATE
            CURRENT_STATE = 'All Points'
            ax.clear()
            ax.scatter(xs=rx, ys=ry, zs=rz, c='r', marker='o')
            ax.scatter(xs=gx, ys=gy, zs=gz, c='g', marker='^')

        def hide_red(event):
            global CURRENT_STATE, CLUSTERS
            ax.clear()
            if CURRENT_STATE=='All Points':
                ax.scatter(xs=gx,ys=gy,zs=gz, c='g',marker='^')
            elif CURRENT_STATE=='Centroids':
                cx,cy,cz = [],[],[]
                for clust in CLUSTERS:
                    if clust.color == 'green':
                        cx.append(clust.center[0])
                        cy.append(clust.center[1])
                        cz.append(clust.ZCenter)
                ax.scatter(xs=cx,ys=cy,zs=cz,c='g',marker='^')
            elif CURRENT_STATE=='Cluster Points':
                cx, cy, cz = [], [], []
                for clust in CLUSTERS:
                    if clust.color == 'green':
                        for i in range(len(clust.points)):
                            cx.append(clust.points[i][0])
                            cy.append(clust.points[i][1])
                            cz.append(clust.zs[i])
                ax.scatter(xs=cx, ys=cy, zs=cz, c='g', marker='^')

        def hide_green(event):
            global CURRENT_STATE, CLUSTERS
            ax.clear()
            if CURRENT_STATE == 'All Points':
                ax.scatter(xs=rx, ys=ry, zs=rz, c='r', marker='o')
            elif CURRENT_STATE == 'Centroids':
                cx, cy, cz = [], [], []
                for clust in CLUSTERS:
                    if clust.color == 'red':
                        cx.append(clust.center[0])
                        cy.append(clust.center[1])
                        cz.append(clust.ZCenter)
                ax.scatter(xs=cx, ys=cy, zs=cz, c='r', marker='o')
            elif CURRENT_STATE=='Cluster Points':
                cx, cy, cz = [], [], []
                for clust in CLUSTERS:
                    if clust.color == 'red':
                        for i in range(len(clust.points)):
                            cx.append(clust.points[i][0])
                            cy.append(clust.points[i][1])
                            cz.append(clust.zs[i])
                ax.scatter(xs=cx, ys=cy, zs=cz, c='r', marker='o')

        def update_display():
            global CLUSTERS
            g.settings.update(epsilon=epsilon_spin.value(), min_neighbors=min_neighbors_spin.value(),
                              min_density=min_density_spin.value())
            get_clusters(x_axis_selection, y_axis_selection, z_axis_selection, self.keys,
                                                  self.text)

        ax.scatter(xs=rx, ys=ry, zs=rz, c='r', marker='o')
        ax.scatter(xs=gx, ys=gy, zs=gz, c='g', marker='^')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        axcutC = plt.axes([0.32, 0.9, 0.15, 0.075])
        axcutClust = plt.axes([0.16, 0.9, 0.15, 0.075])
        axcutP = plt.axes([0.0, 0.9, 0.15, 0.075])
        axcutR = plt.axes([0.0, 0.0, 0.15, 0.075])
        axcutG = plt.axes([0.0, 0.1, 0.15, 0.075])
        centroid_button = Button(axcutC, 'Centroids', color='white', hovercolor='green')
        cluster_button = Button(axcutClust, 'Clusters', color='white', hovercolor='green')
        point_button = Button(axcutP, 'All Points', color='white', hovercolor='green')
        red_button = Button(axcutR, 'Hide Red', color='white', hovercolor='green')
        green_button = Button(axcutG, 'Hide Green', color='white', hovercolor='green')
        centroid_button.on_clicked(show_centroids)
        cluster_button.on_clicked(show_clusters)
        point_button.on_clicked(show_all_points)
        red_button.on_clicked(hide_red)
        green_button.on_clicked(hide_green)


        win = QMainWindow()
        widg = QWidget()
        win.setCentralWidget(widg)
        win.setAcceptDrops(True)
        ly = QFormLayout(widg)
        epsilon_spin = pg.SpinBox(value=g.settings['epsilon'])
        min_neighbors_spin = pg.SpinBox(value=g.settings['min_neighbors'], int=True, step=1)
        min_density_spin = pg.SpinBox(value=g.settings['min_density'], int=True, step=1)
        simulateCheck = QCheckBox("Simulate Center Proximities")
        clusterButton = QPushButton("Update")
        clusterButton.pressed.connect(update_display)

        ly.addRow("Epsilon", epsilon_spin)
        ly.addRow("Minimum neighbors to consider a point", min_neighbors_spin)
        ly.addRow("Minimum Cluster Density", min_density_spin)
        ly.addWidget(simulateCheck)
        ly.addWidget(clusterButton)

        win.installEventFilter(mainWindowEventEater)
        win.setWindowTitle("DBScan Clustering")
        win.closeEvent = close_and_save
        win.show()
        QApplication.instance().exec_()

        plt.show()

ClusterAnalysis = ClusterAnalysis()