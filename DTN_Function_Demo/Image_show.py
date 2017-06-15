#python 3.0
#coding:UTF-8
'''
@author Yongjun Chen 06/01/2017

'''
#for showing images
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pylab import *

class ShowImage(object):

    def __init__(self):
        self.fig = plt.figure(figsize=(18, 6), frameon=False)
        self.faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                 [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]
        # self.fig.patch.set_alpha(0)

    def T_image_show(self, D, H, W, Image):
        ax = self.fig.add_subplot(111, projection='3d')
        for i in range(D):
            Z = np.linspace(i, i, W)
            for j in range(H):
                Y = np.linspace(j, j, W)
                X = np.linspace(1, W, W)
                color = [str(item/255.) for item in range(W)]
                ax.scatter(X, Y, Z, c=color, s=20)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

    def get_verts(self, xgs, ygs, zgs, N):
        verts = [(xgs[0], ygs[0], zgs[0]),
                 (xgs[(N-1)*N], ygs[(N-1)*N], zgs[(N-1)*N]),
                 (xgs[N*N-1], ygs[N*N-1], zgs[N*N-1]),
                 (xgs[N-1], ygs[N-1], zgs[N-1]),
                 (xgs[N*N*(N-1)], ygs[N*N*(N-1)], zgs[N*N*(N-1)]),
                 (xgs[(N+1)*N*(N-1)], ygs[(N+1)*N*(N-1)], zgs[(N+1)*N*(N-1)]),
                 (xgs[N*N*N-1], ygs[N*N*N-1], zgs[N*N*N-1]),
                 (xgs[(N*N+1)*(N-1)], ygs[(N*N+1)*(N-1)], zgs[(N*N+1)*(N-1)]), ]
        return verts
	
    def show_signle_img(self,location,x,y,z,sizex,sizey,sizez,nameofT,namefimg,N,colors): 
        self.ax = self.fig.add_subplot(location, projection='3d')
        if colors == 'default':
            color = [str(item) for item in range(len(x))]
            print("color type",type(color))
            print("color shape", len(color))
            print("color",color[57:67])
            #print("color value",color)
        else:
            color = colors
            print("color type",type(color))
            print("color shape", len(color))
            print("color",color[57:67])
            #print("color value",color)

        self.ax.scatter(x, y, z, marker='.', c=color, s=200, depthshade=True)
        verts = self.get_verts(x, y, z, N)
        poly3d = [[verts[vert_id] for vert_id in face] for face in self.faces]
        #self.ax.add_collection3d(Poly3DCollection(
	    #    poly3d, facecolors='w', linewidths=1, alpha=0.3))
        self.ax.set_title(namefimg + nameofT)
        self.ax.set_xlim3d(-sizex*1.3, sizex*1.3)
        self.ax.set_ylim3d(-sizey*1.3, sizey*1.3)
        self.ax.set_zlim3d(-sizez*1.3, sizez*1.3)


    def Show_transform(self, xgs, ygs, zgs, xgt, ygt, zgt, sizex,sizey,sizez,stop_time, nameofT, N,colors):
        # before affine
        self.show_signle_img(131, xgt, ygt, zgt,sizex,sizey,sizez,nameofT,' Encoder Input ',N,colors)
        # after affine before DTN
        self.show_signle_img(132, xgs, ygs, zgs,sizex,sizey,sizez,nameofT,'Encoder output ',N,colors)
        # after DTN
        self.show_signle_img(133,xgt, ygt, zgt,sizex,sizey,sizez,nameofT,'Decoder output ',N,colors)
        # pause time
        #gray()
        #plt.show()
        plt.pause(stop_time)
