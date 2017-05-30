import pdb
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Image_show import *
import random
import h5py
from pylab import *

class Transformation_Tests(object):
    def Affine_test(self,N,sizex,sizey,sizez,times,stop_time,typeofT,colors):
        for i in range(times):
            # Theta
            idx = np.random.uniform(-1, 1);idy = np.random.uniform(-1, 1);idz = np.random.uniform(-1, 1)
            swithx = np.random.uniform(0,1);swithy = np.random.uniform(0,1);swithz = np.random.uniform(0,1)
            rotatex = np.random.uniform(-1, 1);rotatey = np.random.uniform(-1, 1);rotatez = np.random.uniform(-1, 1)
            cx = np.array([idx,rotatey,rotatez,swithx]);cy = np.array([rotatex,idy,rotatez,swithy]);cz = np.array([rotatex,rotatey,idz,swithz])
            # Source Grid
            x = np.linspace(-sizex, sizex, N);y = np.linspace(-sizey, sizey, N);z = np.linspace(-sizez, sizez, N)
            x, y, z = np.meshgrid(x, y, z)
            xgs, ygs, zgs = x.flatten(), y.flatten(),z.flatten()
            gps = np.vstack([xgs, ygs, zgs, np.ones_like(xgs)]).T
            # transform
            xgt = np.dot(gps, cx);ygt = np.dot(gps, cy);zgt = np.dot(gps, cz)
            # display
            showIm = ShowImage()
            showIm.Show_transform(xgs,ygs,zgs,xgt,ygt,zgt,sizex,sizey,sizez,stop_time,typeofT,N,colors)

    def makeT(self,cp):
        # cp: [(k*k*k) x 3] control points
        # T: [((k*k*k)+4) x ((k*k*k)+4)]
        K = cp.shape[0]
        T = np.zeros((K+4, K+4))
        T[:K, 0] = 1; T[:K, 1:4] = cp; T[K, 4:] = 1; T[K+1:, 4:] = cp.T
        R = squareform(pdist(cp, metric='euclidean'))
        R = R * R;R[R == 0] = 1 # a trick to make R ln(R) 0
        R = R * np.log(R)
        np.fill_diagonal(R, 0)
        T[:K, 4:] = R
        return T

    def liftPts(self,p, cp):
        # p: [N x 3], input points
        # cp: [K x 3], control points
        # pLift: [N x (4+K)], lifted input points
        N, K = p.shape[0], cp.shape[0]
        pLift = np.zeros((N, K+4));pLift[:,0] = 1; pLift[:,1:4] = p
        R = cdist(p, cp, 'euclidean')
        R = R * R
        R[R == 0] = 1
        R = R * np.log(R)
        pLift[:,4:] = R
        return pLift

    def TPS_test(self,N,sizex,sizey,sizez,control_num,show_times,stop_time,typeofT,colors):
        for i in range(show_times):
            # source control points
            x, y, z = np.meshgrid(np.linspace(-1, 1, control_num),np.linspace(-1, 1, control_num), np.linspace(-1, 1, control_num))
            xs = x.flatten();ys = y.flatten();zs = z.flatten()
            cps = np.vstack([xs, ys, zs]).T
            # target control points
            xt = xs + np.random.uniform(-0.3, 0.3, size=xs.size);yt = ys + np.random.uniform(-0.3, 0.3, size=ys.size); zt = zs + np.random.uniform(-0.3, 0.3, size=zs.size)
            # construct T
            T = self.makeT(cps)
            # solve cx, cy (coefficients for x and y)
            xtAug = np.concatenate([xt, np.zeros(4)]);ytAug = np.concatenate([yt, np.zeros(4)]);ztAug = np.concatenate([zt, np.zeros(4)])
            cx = nl.solve(T, xtAug); cy = nl.solve(T, ytAug); cz = nl.solve(T, ztAug)
            # dense grid
            x = np.linspace(-sizex, sizex, 2*sizex); y = np.linspace(-sizey, sizey, 2*sizey); z = np.linspace(-sizez, sizez, 2*sizez)
            x, y, z = np.meshgrid(x, y, z)
            xgs, ygs, zgs = x.flatten(), y.flatten(), z.flatten()
            gps = np.vstack([xgs, ygs, zgs]).T
            # transform
            pgLift = self.liftPts(gps, cps) # [N x (K+3)]
            xgt = np.dot(pgLift, cx.T);ygt = np.dot(pgLift, cy.T); zgt = np.dot(pgLift,cz.T)
            # display
            showIm = ShowImage()
            showIm.Show_transform(xgs,ygs,zgs,xgt,ygt,zgt,sizex,sizey,sizez,stop_time,typeofT,N,colors)

    def show_dataset(self,data,nums):
        for i in range(nums):
            fig = plt.figure()
            ax = fig.add_subplot(221)
            ax.imshow(data[:,:,i])
            plt.pause(1)

if __name__ == "__main__":
    #data
    test = Transformation_Tests()  
	#N,sizex,sizey,sizez,control_num,show_times,stop_time,typeofT,colors)
    #test.Affine_test(10,5,5,2,10,2,'Affine','default')
    test.TPS_test(40, 20, 20, 20, 10, 10, 1, 'TPS','default')



