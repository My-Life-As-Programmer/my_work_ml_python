import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class sup_vtr_mch:
    def __init__(self, vis=True):
        self.vis = vis
        self.colors = {1:'r', -1:'b'}
        if self.vis:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    #training
    def fit(self, data):
        self.data = data
        # {  ||w||: [w,b]}
        opt_dict = {}

        transforms = [[1,1],
                      [1,-1],
                      [-1,1],
                      [-1,-1]]

        all_data = []
        for yi in self.data:
            for featset in self.data[yi]:
                for feat in featset:
                    all_data.append(feat)

        self.max_feat_val = max(all_data)
        self.min_feat_val = min(all_data)
        all_data = None # to remove data from memory

        step_sizes = [self.max_feat_val * 0.1,
                      self.max_feat_val * 0.01,
                      # expensive
                      self.max_feat_val * 0.001,]
        # extreemly expensive
        b_range_multiple = 5
        #
        b_multiple = 5

        latest_optimum = self.max_feat_val*10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            optimized = False

            while not optimized:
                for b in np.arange(-1*(self.max_feat_val*b_range_multiple), self.max_feat_val*b_range_multiple, step*b_multiple):
                    for trans in transforms:
                        w_t = w*trans
                        found_opt = True

                        # weakest link in SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi>w+b) >= 1

                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not (yi*(np.dot(w_t,xi)+b)) >= 1:
                                    found_opt = False
                                    
                        if found_opt:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0]< 0:
                    optimized= True
                    print(' Optimized a step ')
                else:
                    # w = [5,5] - step =1 => [4,4]
                    w = w - step
                    
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            

    def predict(self, feat):
        # sign ( x.w+b )
        classification = np.sign(np.dot(np.array(feat),self.w)+self.b)
        if classification !=0 and self.vis:
            self.ax.scatter(feat[0], feat[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feat_val*0.9,self.max_feat_val*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vecor hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')

        # (w.x+b) = -1
        # negative support vecor hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')

        # (w.x+b) = 0
        # decision boundary vecor hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--')

        plt.show()
        



data_dict = {-1:np.array([[1,7],[2,8],[3,8],]),
             1:np.array([[5,1],[6,-1],[7,3],])}



svm = sup_vtr_mch()
svm.fit(data=data_dict)

predict_us = [[1,3],
              #[-6,-5],
              [2,4],
              [3,-6],
              [-7,9],
              [4,6],
              [6,4]]


for p in predict_us:
    svm.predict(p)
svm.visualize()
