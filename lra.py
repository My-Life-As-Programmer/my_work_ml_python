from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([2,3,4,5,6], dtype=np.float64)
#ys = np.array([4,7,9,11,13], dtype=np.float64)

#plt.plot(xs,ys)

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_n_intercept(xs,ys):
    m = ( ( (mean(xs)*mean(ys)) - mean(xs*ys) ) /
          ( (mean(xs)**2) - mean(xs**2) ) )
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - ( squared_error_regr / squared_error_y_mean )


xs, ys = create_dataset(40, 20, 2,'pos')

m,b = best_fit_slope_n_intercept(xs,ys)

regression_line = [(m*x) for x in xs]

pre_x = 41
pre_y = (m*pre_x)+b

r_square = coefficient_of_determination(ys, regression_line)

print("r^2 = ",r_square)
print(pre_y)
#print(m, b)
plt.scatter(xs,ys)
plt.scatter(pre_x,pre_y,color='r')
plt.plot(xs,regression_line)
plt.show()
