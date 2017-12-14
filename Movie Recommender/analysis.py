import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm

df = pd.read_csv("analysis.csv")
df = df.drop("Unnamed: 0", axis=1)
data = df.as_matrix()

color = ['b', 'w', 'g', 'w', 'r', 'c', 'm', 'y', 'k']
'''
# Plotting user group vs error for CF.
plt.figure(1)
for i in range(5, 49, 10):
    t = df.loc[df['moviegrp'] == i].as_matrix().T
    plt.plot(t[0], t[2], color[int((i - 5) / 5)], label="UserGroup#" + str(i))
plt.xlabel("# of User Groups")
plt.ylabel("Error")
plt.title("How # of User Groups effect CF Error")
plt.show()

# Plotting user group vs error for SVD.
plt.figure(2)
for i in range(5, 49, 10):
    t = df.loc[df['moviegrp'] == i].as_matrix().T
    plt.plot(t[0], t[3], color[int((i - 5) / 5)], label="UserGroup#" + str(i))
plt.xlabel("# of User Groups")
plt.ylabel("Error")
plt.title("How # of User Groups effect SVD Error")
plt.show()

# Plotting movie group vs error for CF.
plt.figure(3)
for i in range(5, 49, 10):
    t = df.loc[df['usergrp'] == i].as_matrix().T
    plt.plot(t[1], t[2], color[int((i - 5) / 5)], label="MovieGroup#" + str(i))
plt.xlabel("# of Movie Groups")
plt.ylabel("Error")
plt.title("How # of Movie Groups effect CF Error")
plt.show()

# Plotting movie group vs error for SVD.
plt.figure(4)
for i in range(5, 49, 10):
    t = df.loc[df['usergrp'] == i].as_matrix().T
    plt.plot(t[1], t[3], color[int((i - 5) / 5)], label="MovieGroup#" + str(i))
plt.xlabel("# of Movie Groups")
plt.ylabel("Error")
plt.title("How # of Movie Groups effect SVD Error")
plt.show()

# Plotting user group vs time for CF.
plt.figure(5)
for i in range(5, 49, 10):
    t = df.loc[df['moviegrp'] == i].as_matrix().T
    plt.plot(t[0], t[4], color[int((i - 5) / 5)], label="UserGroup#" + str(i))
plt.xlabel("# of User Groups")
plt.ylabel("Time")
plt.title("How # of User Groups effect CF Time")
plt.show()

# Plotting user group vs time for SVD.
plt.figure(6)
for i in range(5, 49, 10):
    t = df.loc[df['moviegrp'] == i].as_matrix().T
    plt.plot(t[0], t[5], color[int((i - 5) / 5)], label="UserGroup#" + str(i))
plt.xlabel("# of User Groups")
plt.ylabel("Time")
plt.title("How # of User Groups effect SVD Time")
plt.show()

# Plotting movie group vs time for CF.
plt.figure(7)
for i in range(5, 49, 10):
    t = df.loc[df['usergrp'] == i].as_matrix().T
    plt.plot(t[1], t[4], color[int((i - 5) / 5)], label="MovieGroup#" + str(i))
plt.xlabel("# of Movie Groups")
plt.ylabel("Time")
plt.title("How # of Movie Groups effect CF Time")
plt.show()

# Plotting movie group vs time for SVD.
plt.figure(8)
for i in range(5, 49, 10):
    t = df.loc[df['usergrp'] == i].as_matrix().T
    plt.plot(t[1], t[5], color[int((i - 5) / 5)], label="MovieGroup#" + str(i))
plt.xlabel("# of Movie Groups")
plt.ylabel("Time")
plt.title("How # of Movie Groups effect SVD Time")
plt.show()
'''
# ----------------------------------------------------------------------

x = []
colerr = []
svderr = []
coltime = []
svdtime = []
for i in range(5, 50, 5):
	t = df.loc[df['moviegrp'] == i].mean()
	x.append(i)
	colerr.append(t['colerr'])
	svderr.append(t['svderr'])
	coltime.append(t['coltime'])
	svdtime.append(t['svdtime'])

plt.figure(9)
plt.plot(x, colerr, 'ro')
plt.plot(x, svderr, 'go')
plt.xlabel("# of Movie Groups")
plt.ylabel("Error")
plt.title("How # of Movie Groups Effects Error")
plt.show()


plt.figure(10)
plt.plot(x, coltime, 'ro')
plt.plot(x, svdtime, 'go')
plt.xlabel("# of Movie Groups")
plt.ylabel("Time")
plt.title("How # of Movie Groups Effects Time")
plt.show()
print(lm.LinearRegression().fit(np.array(x).reshape((9, 1)), colerr).coef_)
print(lm.LinearRegression().fit(np.array(x).reshape((9, 1)), svderr).coef_)
print(lm.LinearRegression().fit(np.array(x).reshape((9, 1)), coltime).coef_)
print(lm.LinearRegression().fit(np.array(x).reshape((9, 1)), svdtime).coef_)

x = []
colerr = []
svderr = []
coltime = []
svdtime = []
for i in range(5, 50, 5):
	t = df.loc[df['usergrp'] == i].mean()
	x.append(i)
	colerr.append(t['colerr'])
	svderr.append(t['svderr'])
	coltime.append(t['coltime'])
	svdtime.append(t['svdtime'])

plt.figure(11)
plt.plot(x, colerr, 'ro')
plt.plot(x, svderr, 'go')
plt.xlabel("# of User Groups")
plt.ylabel("Error")
plt.title("How # of User Groups Effects Error")
plt.show()

plt.figure(12)
plt.plot(x, coltime, 'ro')
plt.plot(x, svdtime, 'go')
plt.xlabel("# of User Groups")
plt.ylabel("Time")
plt.title("How # of User Groups Effects Time")
plt.show()
print(lm.LinearRegression().fit(np.array(x).reshape((9, 1)), colerr).coef_)
print(lm.LinearRegression().fit(np.array(x).reshape((9, 1)), svderr).coef_)
print(lm.LinearRegression().fit(np.array(x).reshape((9, 1)), coltime).coef_)
print(lm.LinearRegression().fit(np.array(x).reshape((9, 1)), svdtime).coef_)
