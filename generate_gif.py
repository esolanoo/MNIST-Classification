import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from IPython import display 
import matplotlib.animation as animation

X_tsne = pd.read_csv('tsne_3.csv', index_col=False).iloc[:, 1:]

x = X_tsne[:7000]['dim1'].to_numpy()
y = X_tsne[:7000]['dim2'].to_numpy()
z = X_tsne[:7000]['dim3'].to_numpy()

fig = plt.figure(figsize=(15, 15))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

classes = X_tsne[:7000]['label']

for class_name in set(classes):
    indices = np.where(classes == class_name)
    ax.scatter(x[indices], y[indices], z[indices], label=class_name)

ax.set_xlabel('1st Embedded Dim')
ax.set_ylabel('2nd Embedded Dim')
ax.set_zlabel('3rd Embedded Dim')

plt.legend()
fig.suptitle('Scatter plot by class', fontsize=14) 

def init():
    for class_name in set(classes):
        indices = np.where(classes == class_name)
        ax.scatter(x[indices], y[indices], z[indices], label=class_name)
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True, save_count=1500)

f = r"c://Users/Eduardo/Videos/animation.gif" 
writergif = animation.PillowWriter(fps=40) 
anim.save(f, writer=writergif)
