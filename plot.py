import numpy as np  
import matplotlib.pyplot as plt  

font = {'family' : 'serif',  
        'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 11,  
        }
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

y1=[20.42,19.52,53.85,46.35,70.27,76.98,83.18,79.28,80.86,79.28,79.88,80.58,80.58,77.98,80.84]
# y2=[1.0220490305631369, 1.0239620393102948, 1.039077349253245, 1.030210054851509, 1.0108771749564238, 0.99330649009576666]
ax = plt.gca()  
ax.set_ylabel('val_acc',fontdict=font)  
ax.set_xlabel('epochs',fontdict=font)

plt.grid(True, linestyle = "-.", color = "r", linewidth = "1")  

plt.plot(x,y1)
# plt.plot(x,y2)  
plt.savefig("plot.jpg")

