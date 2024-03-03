import matplotlib.pyplot as plt
import numpy as np
x = [1,2,3,4,5,6]
x_ticks_label = ['0-200','200-400','400-700','700-1000','1000-1500','1500+']
y1 = [75.3, 83.1, 84.1, 86.8, 82.0, 73.0]
y2 = [70.8, 82.3, 84.5, 86.1, 84.7, 77.0]
y3 = [70.8, 83.2, 84.5, 89.8, 86.0, 84.8]
plt.xticks(x, x_ticks_label, fontsize=10)
plt.plot(x, y1, 'o-', color='cyan', label='Baseline')
plt.plot(x, y2, '^-', color='green',label='Ours++')
plt.plot(x, y3, 's-', color='magenta',label='Ours++ (Fully)')
#plt.xlabel('Crowd Number')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.title('ShanghaiTech PartA')
plt.show()
