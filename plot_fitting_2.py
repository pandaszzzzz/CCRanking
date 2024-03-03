import matplotlib.pyplot as plt
import numpy as np
x = [1,2,3,4,5,6,7]
x_ticks_label = ['0-200','200-400','400-700','700-1000','1000-1500','1500-3000', '3000+']
y1 = [72.1, 81.1, 81.2, 83.1, 79.9, 84.2, 81.0]
y2 = [71.5, 78.9, 79.2, 79.9, 79.3, 83.3, 82.4]
y3 = [73.5, 82.2, 81.3, 83.9, 81.4, 87.5, 86.2]
plt.xticks(x, x_ticks_label, fontsize=10)
plt.plot(x, y1, 'o-', color='cyan', label='Baseline')
plt.plot(x, y2, '^-', color='green',label='Ours++')
plt.plot(x, y3, 's-', color='magenta',label='Ours++ (Fully)')
#plt.xlabel('Crowd Number')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.title('UCF-QNRF')
plt.show()
