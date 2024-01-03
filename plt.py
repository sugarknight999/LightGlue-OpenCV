import numpy as np
import matplotlib.pyplot as plt
import time
import pyautogui

# 创建数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制图形
plt.plot(x, y)
plt.title("Sin wave")

# 显示图形
plt.show()

# 延迟5秒钟
time.sleep(5)

# 模拟按下Alt+F4关闭窗口
pyautogui.hotkey('alt', 'f4')

