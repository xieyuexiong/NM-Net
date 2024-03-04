import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os as os
from PIL import Image
import numpy as np

# 读取原始图像
heatmap_data = cv2.imread("/root/workspace/mmdetection/feature_map/bbb3.png")  # 替换为你的图像路径



# 将图像转换为灰度图
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 通过applyColorMap函数将灰度图转换为热力图数据
# heatmap_data = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

# 显示热力图




# 生成热力图数据，这里使用随机数据代替
# heatmap_data = np.random.rand(256, 256)

# 将热力图数据缩放到0~255的范围，以便与图像叠加
heatmap_data = (heatmap_data * 255).astype(np.uint8)

# 读取原始图像
original_image = cv2.imread("/root/workspace/MRI_dataset/MRI_ORG/folder11_Org_148dcm_5.png")  # 替换为你的原始图像路径




# 将热力图数据应用上颜色映射（通常是热力图的颜色，比如红、黄、蓝等）
heatmap_colormap = cv2.applyColorMap(heatmap_data, cv2.COLORMAP_JET)

# 将热力图叠加到原始图像上
alpha = 0.3  # 设置叠加的透明度
blended_image = cv2.addWeighted(original_image, 1-alpha, heatmap_colormap, alpha, 0)

# # 显示结果
# plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

cv2.imwrite(os.path.join("/root/workspace/mmdetection/",'heatmap0.png'), blended_image)
