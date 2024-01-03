# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path

import matplotlib.pyplot as plt

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import cv2
import time


torch.set_grad_enabled(False)
images = Path("assets")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor1 = SuperPoint(max_num_keypoints=500).eval().to(device)  # load the extractor
extractor2 = SuperPoint(max_num_keypoints=10000).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

plt.figure()
cap = cv2.VideoCapture(0)
ret0, image0 = cap.read()
if not ret0:
    exit(-1)
image0 = load_image(image0)
feats0_raw = extractor1.extract(image0.to(device))
n = 0
while True:
    n += 1
    time1 = time.time()
    ret1, image1 = cap.read()
    if not ret1:
        break

    image1 = load_image(image1)
    feats1_raw = extractor2.extract(image1.to(device))
    matches01 = matcher({"image0": feats0_raw, "image1": feats1_raw})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0_raw, feats1_raw, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    time2 = time.time()
    print("-------------------------------------------------------------")
    print("Inference: %d", time2 - time1)              # 输出推理得到匹配的特征点的时间

    plot_image = viz2d.plot_images([image0, image1])
    time3 = time.time()
    print("         Image Mosaic: %d", time3 - time2)              # 输出图像拼接所花费的时间
    viz2d.plot_matches(plot_image, m_kpts0, m_kpts1, color=(0, 255, 0), lw=1)
    time4 = time.time()
    print("                 Feature Points Line: %d", time4 - time3)              # 输出图像特征点之间的连线所花费的时间
    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    result_image = viz2d.plot_keypoints(plot_image, [m_kpts0, m_kpts1], colors=[kpc0, kpc1], radius=3)
    time5 = time.time()
    print("                         Image Label: %d", time5 - time4)              # 输出在图像上标注出匹配好的特征点所花费的时间
    #
    cv2.imshow("", result_image)
    # cv2.imshow("", plot_image)
    cv2.waitKey(20)

cv2.destroyAllWindows()
cap.release()

