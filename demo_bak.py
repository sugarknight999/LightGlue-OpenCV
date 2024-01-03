# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path

import matplotlib.pyplot as plt

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d_bak
import torch
import cv2
import time


torch.set_grad_enabled(False)
images = Path("assets")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor1 = SuperPoint(max_num_keypoints=1000).eval().to(device)  # load the extractor
extractor2 = SuperPoint(max_num_keypoints=5000).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

plt.figure()
cap = cv2.VideoCapture(0)
ret0, image0 = cap.read()
if not ret0:
    exit(-1)
image0 = load_image(image0)
feats0_raw = extractor1.extract(image0.to(device))
print(feats0_raw)
n = 0
while True:
    n += 1
    ret1, image1 = cap.read()
    if not ret1:
        break
    # 特征点匹配过程
    image1 = load_image(image1)
    feats1_raw = extractor2.extract(image1.to(device))
    matches01 = matcher({"image0": feats0_raw, "image1": feats1_raw})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0_raw, feats1_raw, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    axes = viz2d_bak.plot_images([image0, image1])
    viz2d_bak.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)

    viz2d_bak.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    kpc0, kpc1 = viz2d_bak.cm_prune(matches01["prune0"]), viz2d_bak.cm_prune(matches01["prune1"])
    viz2d_bak.plot_images([image0, image1])
    viz2d_bak.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

    time1 = time.time()
    plt.show(block=False)     # 还挺花时间的
    plt.show(block=False)
    plt.close()
    time2 = time.time()
    print("close1: %d", time2 - time1)
    plt.savefig(images / "saved_images" / "image_{:03d}.jpeg".format(n+1))
    time3 = time.time()
    print("      save: %d", time3 - time2)
    plt.close()
    time4 = time.time()
    print("             close2: %d", time4 - time3)

