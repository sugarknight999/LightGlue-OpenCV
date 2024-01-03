"""
2D visualization primitives based on Matplotlib.
1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch


# 颜色映射函数，用于将数值映射到红-黄-绿色的颜色空间
def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


# 自定义颜色映射函数，与之前的 cm_RdGn 函数类似，但是在该函数中增加了蓝色到红色的映射
def cm_BlRdGn(x_):
    """Custom colormap: blue (-1) -> red (0.0) -> green (1)."""
    x = np.clip(x_, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0, 1.0]])   # 分别表示绿色和红色的 RGBA 值

    xn = -np.clip(x_, -1, 0)[..., None] * 2
    cn = xn * np.array([[0, 0.1, 1, 1.0]]) + (2 - xn) * np.array([[1.0, 0, 0, 1.0]])    # 分别表示蓝色和红色的 RGBA 值
    # 输入值小于 0 的部分使用蓝色映射，大于等于 0 的部分使用红-黄-绿色映射
    # 然后使用 np.clip 将颜色值限制在 0 和 1 之间
    out = np.clip(np.where(x_[..., None] < 0, cn, c), 0, 1) # [..., None] 表示在 x_ 的最后一维上添加一个新的维度
    return out


# 自定义颜色映射函数，用于可视化修剪操作, 将不同的修剪程度表示为颜色的变化
def cm_prune(x_):
    """Custom colormap to visualize pruning"""
    if isinstance(x_, torch.Tensor):            # 检查 x_ 是否为 torch.Tensor 类型，如果是，则变成 numpy 类型
        x_ = x_.cpu().numpy()
    max_i = max(x_)                             # x_ 的最大值保存到 max_i
    # 计算归一化后的 max_i 并保存到 norm_x.
    norm_x = np.where(x_ == max_i, -1, (x_ - 1) / 9)    # 将最大值对应的位置设为 -1，其余值进行线性映射到 (x_ -1, 0) 的范围内
    # 可以保证最大值对应的颜色为蓝色，其他值按照线性比例映射到红色和绿色之间
    return cm_BlRdGn(norm_x)


# 用于水平排列并显示一组图像
def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: list of NumPy RGB (H, W, 3) or PyTorch RGB (3, H, W) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    # conversion to (H, W, 3) for torch.Tensor
    imgs = [    # 图像列表 imgs
        img.permute(1, 2, 0).cpu().numpy()                  # 将通道维度移到最后一个维度，并 pytorch 转 numpy
        if (isinstance(img, torch.Tensor) and img.dim() == 3)
        else img
        for img in imgs
    ]

    n = len(imgs)                               # 实时情况下，n = 2。n 为获取图像的数量
    if not isinstance(cmaps, (list, tuple)):    # 如果 cmaps 不是列表或元组，则将其复制为长度为 n 的列表
        cmaps = [cmaps] * n

    if adaptive:                                # adaptive 确定图像显示区域的大小 figsize
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n                    # 为 False，则默认为 4：3
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(                     # 根据计算得到的 figsize 创建 Figure 和 Axes 对象
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
    )
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))  # 在当前子图 cmap 上显示图像
        ax[i].get_yaxis().set_ticks([])         # 将子图的 y，x 刻度设置为空，即不显示刻度
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()                    #  将子图的轴线关闭，即不显示轴线
        for spine in ax[i].spines.values():     # remove frame 删除边框
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])          # 设置子图的标题
    fig.tight_layout(pad=pad)                   # 函数调整子图布局，通过 pad 参数来控制子图之间的间距


# 用于绘制关键点. 接受关键点坐标列表（kpts），颜色参数（colors）、关键点大小参数（ps）以及 axes 参数
def plot_keypoints(kpts, colors="lime", ps=4, axes=None, a=1.0):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):        # 如果 colors 参数是一个列表，那么每个关键点可以有不同的颜色
        colors = [colors] * len(kpts)
    if not isinstance(a, list):
        a = [a] * len(kpts)
    if axes is None:                        # 如果未提供 axes 参数，则默认使用当前活动的 axes
        axes = plt.gcf().axes               # 实时的，所以 axes 为 None，两个图，所以有两个 axes. print(len(axes)) 为 2
    for ax, k, c, alpha in zip(axes, kpts, colors, a):
        if isinstance(k, torch.Tensor):
            k = k.cpu().numpy()
        # 在给定的 ax 上绘制关键点（关键点的 x，关键点的 y，颜色，点的大小，linewidths = 0 消除关键点周围的边框线条，透明度）
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


# 用于将匹配点对可视化显示在两张图像之间（两张图像中的特征点坐标，匹配点对颜色，线宽，端点尺寸，线条透明度，每个匹配点对的标签，指定要在哪些轴上显示匹配点对）
def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, a=1.0, labels=None, axes=None):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()                         # 获取当前图形对象 fig
    if axes is None:                        # 根据给定的轴 axes 获取要显示匹配点对的两个轴 ax0 和 ax1
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes
    if isinstance(kpts0, torch.Tensor):     # 将特征点坐标转换成 numpy 数组
        kpts0 = kpts0.cpu().numpy()
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    assert len(kpts0) == len(kpts1)         # 并检查两个输入数组的长度是否相等
    if color is None:                       # 没有指定颜色，则 hsv() 函数随机生成颜色
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):    # elif (color[0]不是 元组或者列表)
        color = [color] * len(kpts0)                                    # 则将其复制 len(kpts0) 次以得到一个列表

    if lw > 0:
        for i in range(len(kpts0)):     # 便利特征匹配点，用 ConnectionPatch() 函数绘制一条线段，并设置其颜色、线宽、透明度、标签等参数
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:  # 如果指定了端点尺寸，则在两个轴上分别绘制特征点
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


# 用于在指定图像索引 idx 对应的图形中添加文本 text
def add_text(
    idx,
    text,
    pos=(0.01, 0.99),
    fs=15,
    color="w",
    lcolor="k",
    lwidth=2,
    ha="left",
    va="top",
):
    ax = plt.gcf().axes[idx]        # 根据给定的索引获取要添加文本的图形对象 ax
    t = ax.text(                    # 使用 ax.text() 函数在指定的位置按照参数添加文本
        *pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes
    )
    if lcolor is not None:
        t.set_path_effects(         # 用 path_effects.Stroke() 函数为文本添加描边效果
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )


# 保存当前的图形文件，并且使用紧凑边界框（bbox_inches="tight"）和无间距（pad_inches=0）的设置进行保存
def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
