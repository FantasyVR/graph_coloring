import numpy as np
from coloring import check_validity
import matplotlib.pyplot as plt

with open('data/positions.txt', 'r') as f:
    positions = np.array([[float(x) for x in line.split()] for line in f])

with open('data/edges.txt', 'r') as f:
    edges = np.array([[int(x) for x in line.split()] for line in f])

with open('data/coloring.txt', 'r') as f:
    colors = np.array([int(x) for x in f.readlines()])
    color_spec = np.max(colors) - np.min(colors) + 1
    print(f"number of colors: {color_spec}")
    color_bar = [0] * (color_spec)
    for c in colors:
        color_bar[c] += 1
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.pie(color_bar,
            labels=[str(i) for i in range(color_spec)],
            autopct='%.2f%%')
    plt.title("蒙特-卡洛方法颜色分布")  # 设置标题
    plt.savefig('color_distribution.png')

print(f"check validity: {check_validity(edges, colors)}")

color_map = [
    0xFFFFFF, 0xFF0000, 0x00FF00, 0x0000FF, 0x555500, 0x005555, 0x550055,
    0xFFFF00, 0x00FFFF, 0xFF00FF, 0x212121, 0xBBBB00, 0x00BBBB, 0xBB00BB,
    0x123456, 0x654321
]
import taichi as ti

ti.init(arch=ti.cpu)

gui = ti.GUI('cloth', (800, 800), show_gui=False)
while gui.running:
    for i in range(len(edges)):
        p0, p1 = positions[edges[i][0]], positions[edges[i][1]]
        p0, p1 = np.array([p0[0], p0[2]]), np.array([p1[0], p1[2]])
        gui.line(p0, p1, color=color_map[colors[i]])
        gui.text(f"{colors[i]}", (p0 + p1) * 0.5,
                 font_size=20,
                 color=color_map[colors[i]])
    gui.show("coloring.png")
