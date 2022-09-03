import taichi as ti
import numpy as np
import time

ti.init(arch=ti.vulkan)
N = 100
NV = (N + 1)**2
NT = 2 * N**2
NE = 2 * N * (N + 1) + N**2
pos = ti.Vector.field(3, ti.f32, shape=NV)
tri = ti.field(ti.i32, shape=3 * NT)
edge = ti.Vector.field(2, ti.i32, shape=NE)
num_colors = 0
color_idx = ti.field(ti.i32, shape=NE)

old_pos = ti.Vector.field(3, ti.f32, NV)
inv_mass = ti.field(ti.f32, NV)
vel = ti.Vector.field(3, ti.f32, NV)
rest_len = ti.field(ti.f32, NE)
h = 0.01
MaxIte = 100

paused = ti.field(ti.i32, shape=())


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        pos[idx] = ti.Vector([i / N, 0.5, j / N])
        inv_mass[idx] = 1.0
    inv_mass[N] = 0.0
    inv_mass[NV - 1] = 0.0


@ti.kernel
def init_tri():
    for i, j in ti.ndrange(N, N):
        tri_idx = 6 * (i * N + j)
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            tri[tri_idx + 0] = pos_idx
            tri[tri_idx + 1] = pos_idx + N + 2
            tri[tri_idx + 2] = pos_idx + 1
            tri[tri_idx + 3] = pos_idx
            tri[tri_idx + 4] = pos_idx + N + 1
            tri[tri_idx + 5] = pos_idx + N + 2
        else:
            tri[tri_idx + 0] = pos_idx
            tri[tri_idx + 1] = pos_idx + N + 1
            tri[tri_idx + 2] = pos_idx + 1
            tri[tri_idx + 3] = pos_idx + 1
            tri[tri_idx + 4] = pos_idx + N + 1
            tri[tri_idx + 5] = pos_idx + N + 2


@ti.kernel
def init_edge():
    for i, j in ti.ndrange(N + 1, N):
        edge_idx = i * N + j
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        edge_idx = start + j * N + i
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
    start = 2 * N * (N + 1)
    for i, j in ti.ndrange(N, N):
        edge_idx = start + i * N + j
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
        else:
            edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])
    for i in range(NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        rest_len[i] = (p1 - p2).norm()


@ti.kernel
def semi_euler():
    gravity = ti.Vector([0.0, -0.1, 0.0])
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] += h * gravity
            old_pos[i] = pos[i]
            pos[i] += h * vel[i]


@ti.kernel
def solve_constraints():
    ti.loop_config(serialize=True)
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        if invM0 != 0.0:
            pos[idx0] += 0.5 * invM0 * l * gradient
        if invM1 != 0.0:
            pos[idx1] -= 0.5 * invM1 * l * gradient


@ti.kernel
def update_vel():
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / h


@ti.kernel
def collision():
    for i in range(NV):
        if pos[i][2] < -2.0:
            pos[i][2] = 0.0


def step():
    semi_euler()
    for i in range(MaxIte):
        solve_constraints()
        collision()
    update_vel()


def save_initial_state(generate=False):
    if generate:
        from coloring import graph_coloring
        c_v = graph_coloring(edge.to_numpy(), 0)
        with open('data/coloring.txt', 'w') as f:
            for i in range(len(c_v)):
                f.write(f"{c_v[i]}\n")
    else:
        with open('data/coloring.txt', 'r') as f:
            c_v = [int(x) for x in f.readlines()]
    color_idx.from_numpy(np.asarray(c_v, dtype=np.int32))
    global num_colors
    num_colors = np.max(c_v) - np.min(c_v) + 1
    print(f"number of colors: {num_colors}")

    with open('data/edges.txt', 'w') as f:
        e = edge.to_numpy()
        for i in range(NE):
            f.write(str(e[i][0]) + ' ' + str(e[i][1]) + '\n')

    with open('data/positions.txt', 'w') as f:
        p = pos.to_numpy()
        for i in range(NV):
            f.write(
                str(p[i][0]) + ' ' + str(p[i][1]) + ' ' + str(p[i][2]) + '\n')


init_pos()
init_tri()
init_edge()
save_initial_state(generate=True)
print('initial state saved')

step()
frame = 2500
start = time.time()
for i in range(frame):
    print(f"frame {i}")
    step()
end = time.time()
print(f"GS cloth Average simulation time: {(end - start)/frame}s")

# window = ti.ui.Window("Display Mesh", (1024, 1024), show_window=False)
# canvas = window.get_canvas()
# scene = ti.ui.Scene()
# camera = ti.ui.make_camera()
# camera.position(0.5, 0.0, 2.5)
# camera.lookat(0.5, 0.5, 0.0)
# camera.fov(90)

# paused[None] = 1
# while window.running:
#     for e in window.get_events(ti.ui.PRESS):
#         if e.key in [ti.ui.ESCAPE]:
#             exit()
#     if window.is_pressed(ti.ui.SPACE):
#         paused[None] = not paused[None]

#     step()
#     frame += 1
#     # if not paused[None]:
#     #     step()
#     #     paused[None] = not paused[None]

#     camera.track_user_inputs(window, movement_speed=0.003, hold_key=ti.ui.RMB)
#     scene.set_camera(camera)
#     scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

#     scene.mesh(pos, tri, color=(1.0, 1.0, 1.0), two_sided=True)
#     scene.particles(pos, radius=0.01, color=(0.6, 0.0, 0.0))
#     canvas.scene(scene)
#     # window.show()
#     if frame > 100:
#         break
