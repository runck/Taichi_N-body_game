import taichi as ti
# import numpy as np

# ti.init(arch=ti.gpu)
ti.init(arch=ti.gpu, debug=True)

res_x, res_y = 640, 640
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))
gui = ti.GUI('N-body Dodge', res=(res_x, res_y), fast_gui=False)
# gui.fps_limit = 24


my_pos = ti.Vector.field(2, ti.f32, ())

N = 20
n_pos = ti.Vector.field(2, ti.f32, N)
n_vel = ti.Vector.field(2, ti.f32, N)
n_force = ti.Vector.field(2, ti.f32, N)

dt_show = 1e-3
dt_subnum = 5
min_diff = 1e-5

move_dpos_x = 0.005/dt_subnum
move_dpos_y = move_dpos_x * res_x/res_y

score = ti.field(ti.i32, ())
is_alive = ti.field(ti.i32, ())

@ti.kernel
def initialize():
    safe_distance = 0.2
    i = 0
    
    my_pos[None] = ti.Vector([0.5, 0.5])
    while i<N:
        x = ti.random()
        y = ti.random()
        pos = ti.Vector([x, y])
        if (pos - my_pos[None]).norm(1e-5) < safe_distance:
            continue
        n_pos[i] = pos
        n_vel[i] = ti.Vector([ti.random(), ti.random()]) * 1
        i+=1
    
    is_alive[None] = True    
    score[None]=-50


@ti.kernel
def compute_force():
    ti.block_local(n_pos)
    
    my_inv_mass = 1e-2
    for i in range(N):
        # n_force[i] = ti.Vector([0.0, 0.0])
        diff = n_pos[i] - my_pos[None]
        r = diff.norm(min_diff)

        if score[None]>0 and r < 8/max(res_x, res_y):
            is_alive[None] = False
            print(r, my_pos[None], n_pos[i])
        f = -(1.0/r)**3 * diff * my_inv_mass
        n_force[i] = f
        
    
    inv_mass = 1
    for i in range(N):
        p = n_pos[i]
        for j in range(N):
            if i != j: 
                diff = p-n_pos[j]
                r = diff.norm(min_diff)
                f = -(1.0/r)**3 * diff
                n_force[i] += f
        
    
@ti.kernel
def update():
    dt = dt_show/dt_subnum
    for i in range(N):
        n_vel[i] += dt*n_force[i]
        n_pos[i] += dt*n_vel[i]
        
        bnd_gain = 1.5
        vel_max = 5
        x, y = n_pos[i]
        if x > 1 or x < 0:
            n_vel[i][0] = -n_vel[i][0]
            if n_vel[i].norm()*bnd_gain < vel_max:
                n_vel[i]*=bnd_gain
        if y > 1 or y < 0:
            n_vel[i][1] = -n_vel[i][1]
            if n_vel[i].norm()*bnd_gain < vel_max:
                n_vel[i]*=bnd_gain

        

def move():
    x , y = my_pos[None][0], my_pos[None][1]
    # if gui.get_event(ti.GUI.PRESS):
        # if gui.event.key == 'r': initialize()
        # elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: return
    if gui.is_pressed('w'): y += move_dpos_y
    if gui.is_pressed('s'): y -= move_dpos_y
    if gui.is_pressed('a'): x -= move_dpos_x
    if gui.is_pressed('d'): x += move_dpos_x
        
    x = min(1, max(0, x))
    y = min(1, max(0, y))
    
    my_pos[None][0], my_pos[None][1] = x, y


def draw():
    if is_alive[None]:
        gui.text(str(score[None]), (0.5, 0.95), font_size=20, color = 0x777777)
        score[None] += 1
    else:
        gui.text('Die '+str(score[None]), (0.5, 0.5), font_size=80)

    gui.circle(my_pos[None].to_numpy(), color=0xff0000, radius=10)
    gui.circles(n_pos.to_numpy(), color=0x0055ff, radius=6)  
    gui.show()
    # gui.show(f'img/{51+score[None]:0>3d}.png')



initialize()
while gui.running:
    for i in range(dt_subnum):
        if is_alive[None]:
            move()
            compute_force()
            update()
    draw()
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'r': initialize()
