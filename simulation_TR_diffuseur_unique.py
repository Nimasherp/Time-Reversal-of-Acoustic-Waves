import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import os

# Simulation dans l'eau de mer 

# PARAMETRES 
#----------------------------------------------------
c0 = 4000 # m/s
rho0 = 3500 # kg/m^3

dx = 0.05 # m
dy = dx

sigma = 20
mu = 100

time = 2_000

Lx = 20 # largeur m
Ly = 20 # profondeur m
Nx = int(Lx/dx)
Ny = int(Ly/dx)

Nsrc = 10
pml_width = 40
sigma_max = 8.0

def get_H_path(with_object):
    return f"H_with_object_dif_unique_{with_object}.npy"

c_field =np.full((Nx, Ny), c0)
rho_field =np.full((Nx, Ny), rho0)
#----------------------------------------------------


# SOURCE / RECEPTEUR
#----------------------------------------------------
sourceX = np.linspace(pml_width+1, Nx-(pml_width+1), Nsrc, dtype=int)
sourceY = np.full(Nsrc, Ny - pml_width + 1)
#----------------------------------------------------


# OBJET
#----------------------------------------------------
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

objx = Nx//2
objy = Ny//2
taille = 5

cercle_mask = (X - objx)**2 + (Y - objy)**2 <= taille**2
#----------------------------------------------------

# MILIEU 
#----------------------------------------------------
c_field[cercle_mask] = 1500 # on considère l'objet comme du metal
rho_field[cercle_mask] = 997 
cmax = np.max(c_field) 

# CFL stable en 2D : c*dt/dx < 1/sqrt(2)
dt = 0.5 * dx / (cmax*np.sqrt(2))
assert (cmax*dt/dx) <= 1/np.sqrt(2), "Condition CFL non verifié"

# PML ZONE
#----------------------------------------------------
sigma_x = np.zeros((Nx, Ny))
sigma_y = np.zeros((Nx, Ny))
for i in range(Nx):
    # gauche
    if i < pml_width:
        x = (pml_width - i) / pml_width
        sigma_x[i, :] = sigma_max * x**2
    # droite
    elif i >= Nx - pml_width:
        x = (i - (Nx - pml_width)) / pml_width
        sigma_x[i, :] = sigma_max * x**2
for j in range(Ny):
    # haut
    if j >= Ny - pml_width:
        y = (j - (Ny - pml_width)) / pml_width
        sigma_y[:, j] = sigma_max * y**2
    # bas
    elif j < pml_width:
        y =(pml_width - j) / pml_width
        sigma_y[:, j] = sigma_max * y**2
#----------------------------------------------------
dpdx = np.zeros((Nx, Ny))
dpdy = np.zeros_like(dpdx)


def compute_H(visualize=False, viz_step=20, save=True, with_object=True):

    print("Computation de H...")

    H = np.zeros((Nsrc, Nsrc, time))
    c_local = c_field.copy()
    rho_local = rho_field.copy()
    if not with_object:
        c_local[:] = c0
        rho_local[:] = rho0

    if visualize:
        fig, ax = plt.subplots(figsize=(7,7))

    for i_src in range(1):

        print(f"Source {i_src+1}/{Nsrc}")

        p = np.zeros((Nx, Ny))
        u = np.zeros_like(p)
        v = np.zeros_like(p)

        for n in range(time):
            print(p[100, 100])
            dpdx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2*dx)
            dpdy = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2*dy)

            u += dt * (-dpdx/rho_local - c_local*sigma_x*u)
            v += dt * (-dpdy/rho_local - c_local*sigma_y*v)
            # if with_object:
            #     u[cercle_mask] = 0
            #     v[cercle_mask] = 0

            dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
            dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2*dy)

            div = dudx + dvdy

            p += dt * (
                -rho_local*c_local**2 * div
                - c_local*(sigma_x + sigma_y)*p
            )

            pulse = np.exp(-((n - mu)/sigma)**2)
            p[sourceX[i_src], sourceY[i_src]] += pulse

            for j in range(Nsrc):
                if j != i_src:
                    H[i_src, j, n] = p[sourceX[j], sourceY[j]]

            if visualize and (n % viz_step == 0):
                ax.clear()
                vmax = np.max(np.abs(p)) + 1e-12
                ax.imshow(
                    p.T,
                    origin="lower",
                    cmap="seismic",
                    vmin=-vmax,
                    vmax=vmax
                )

                ax.scatter(objx, objy, c="black", s=30)
                ax.set_title(f"src {i_src+1}/{Nsrc} t={int(n*dt*1000)}ms")

                plt.pause(0.001)

    H_PATH = get_H_path(with_object)

    if save:
        np.save(H_PATH, H)
        print(f"H sauvegardé -> {H_PATH}")

    return H

def load_or_compute(with_object, visualize=False, force=False):

    H_PATH = get_H_path(with_object)

    if (not force) and os.path.exists(H_PATH):
        print(f"Loading {H_PATH}")
        return np.load(H_PATH)

    return compute_H(
        visualize=visualize,
        save=True,
        with_object=with_object
    )

H_obj = load_or_compute(with_object=True, visualize=True)
H_free = load_or_compute(with_object=False, visualize=True)
H = H_obj - H_free

# RETOURNEMENT TEMPOREL 
signals_RT = H[0, :, ::-1]

p = np.zeros((Nx, Ny))
dpdx = np.zeros_like(p)
dpdy = np.zeros_like(p)
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))

fig_rt, ax_rt = plt.subplots(figsize=(7,7))


for n in range(time):

    dpdx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2*dx)
    dpdy = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2*dy)

    u += dt * (-dpdx/rho_field - c_field*sigma_x*u)
    v += dt * (-dpdy/rho_field - c_field*sigma_y*v)


    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
    dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2*dy)

    div = dudx + dvdy

    p += dt * (
        -rho_field*c_field**2 * div
        - c_field*(sigma_x + sigma_y)*p
    )

    for j in range(Nsrc):
        if (j != 0):
            p[sourceX[j], sourceY[j]] += signals_RT[j, n]

 
    if(n % 20) == 0:
        ax_rt.clear()

        vmax = np.max(np.abs(p))

        ax_rt.imshow(
            p.T,
            origin="lower",
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax
        )

        ax_rt.scatter(objx, objy, c="black", s=30)
        ax_rt.set_title(f"retournement t = {n*dt*1000} ms")

        plt.pause(0.001)

plt.show()
