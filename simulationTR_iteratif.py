import numpy as np
import matplotlib.pyplot as plt
import os

c0 = 4000
rho0 = 3500

dx = 0.05
dy = dx

sigma = 20
mu = 100

time = 2000

Lx = 20
Ly = 20

Nx = int(Lx / dx)
Ny = int(Ly / dx)

Nsrc = 10

pml_width = 40
sigma_max = 8.0


def get_H_path(with_object):
    return f"H_with_object_{with_object}.npy"


# MILIEU + OBJETS 
def create_medium():
    c_field = np.full((Nx, Ny), c0)
    rho_field = np.full((Nx, Ny), rho0)

    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")


    # diffuseur 1 (FORT) - bas gauche
    obj1x = 120
    obj1y = 110

    # diffuseur 2 (FAIBLE) - bas droite
    obj2x = 320
    obj2y = 130

    # sécurité bornes
    obj1x = np.clip(obj1x, 20, Nx - 20)
    obj1y = np.clip(obj1y, 20, Ny - 20)

    obj2x = np.clip(obj2x, 20, Nx - 20)
    obj2y = np.clip(obj2y, 20, Ny - 20)
    size = 5

    mask1 = (X - obj1x)**2 + (Y - obj1y)**2 <= size**2
    mask2 = (X - obj2x)**2 + (Y - obj2y)**2 <= size**2

    # fort diffuseur
    c_field[mask1] = 1200
    rho_field[mask1] = 8000

    # faible diffuseur
    c_field[mask2] = 2500
    rho_field[mask2] = 4000

    objects = [(obj1x, obj1y), (obj2x, obj2y)]
    masks = [mask1, mask2]

    return c_field, rho_field, masks, objects

# SOURCE / RECEPTEURS
def create_array():
    sourceX = np.linspace(
        pml_width + 1,
        Nx - (pml_width + 1),
        Nsrc,
        dtype=int
    )
    sourceY = np.full(Nsrc, Ny-pml_width + 1)
    return sourceX, sourceY

def gaussian_pulse(n):
    return np.exp(-((n-mu) / sigma)**2)

# PML
def create_pml():
    sigma_x = np.zeros((Nx, Ny))
    sigma_y = np.zeros((Nx, Ny))

    for i in range(Nx):
        if i < pml_width:
            x = (pml_width - i) / pml_width
            sigma_x[i, :] = sigma_max * x**2
        elif i >= Nx - pml_width:
            x = (i - (Nx - pml_width)) / pml_width
            sigma_x[i, :] = sigma_max * x**2

    for j in range(Ny):
        if j >= Ny - pml_width:
            y = (j - (Ny - pml_width)) / pml_width
            sigma_y[:, j] = sigma_max * y**2
        elif j < pml_width:
            y = (pml_width - j) / pml_width
            sigma_y[:, j] = sigma_max * y**2

    return sigma_x, sigma_y


# CFL
def compute_dt(c_field):
    cmax = np.max(c_field)
    dt = 0.5 * dx / (cmax * np.sqrt(2))
    assert (cmax * dt / dx) <= 1 / np.sqrt(2), "CFL non respectée"
    return dt

# PROPAGATION
def propagate_one_source(c_field, rho_field,
                         sigma_x, sigma_y,
                         sourceX, sourceY,
                         E_t,
                         objects=None,
                         visualize=False):

    p = np.zeros((Nx, Ny))
    u = np.zeros_like(p)
    v = np.zeros_like(p)

    H_local = np.zeros((Nsrc, time))
    dt = compute_dt(c_field)

    fig, ax = None, None
    amplitude_objects = np.zeros((2, time))
    if visualize:
        fig, ax = plt.subplots(figsize=(7, 7))

    for n in range(time):

        dpdx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dx)
        dpdy = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dy)

        u += dt * (-dpdx / rho_field - c_field * sigma_x * u)
        v += dt * (-dpdy / rho_field - c_field * sigma_y * v)

        dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
        dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)

        div = dudx + dvdy

        p += dt * (
            -rho_field * c_field**2 * div
            - c_field * (sigma_x + sigma_y) * p
        )

        for j in range(Nsrc):
            p[sourceX[j], sourceY[j]] += E_t[j, n]
            H_local[j, n] = p[sourceX[j], sourceY[j]]
        amplitude_objects[0, n] = p[120,110]
        amplitude_objects[1, n] = p[320,130]

        if visualize and n % 20 == 0:
            ax.clear()
            vmax = np.max(np.abs(p)) + 1e-12
            ax.imshow(p.T, origin="lower", cmap="seismic",
                      vmin=-vmax, vmax=vmax)

            for (x, y) in objects:
                ax.scatter(x, y, s=50)

            ax.set_title(f"t={n*dt*1000:.2f} ms")
            plt.pause(0.001)

    return H_local, amplitude_objects


# MATRICE H
def compute_H(c_field, rho_field,
              sigma_x, sigma_y,
              sourceX, sourceY,
              E_t,
              objects,
              with_object=True,
              visualize=False):

    H = np.zeros(( Nsrc, time))
    amplitude_objects = np.zeros((2, time))

    if not with_object:
        c_field = np.full((Nx, Ny), c0)
        rho_field = np.full((Nx, Ny), rho0)


    H, amplitude_objects = propagate_one_source(
        c_field,
        rho_field,
        sigma_x,
        sigma_y,
        sourceX,
        sourceY,
        E_t,
        objects=objects,
        visualize=visualize
    )

    return H, amplitude_objects


# LOAD / SAVE
def load_or_compute(c_field, rho_field,
                    sigma_x, sigma_y,
                    sourceX, sourceY,
                    E_t,
                    objects,
                    with_object,
                    visualize=False,
                    force=False):

    path = get_H_path(with_object)

    if (not force) and os.path.exists(path):
        print("Loading", path)
        return np.load(path), None

    H, amplitude_objects = compute_H(
        c_field, rho_field,
        sigma_x, sigma_y,
        sourceX, sourceY,
        E_t,
        objects,
        with_object=with_object,
        visualize=visualize
    )
    np.save(path, H)
    return H, amplitude_objects



# MAIN
def main():

    c_field, rho_field, masks, objects = create_medium()

    sourceX, sourceY = create_array()
    sigma_x, sigma_y = create_pml()
    t = np.arange(time) * compute_dt(c_field)
    E_t = np.zeros((Nsrc, time))
    E_t[:, :] = np.array([gaussian_pulse(n) for n in range(time)])
    for i in range(5):

        print(f"iteration n°{i}")

        H_obj, amplitude_objects = load_or_compute(
            c_field, rho_field,
            sigma_x, sigma_y,
            sourceX, sourceY,
            E_t,
            objects,
            with_object=True,
            visualize=True,
            force = True
        )

        H_free, amplitude_objects = load_or_compute(
            c_field, rho_field,
            sigma_x, sigma_y,
            sourceX, sourceY,
            E_t,
            objects,
            with_object=False,
            visualize=True,
            force = True
        )

        H = H_obj - H_free
        
        E_t = H[:, ::-1]
        plt.close('all')


main()