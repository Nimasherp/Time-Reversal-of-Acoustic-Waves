import numpy as np
import matplotlib.pyplot as plt
from PIL import Image   

# PARAMETRES PHYSIQUES
c = 343 # m/s
dx = 0.01 # en m 
temps = 2000 

# GRILLE
L = 5 # Taille de la chambre
Nx = Ny = int(L / dx) # Taille de la grille
def init():
    p_prev = np.zeros((Nx, Ny)) 
    p = np.zeros((Nx, Ny))
    p_next = np.zeros((Nx, Ny))
    return p_prev, p, p_next

# TRANSDUCTEUR
n_t = 128 # Nombre de transducteur 
tx_pos = np.linspace(10, Ny-10, n_t, dtype=int) # Positions transducteurs, sur le côté droit
ty_pos = np.linspace(10, Nx-10, n_t, dtype=int) # Positions transducteurs, sur le côté haut
transducteur = np.zeros((2, n_t, temps)) 

# SOURCE 
def gauss_pulse(n, t0 = 50, spread = 25, f0 = 2000):
    return (np.exp(-((n - t0)/spread)**2)* np.sin(2*np.pi*f0*n*dt))

nb_source = int(input("Combien de source ? : "))
# première source au centre
sources = np.zeros((nb_source, 2), dtype=int)
sources[0] = [Nx // 2, Ny // 2]
sources[1:] = np.random.randint(
    low=1,
    high=min(Nx, Ny) - 1,
    size=(nb_source - 1, 2)
)

# MILIEU
# l'hétérogéneité étant dessiné en noir ses valeurs RGB sur 255 vont valoir 1
img = Image.open("random_Heterogenous.png").convert("L")
img = img.resize((Nx, Ny))
arr = np.array(img)/255 
c_map = c + 200*arr # distribution de vitesse 343 dans l'air et 343 + 200 dans l'hétérogéneité

c_max = np.max(c_map)
dt = 0.5 * dx / (c_max*np.sqrt(2))
assert (c_max*dt/dx) <= 1/np.sqrt(2), "Condition CFL non verifié"
lam2 = (c_map * dt / dx)**2

# VISUEL MILIEU HETEROGENE
contrast = np.abs(c_map - c)
contrast = contrast / np.max(contrast)
background = np.ones((Nx, Ny, 4))  # RGBA (blanc par défaut)

background[..., 3] = np.abs(contrast.T-1)  # canal alpha
background[..., 0] = contrast.T *255
background[..., 1] = contrast.T *255
background[..., 2] = contrast.T *255

fig, ax = plt.subplots(figsize=(7, 7))

# EMISSION
p_prev, p, p_next = init()
for n in range(temps):
    # FDTD Méthode :
    p_next[1:-1, 1:-1] = (
        2 * p[1:-1, 1:-1]
        - p_prev[1:-1, 1:-1]
        + lam2[1:-1, 1:-1] * (
            p[2:, 1:-1] +
            p[:-2, 1:-1] +
            p[1:-1, 2:] +
            p[1:-1, :-2]
            - 4 * p[1:-1, 1:-1]
        )
    )

    for k in range(nb_source):
        p_next[
            sources[k, 0],
            sources[k, 1]
        ] += gauss_pulse(n)

    print(p_next[100, 100])

    if(n % 20) == 0:
        ax.clear()
        
        ax.imshow(p_next.T, origin="lower", cmap="seismic",vmin=-0.5, vmax=0.5)
                  
        ax.scatter(sources[:, 0], sources[:, 1], c="white", s=10)

        ax.imshow(
            background,
            origin="lower",
        )

        ax.set_title(f"Emission t={int(n*dt*1000)} ms")

        plt.pause(0.001)

    # Enregistrement du signal
    transducteur[0, :, n] = p_next[-2, tx_pos] 
    transducteur[1, :, n] = p_next[ty_pos , -2]
    p_prev, p, p_next = p, p_next, p_prev
            


# RETOURNEMENT TEMPOREL
transducteur_retourne = transducteur[:, :, ::-1]

p_prev, p, p_next = init()

# Courbe du signal au sources
signal_sources = np.zeros((nb_source, temps))

fig_rt, ax_rt = plt.subplots(figsize=(7, 7))

for n in range(temps):

    p_next[1:-1, 1:-1] = (
        2 * p[1:-1, 1:-1]
        - p_prev[1:-1, 1:-1]
        + lam2[1:-1, 1:-1] * (
            p[2:, 1:-1] + p[:-2, 1:-1] +
            p[1:-1, 2:] + p[1:-1, :-2]
            - 4 * p[1:-1, 1:-1]
        )
    )

    # injection TR
    p_next[-2, tx_pos] += transducteur_retourne[0, :, n]
    p_next[ty_pos, -2] += transducteur_retourne[1, :, n] 

    # signaux aux sources
    signal_sources[:, n] = p_next[sources[:, 0], sources[:, 1]]

    # affichage
    if n % 20 == 0:
        ax_rt.clear()
    
        ax_rt.imshow(p_next.T, origin="lower", cmap="seismic", vmin=-0.5, vmax=0.5)
                  
        ax_rt.scatter(sources[:, 0], sources[:, 1], c="white", s=10)

        ax_rt.imshow(background,origin="lower",)

        ax_rt.set_title(f"Time reversal t={int(n*dt*1000)} ms")

        plt.pause(0.001)

    p_prev, p, p_next = p, p_next, p_prev


plt.figure()
plt.imshow((p**2).T, origin="lower", cmap="plasma")
plt.colorbar()
plt.title("Énergie à la fin")
plt.show()




