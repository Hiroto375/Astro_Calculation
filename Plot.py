import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Load CSV
df = pd.read_csv("trajectory.csv")

# Get sorted step values
steps = sorted(df["step"].unique())

# Set plot range
xmin, xmax = df["x"].min(), df["x"].max()
ymin, ymax = df["y"].min(), df["y"].max()

margin_x = 0.1 * (xmax - xmin + 1e-6)
margin_y = 0.1 * (ymax - ymin + 1e-6)

fig, ax = plt.subplots(figsize=(8, 8))
scat = ax.scatter([], [], s=5)

ax.set_xlim(xmin - margin_x, xmax + margin_x)
ax.set_ylim(ymin - margin_y, ymax + margin_y)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("N-body simulation")
ax.set_aspect("equal", adjustable="box")

text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

def init():
    scat.set_offsets(np.empty((0, 2)))
    text.set_text("")
    return scat, text

def update(frame_step):
    frame_data = df[df["step"] == frame_step]
    coords = frame_data[["x", "y"]].to_numpy()
    scat.set_offsets(coords)
    text.set_text(f"step = {frame_step}")
    return scat, text

ani = FuncAnimation(
    fig,
    update,
    frames=steps,
    init_func=init,
    interval=50,
    blit=True,
    repeat=True
)

ani.save("nbody_animation.mp4", fps=20, dpi=150)
plt.show()