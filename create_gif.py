import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = "datasets/pems_bay/pems_bay.h5"
speed_df = pd.read_hdf(path)
speed_vals = speed_df.values

df = pd.read_csv("datasets/pems_bay/sensor_locations_bay.csv", header=None)


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, lat, long, speed_vals):
        self.num_frames = len(speed_vals)
        self.lat = lat
        self.long = long
        self.c_list = speed_vals

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=5,
            frames=range(self.num_frames),
            init_func=self.setup_plot,
            blit=True,
        )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        self.scat = self.ax.scatter(self.long, self.lat, c=self.c_list[0], s=100)
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return (self.scat,)

    def update(self, i):
        self.scat.set_array(self.c_list[i])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return (self.scat,)


a = AnimatedScatter(df[1], df[2], speed_vals)
plt.title("Traffic speed evolution")
plt.colorbar(a.scat)
plt.xlabel("Longitude")
plt.ylabel("Latitude")

writer = animation.PillowWriter()
# a.ani.save('scatter.gif', writer=writer)

plt.show()
