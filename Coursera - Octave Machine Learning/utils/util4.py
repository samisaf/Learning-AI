import numpy as np
import matplotlib.pyplot as plt


def displayData(X, y, plot_size=10, fig_size=5):
    indices = np.random.choice(len(X), plot_size*plot_size, replace=False)
    labels = y[indices]
    selected_images = X[indices]
    fig, axes = plt.subplots(plot_size, plot_size, figsize=(fig_size, fig_size))
    for i, ax in enumerate(axes.flat):
        rotated_image = np.rot90(np.fliplr((selected_images[i].reshape(20, 20))))
        ax.imshow(rotated_image.reshape(20, 20), cmap='gray')
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return labels.reshape(plot_size, plot_size)