import numpy as np
from scipy.spatial import distance
import distclassipy as dcpy
import matplotlib.pyplot as plt

def get_metric_name(metric):
    if callable(metric):
        metric_str = metric.__name__
    else:
        metric_str = metric
    return metric_str.title()

def visualize_distance(metric: str):
    metric_str = metric

    ref_point = np.array([5, 5])
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    
    metric_fn_, metric = dcpy.classifier.initialize_metric_function(metric_str)

    distances = distance.cdist(ref_point.reshape(1, -1), points, metric=metric)
    distances = distances.reshape(X.shape)
    
    fig, ax = plt.subplots(1,1, figsize=(4,4), sharex=False, sharey=False)
    im = ax.imshow(
        distances,
        cmap="Blues",
        interpolation="nearest",
        extent=[0, 10, 0, 10],
        origin="lower",
    )

    ax.scatter(5, 5, s=50, color="k", marker="x")
    cs = ax.contour(X, Y, distances, colors="k")
    ax.clabel(cs, inline=True, fontsize=10, colors="k")
    ax.set_title(f"{metric_str.title().replace('_','-')}")
    
    return ax