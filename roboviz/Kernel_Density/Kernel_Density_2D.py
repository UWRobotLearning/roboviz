import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def load_data_from_hdf5(file_path, demo_name, data_type='states', obs_type='obs'):
    with h5py.File(file_path, 'r') as f:
        data_path = f'data/{demo_name}/{obs_type}/{data_type}'
        if data_path not in f:
            raise KeyError(f"Key '{data_type}' not found at {data_path}.")
        states = f[data_path][:]
        print(f"Loaded {data_type} from {obs_type} with shape: {states.shape} for {demo_name}")
        return states

def extract_projection(states, axes=(0, 1), negate=(False, False)):
    proj = states[:, list(axes)].copy()
    for i, neg in enumerate(negate):
        if neg:
            proj[:, i] *= -1
    return proj

def compute_kde_2d(data, bandwidth=0.1):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(data)
    return kde

def plot_density(ax, data, view_name, contour=None):
    kde = compute_kde_2d(data)

    x_min, y_min = data.min(axis=0)
    x_max, y_max = data.max(axis=0)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_samples = np.vstack([xx.ravel(), yy.ravel()]).T

    log_dens = kde.score_samples(grid_samples)
    dens = np.exp(log_dens).reshape(xx.shape)

    contour = ax.contourf(xx, yy, dens, levels=50, cmap='viridis')
    ax.contour(xx, yy, dens, levels=10, colors='white', linewidths=0.5)
    ax.scatter(data[:, 0], data[:, 1], s=2, c='black', alpha=0.3)

    ax.set_title(view_name)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    return contour

def fig_to_base64(fig):
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def main(file_path):
    # Diff views
    projections = {
        "Top View (X vs Y)": (0, 1, (False, False)),
        "Front View (X vs Z)": (0, 2, (False, False)),
        "Left View (Y vs Z)": (1, 2, (False, False)),
        "Right View (Y vs -Z)": (1, 2, (True, False)),
    }

    data_type = 'states'
    obs_type = 'obs'

    with h5py.File(file_path, 'r') as f:
        demos = list(f['data'].keys())
        all_states = []

        for demo_name in demos:
            try:
                states = load_data_from_hdf5(file_path, demo_name, data_type, obs_type)
                all_states.append(states)
            except KeyError as e:
                print(f"Skipping {demo_name}: {e}")

    full_data = np.vstack(all_states)
    images = []

    # 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    contour = None

    # Generate the views
    for i, (view_name, (axis_x, axis_y, negate_flags)) in enumerate(projections.items()):
        projected = extract_projection(full_data, axes=(axis_x, axis_y), negate=negate_flags)
        contour = plot_density(axs[i], projected, view_name, contour)

    # Legend as a colorbar display
    cbar = fig.colorbar(contour, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Density')

    with open("static/plot.html", "w") as f:
        f.write("<html><head><title>2D Density Views</title></head><body>\n")
        f.write("<table>\n")
        
        for i, (title, img) in enumerate(images):
            if i % 2 == 0:a
            f.write("<tr>\n")  # Start a new row for every 2 images
            f.write(f"<td style='padding: 10px;'>\n")
            f.write(f"<h3>{title}</h3>\n")
            f.write(f'<img src="data:image/png;base64,{img}" style="width: 100%; height: auto;">\n')
            f.write('</td>\n')
            if i % 2 == 1:
                f.write("</tr>\n")

        f.write("</table>\n")  # End table
        f.write("</body></html>\n")

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.show()

if __name__ == "__main__":
    file_path = sys.argv[1]
    main(file_path)
