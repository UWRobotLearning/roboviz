import h5py
import numpy as np
import plotly.graph_objs as go
from scipy.stats import entropy

def load_data_from_hdf5(file_path, demo_name, data_type='states', obs_type='obs'):
    with h5py.File(file_path, 'r') as f:
        data_path = f'data/{demo_name}/{obs_type}/{data_type}'
        if data_path not in f:
            raise KeyError(f"Key '{data_type}' not found at {data_path}.")
        states = f[data_path][:]
        return states

def extract_translation_data(states):
    return states[:, :3]  # since th first 3 columns are X, Y, Z translations
'''
This function computes the entropy using Shannon Entropy calculation from scipy.stats

**Histogram Calculation**: calculates a 3D histogram of the data with 30 bins along each dimension (X, Y, Z). The `density=True` option ensures the histogram is normalized to represent a probability distribution.

 **Flattening and Filtering**: The histogram is flattened, and zero probabilities (where the data doesn't occur) are removed to avoid taking the potential of getting an undefined value.

**Entropy Calculation**: The entropy of the probability distribution (`prob_dist`) is calculated using the `entropy()` function, which measures the uncertainty in the distribution.
 **Normalization**: The entropy is normalized by dividing it by the maximum possible entropy (`max_entropy`) for a uniform distribution in 3D space. The maximum entropy is calculated as 

**Non-linear Scaling**: To exaggerate the differences in entropy values and improve separation, the normalized entropy is scaled exponentially (`np.exp(normalized_entropy) - 1`), then clipped to ensure the value remains between 0 and 1. 

'''
def compute_entropy(data, bins=30):
    # Calculate the histogram of the data in 3D space
    hist, edges = np.histogramdd(data, bins=bins, density=True)

    # Flatten the histogram and filter out zero probabilities
    prob_dist = hist.flatten()
    prob_dist = prob_dist[prob_dist > 0]  # Avoid log(0) so there are no undefined values

    # Compute entropy using the probability distribution
    entropy_value = entropy(prob_dist)

    # Normalize entropy between 0 and 1 by dividing by log of the number of bins
    max_entropy = np.log(bins ** 3)  # Maximum entropy for uniform distribution in 3D bins
    normalized_entropy = entropy_value / max_entropy

    # Apply non-linear scaling to increase separation
    # For extra human readibility I added an exponential scaling function to exaggerate differences
    scaled_entropy = np.exp(normalized_entropy) - 1  # Shift to keep values between 0 and 1
    scaled_entropy = np.clip(scaled_entropy, 0, 1)  # Ensure it stays within [0, 1]

    return scaled_entropy

'''
This function computes the entropy of the difference between in the demo's translation data.

**Difference Calculation**: `np.diff(demo_translations, axis=0)` computes the difference between each consecutive pair of translation points. The resulting array has one less row than `demo_translations`, since it contains the difference between consecutive translations.

**Entropy Calculation**: The entropy of the computed vectors is then calculated by passing the difference data to `compute_entropy()`

30 bins is an estimated number and needs to be reconsidered for larger scale visualizations
'''

def compute_diff_entropy(demo_translations, bins=30):
    # Compute the difference between consecutive translations
    differences = np.diff(demo_translations, axis=0)  # Shape will be (n-1, 3)
    
    # Compute entropy of the differences between translations (which is a vector)
    return compute_entropy(differences, bins)

def classify_dataset(std_entropy, threshold=0.10):
    # Classify based on the standard deviation of the demo entropies
    if std_entropy <= threshold:
        return "Expert"
    else:
        return "Play"

def create_3d_overlay_plot(all_translations, all_demo_names, entropy_values, overall_entropy, std_entropy, classification):
    traces = []
    for i, translations in enumerate(all_translations):
        x, y, z = translations[:, 0], translations[:, 1], translations[:, 2]
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            name=f'Demo {all_demo_names[i]} (Entropy: {entropy_values[i]:.2f})',
            line=dict(width=3)
        )
        traces.append(trace)
    
    entropy_text = f'Overall Entropy: {overall_entropy:.2f}'
    std_diff_text = f'Standard Deviation: {std_entropy:.2f}'
    classification_text = f'Classification: {classification}'
    
    annotation = dict(
        text=f"{entropy_text}<br>{std_diff_text}<br>{classification_text}",
        showarrow=False,
        x=1, y=1,
        xref='paper', yref='paper',
        font=dict(size=14, color='red')
    )
    layout = go.Layout(
        title="Translation Data for Lampshade Demos",
        scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')),
        annotations=[annotation]
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def main():
    hdf5_file_path = 'play_pushing.hdf5'
    # hdf5_file_path = '/mmfs1/home/jgashler/play_pushing.hdf5' # full path
    # play_pushing.hdf5
    # expert_lampshade2_demos.hdf5
    data_type = 'states'
    obs_type = 'obs'
    
    with h5py.File(hdf5_file_path, 'r') as f:
        demos = list(f['data'].keys())
        all_translations, all_demo_names, entropy_values = [], [], []
        
        for demo_name in demos:
            try:
                states = load_data_from_hdf5(hdf5_file_path, demo_name, data_type, obs_type)
                translations = extract_translation_data(states)
                all_translations.append(translations)
                all_demo_names.append(demo_name)
            except KeyError as e:
                print(f"Skipping {demo_name}: {e}")
        
        # Compute entropy for each demo based on its difference from the average
        for translations in all_translations:
            entropy_values.append(compute_diff_entropy(translations))

        overall_entropy = np.mean(entropy_values)
        
        # Compute the standard deviation of entropy values
        std_entropy = np.std(entropy_values)
        
        # Classify the dataset as Expert or Play based on the standard deviation
        classification = classify_dataset(std_entropy)
        
        # Create the 3D plot with classification
        create_3d_overlay_plot(all_translations, all_demo_names, entropy_values, overall_entropy, std_entropy, classification)

if __name__ == "__main__":
    main()
