# Entropy Calculation for Motion Data

This folder contains a program for calculating the entropy of the translations in 3D space. The entropy is computed from motion data stored in an HDF5 file and measures the variability. A higher entropy value shows a more unpredictable or varied motion, while a lower value suggests a more predictable motion on a scale from 0-1.

### Cool New Features:
- **Standard Deviation Usage**: The dataset also calculates the standard deviation of the entropies per demo and classifies this as either "Expert" or "Play" based on whether it's higher.
  - **"Expert"**: The dataset is classified as Expert if the entropies of the demos are consistent (low standard deviation < 0.1)
  - **"Play"**: The dataset is classified as Play if the entropies of the demos are highly varied (high standard deviation > 0.1)
The classification is used to reflect the consistency or variability of the motion, where Expert demos tend to have more consistent, predictable movements, and Play demos exhibit more variation in their movements.

## Requirements

- Python 3.x
- `numpy`
- `h5py`
- `scipy`
- `plotly`

You can install the required dependencies using pip:

```bash
pip install numpy h5py scipy plotly
--or--
pip install -r requirements.txt
```
### Computing Entropy

The compute_entropy function calculates the entropy of the differences in translation data between consecutive time steps. The entropy is based on a 3D histogram of data which provides a measure of the unpredictability of the motion.

    3D Histogram: A histogram is calculated for the data in 3D space using np.histogramdd.
    
    Probability Distribution: The histogram is flattened and filtered to remove zero probabilities.
    
    Entropy Calculation: The entropy is computed using the scipy.stats.entropy function, which is based on the Shannon entropy formula.
    
    Normalization: The entropy value is normalized by dividing it by the maximum possible entropy for a uniform distribution.
    
    Non-linear Scaling: To further exaggerate the differences in entropy, and make it human-readable a non-linear scaling function is applied (exponential scaling), ensuring entropy values are kept between 0 and 1.

The entropy function from scipy.stats is used to compute the Shannon entropy of the probability distribution. The Shannon entropy is defined as:

$$
H(X) = - \sum_i P(x_i) \log P(x_i)
$$
