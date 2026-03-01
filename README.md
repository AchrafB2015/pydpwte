## pydpwte

**Deep Parsimonious Weibull Time-to-Event with PyTorch**

`pydpwte` is a Python package for survival analysis and time-to-event prediction with PyTorch.  
It implements **DPWTE**, a deep-learning model that learns a parsimonious finite mixture of Weibull distributions for individual survival times.

This work was done during my PhD on survival analysis and is based on two approaches:

- **DPWTE**: Bennis et al., *“DPWTE: A Deep Learning Approach to Survival Analysis Using a Parsimonious Mixture of Weibull Distributions”*, ICANN 2021 (RANK B). [[Springer link](https://link.springer.com/chapter/10.1007/978-3-030-86340-1_15)]
- **DeepWeiSurv**: Bennis et al., *“Estimation of Conditional Mixture Weibull Distribution with Right Censored Data Using Neural Network for Time-to-Event Analysis”*, PAKDD 2020 (RANK A). [[Springer link](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_53)]

The current codebase exposes the DPWTE model and training pipeline; DeepWeiSurv is the conceptual precursor and will be aligned in the same package structure.

### Features

- **Parsimonious Weibull mixtures** for individual survival distributions (DPWTE)
- **Sparse Weibull Mixture layer** to automatically select a small set of mixture components
- **Support for right-censored data** through a likelihood-based loss
- **Cross-validation utilities** with concordance index evaluation
- **Ready-to-run dataset helpers** for METABRIC (and a template for SUPPORT2)

### Installation

- **From source**

```bash
git clone https://github.com/<your-username>/pydpwte.git
cd pydpwte
pip install -r requirements.txt
pip install -e .
```

Make sure you have a compatible PyTorch version installed for your hardware (CPU or GPU). You can also follow the official [PyTorch installation instructions](https://pytorch.org/) and then install only the non-PyTorch dependencies from `requirements.txt`.

### Quickstart (METABRIC example)

```python
import torch
from pydpwte.dpwte.dpwte import dpwte
from pydpwte.datasets.generate_data import generate_METABRIC_Data
from pydpwte.utils.cross_validation import CrossValidation

# Load and preprocess METABRIC data
X, Y = generate_METABRIC_Data()
n_cols = X.shape[1]

# Instantiate DPWTE model
model = dpwte(n_cols=n_cols, p_max=5, sparse_reg=True, lambda_reg=1e-4)

# Run 5-fold cross validation and compute C-index
cv = CrossValidation(
    model=model,
    p=5,
    inputs=X,
    targets=Y,
    optimizer_name='Adam',
    regularization_parameter=1e-4,
    w_th=0.1,
    n_epochs=1000,
    lr=1e-4,
)

c_indices = cv.five_fold_cross_validation()
print("Validation C-index per fold:", c_indices)
print("Mean C-index:", sum(c_indices) / len(c_indices))
```

For the SUPPORT2 dataset, download the CSV and place it as `pydpwte/datasets/support2.csv`, then call `generate_data_SUPPORT()` from `pydpwte.datasets.generate_data`.

### Docker usage

Build the image:

```bash
docker build -t pydpwte .
```

Start an interactive container:

```bash
docker run -it --rm pydpwte python
```

Inside the container you can run the same Python quickstart example to reproduce experiments.

### Citing

If you use this code in your research, please cite:

- DPWTE paper:  
  Bennis, A., Mouysset, S., Serrurier, M. (2021). *DPWTE: A Deep Learning Approach to Survival Analysis Using a Parsimonious Mixture of Weibull Distributions.* In: Farkaš, I., Masulli, P., Otte, S., Wermter, S. (eds) Artificial Neural Networks and Machine Learning – ICANN 2021. Lecture Notes in Computer Science, vol 12892. Springer, Cham. [https://doi.org/10.1007/978-3-030-86340-1_15](https://doi.org/10.1007/978-3-030-86340-1_15)

- DeepWeiSurv paper:  
  Bennis, A., Mouysset, S., Serrurier, M. (2020). *Estimation of Conditional Mixture Weibull Distribution with Right Censored Data Using Neural Network for Time-to-Event Analysis.* In: Lauw, H., Wong, R., Ntoulas, A., Lim, E., Ng, S., Pan, S. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2020. Lecture Notes in Computer Science, vol 12084. Springer, Cham. [https://doi.org/10.1007/978-3-030-47426-3_53](https://doi.org/10.1007/978-3-030-47426-3_53)

