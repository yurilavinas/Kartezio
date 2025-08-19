# Kartezio

[![Discord Channel](https://dcbadge.limes.pink/api/server/uwFwHyRxub)](https://discord.gg/KnJ4XWdQMK)
[![PyPI version](https://badge.fury.io/py/kartezio.svg)](https://badge.fury.io/py/kartezio)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

<div align="center">
  <h2>🧬 Evolutionary Design of Explainable Computer Vision Algorithms</h2>
  <p><strong>Cartesian Genetic Programming framework for automated, interpretable image processing pipelines</strong></p>
</div>

---

## 🎯 What is Kartezio?

**Kartezio** is a revolutionary Cartesian Genetic Programming (CGP) framework that automatically evolves **transparent, interpretable computer vision algorithms** from just a few examples. Originally developed for biomedical image segmentation and [published in Nature Communications](https://www.nature.com/articles/s41467-023-42664-x), Kartezio represents a paradigm shift from black-box deep learning to explainable AI.

Unlike traditional machine learning approaches that require massive datasets and GPU clusters, Kartezio evolves optimal image processing pipelines using evolutionary computation on a single CPU with minimal training data.

### 🏆 Key Achievements
- **Nature Communications Publication**: Proven effectiveness in biomedical applications
- **Few-Shot Learning**: Works with as little as 5-10 annotated examples
- **Zero GPU Requirement**: Runs efficiently on standard CPUs
- **Full Interpretability**: Every operation in the evolved pipeline is transparent and auditable

---

## ✨ Why Choose Kartezio?

### 🔬 **Explainable by Design**
Every evolved algorithm is a transparent sequence of computer vision operations (filters, morphology, thresholding). No black boxes, no hidden layers—just clear, auditable image processing steps.

### 🚀 **Few-Shot Learning**
Forget massive datasets. Kartezio evolves effective solutions from just a handful of annotated examples, making it perfect for specialized applications where data is scarce.

### 💡 **CPU-Only Execution**
No GPUs required. Kartezio runs efficiently on standard hardware, making it ideal for:
- Edge devices and embedded systems
- Resource-constrained environments  
- Real-time applications
- Educational settings

### 🧩 **Modular & Extensible**
Built with a component-based architecture that allows easy customization:
- Add custom image processing primitives
- Define domain-specific fitness functions
- Extend endpoints for different output types
- Integrate with existing computer vision workflows

### 🌍 **Broad Applicability**
While proven in biomedical imaging, Kartezio excels across domains:
- **Medical & Life Sciences**: Cell segmentation, pathology analysis, microscopy
- **Industrial Vision**: Quality control, defect detection, manufacturing
- **Remote Sensing**: Satellite imagery, aerial photography, environmental monitoring
- **Robotics**: Object detection, navigation, manipulation
- **Security & Surveillance**: Anomaly detection, monitoring systems

---

## 🚀 Quick Start

### Installation

Install Kartezio with pip:

```bash
pip install kartezio
```

For development or advanced features:

```bash
# With all optional dependencies
pip install kartezio[dev,viz]

# From source
git clone https://github.com/your-org/kartezio.git
cd kartezio
pip install -e .
```

### Your First Kartezio Model

Here's a complete example that evolves a cell segmentation pipeline:

```python
from kartezio.core.endpoints import EndpointThreshold
from kartezio.core.fitness import IoU  
from kartezio.evolution.base import KartezioTrainer
from kartezio.primitives.matrix import default_matrix_lib
from kartezio.utils.dataset import one_cell_dataset

# 1. Set up components
n_inputs = 1
libraries = default_matrix_lib()    # Library of image operations
endpoint = EndpointThreshold(128)   # Binary output via thresholding
fitness = IoU()                     # Intersection over Union metric

# 2. Create and configure the evolutionary trainer  
model = KartezioTrainer(
    n_inputs=n_inputs,
    n_nodes=n_inputs * 10,          # 10 processing nodes
    libraries=libraries,
    endpoint=endpoint,
    fitness=fitness,
)
model.set_mutation_rates(node_rate=0.05, out_rate=0.1)

# 3. Load your data (or use the included example dataset)
train_x, train_y = one_cell_dataset()  # Example: cell images + masks

# 4. Evolve the algorithm (100 generations)  
elite, history = model.fit(100, train_x, train_y)

# 5. Evaluate performance
score = model.evaluate(train_x, train_y)
print(f"Final IoU Score: {score:.3f}")

# 6. Export as standalone Python code
model.print_python_class("CellSegmenter")
```

That's it! Kartezio has evolved a complete image processing pipeline tailored to your data.

---

## 📚 Core Concepts

### Architecture Overview

Kartezio uses a **component-based architecture** with four main types:

1. **Primitives**: Basic image operations (filters, morphology, arithmetic)
2. **Endpoints**: Output processing (thresholding, watershed, etc.)  
3. **Fitness Functions**: Performance metrics (IoU, AP, custom metrics)
4. **Libraries**: Collections of primitives organized by data type

### Component Registration System

Components are registered using decorators:

```python
from kartezio.core.components import Primitive, register
from kartezio.types import Matrix

@register(Primitive)
class CustomFilter(Primitive):
    def __init__(self):
        super().__init__([Matrix], Matrix, n_parameters=1)
    
    def call(self, x, args):
        kernel_size = args[0]
        return cv2.GaussianBlur(x[0], (kernel_size, kernel_size), 0)
```

### Evolution Process

1. **Initialization**: Random population of image processing graphs
2. **Evaluation**: Each individual processes training images  
3. **Selection**: Best performers survive based on fitness scores
4. **Mutation**: Modify operations, connections, and parameters
5. **Iteration**: Repeat until convergence or generation limit

---

## 🛠️ Advanced Usage

### Custom Fitness Functions

Define domain-specific evaluation metrics:

```python
from kartezio.core.components import Fitness, register
import numpy as np

@register(Fitness)  
class CustomMetric(Fitness):
    def evaluate(self, y_true, y_pred):
        # Your custom metric logic here
        return np.mean((y_true - y_pred) ** 2)  # Example: MSE
```

### Adding New Primitives

Extend Kartezio with domain-specific operations:

```python
@register(Primitive)
class AdvancedMorphology(Primitive):
    def __init__(self):
        super().__init__([Matrix], Matrix, n_parameters=2)
    
    def call(self, x, args):
        operation_type = args[0]  # 0: opening, 1: closing
        kernel_size = args[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (kernel_size, kernel_size))
        
        if operation_type == 0:
            return cv2.morphologyEx(x[0], cv2.MORPH_OPEN, kernel)
        else:
            return cv2.morphologyEx(x[0], cv2.MORPH_CLOSE, kernel)
```

### Multi-Objective Optimization

Balance multiple criteria simultaneously:

```python
from kartezio.core.fitness import MultiObjective

fitness = MultiObjective([
    IoU(),           # Segmentation accuracy
    Complexity(),    # Algorithm simplicity  
    Speed()          # Execution time
], weights=[0.7, 0.2, 0.1])
```

### Visualization and Analysis

Visualize evolved algorithms:

```python
from kartezio.easy import show_graph

# Display the evolved processing graph
graph = show_graph(model, only_active=True, jupyter=True)
graph.view()  # Opens in browser or Jupyter notebook
```

---

## 🔧 Development & Contribution

### Development Setup

```bash
git clone https://github.com/your-org/kartezio.git
cd kartezio

# Install with development dependencies
pip install -e .[dev]

# Or with uv (recommended)
uv sync --extra dev

# Run tests
uv run python tests/test_runner.py --quick
```

### Running Tests

Kartezio includes comprehensive test suites:

```bash
# Quick tests (recommended for development)
uv run python tests/test_runner.py --quick

# Security tests
uv run python scripts/dev.py security

# Core functionality tests
uv run python -m unittest tests.test_core_components

# All tests with our custom runner
uv run python tests/test_runner.py

# Using pytest (if preferred)
uv run python -m pytest tests/
```

### Code Quality

We maintain high code quality standards:

```bash
# Format code
uv run ruff format src/ tests/

# Lint code
uv run ruff check src/ tests/

# Development script (includes all quality checks)
uv run python scripts/dev.py format
uv run python scripts/dev.py lint
```

---

## 📖 Examples & Tutorials

### Available Examples

- **`examples/training/basic_trainer.py`**: Getting started with cell segmentation
- **`examples/training/advanced_trainer.py`**: Multi-objective optimization
- **`examples/components/create_primitive.py`**: Adding custom primitives
- **`examples/components/create_endpoint.py`**: Custom output processing

### Real-World Applications

1. **Biomedical Image Segmentation**
   - Cell counting and classification
   - Tissue analysis and pathology  
   - Microscopy image processing

2. **Industrial Quality Control**  
   - Defect detection in manufacturing
   - Surface inspection and analysis
   - Automated visual inspection

3. **Remote Sensing Applications**
   - Land use classification
   - Environmental monitoring
   - Agricultural analysis

---

## 📊 API Reference

### Core Classes

#### `KartezioTrainer`
Main evolutionary training interface.

```python
class KartezioTrainer:
    def __init__(self, n_inputs, n_nodes, libraries, endpoint, fitness)
    def fit(self, generations, train_x, train_y) -> Tuple[Genotype, List]
    def evaluate(self, test_x, test_y) -> float  
    def predict(self, x) -> List[np.ndarray]
    def print_python_class(self, class_name: str) -> None
```

#### Component Base Classes

- **`Primitive`**: Base class for image processing operations
- **`Endpoint`**: Base class for output processing  
- **`Fitness`**: Base class for evaluation metrics
- **`Library`**: Container for organized primitive collections

### Built-in Libraries

- **Matrix Library**: 50+ image processing primitives
- **Scalar Library**: Statistical and mathematical operations  
- **Vector Library**: Multi-dimensional data processing

---

## 🏥 Biomedical Applications

Kartezio was originally developed for biomedical image analysis and has proven effectiveness in:

### Cell Segmentation & Analysis
- Automated cell counting and classification
- Morphological analysis of cellular structures  
- Time-lapse microscopy analysis
- Multi-modal imaging integration

### Pathology & Diagnostics  
- Tissue segmentation for pathological analysis
- Automated screening and diagnosis assistance
- Quantitative pathology measurements
- Multi-scale histological analysis

### Medical Imaging
- Organ segmentation in medical scans
- Lesion detection and characterization  
- Image enhancement for clinical workflows
- Real-time surgical guidance

---

## 🌟 Research & Publications

### Primary Citation
If you use Kartezio in your research, please cite:

```bibtex
@article{cortacero2023evolutionary,
  title={Evolutionary design of explainable algorithms for biomedical image segmentation},
  author={Cortacero, K{\'e}vin and McKenzie, Brienne and M{\"u}ller, Sabina and Khazen, Roxana and Lafouresse, Fanny and Corsaut, Ga{\"e}lle and Van Acker, Nathalie and Frenois, Fran{\c{c}}ois-Xavier and Lamant, Laurence and Meyer, Nicolas and others},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={7112},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

### Additional Publications
For multimodal applications:

```bibtex
@inproceedings{de2024multimodal,
  title={Multimodal adaptive graph evolution},
  author={De La Torre, Camilo and Cortacero, K{\'e}vin and Cussat-Blanc, Sylvain and Wilson, Dennis},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference Companion},
  pages={499--502},
  year={2024}
}
```

---

## 🤝 Community & Support

### Getting Help

- **Discord Community**: [Join our Discord](https://discord.gg/KnJ4XWdQMK)
- **GitHub Issues**: [Report bugs or request features](https://github.com/your-org/kartezio/issues)
- **Email Support**: kevin.cortacero@protonmail.com


---

## ⚖️ License

Kartezio is available for **Non-Commercial and Academic use only** under a proprietary license. 

- ✅ **Permitted**: Research, education, evaluation, personal projects
- ❌ **Prohibited**: Commercial use, redistribution, modification for commercial purposes

For commercial licensing options, please contact: kevin.cortacero@protonmail.com

See the [LICENSE](LICENSE) file for complete terms and conditions.

---

## 🙏 Acknowledgments

Kartezio was developed with support from:

- **Research Institutions**: INSERM, CNRS, Université Toulouse Capitole, Université Paul Sabatier, ISAE Supaero
- **Funding Agencies**: INSERM, ERC  
- **Open Source Community**: Contributors and maintainers
- **Nature Communications**: For publishing our foundational research

---

<div align="center">
  <p><strong>Ready to evolve the future of explainable computer vision?</strong></p>
  <p>⭐ Star us on GitHub | 💬 Join our Discord </p>
</div>