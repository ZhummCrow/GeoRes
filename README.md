# GeoRes: Predicting Drug Resistance via Graph Neural Networks

This repository contains a Deep Learning framework designed to predict **antimicrobial drug resistance** by representing chemical compounds or biological entities as graphs. By leveraging **Graph Neural Networks (GNNs)**, the model captures complex structural dependencies that traditional descriptors often miss.

## 🚀 Getting Started

### Prerequisites

Ensure you have [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) installed.

```bash
# Clone the repository
git clone git@github.com:ZhummCrow/GeoRes.git
cd GeoRes

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

To run inference on new data:

```bash
python inference.py --dataset_path input.csv --model_path "./weights/seed1" --output_path ./output/seed1.csv
```

-----

## 📜 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{xxxx
}
```
<!-- 
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details. -->