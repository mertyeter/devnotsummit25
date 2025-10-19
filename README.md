# HR Fairness Gender Bias Mitigation

This repository contains the source code and presentation materials for my session about "Neuro-Symbolic AI for Developers: Smarter Decisions in Software Systems" at Devnot Developer Summit 2025, held on October 18th, 2025.

## Project Overview

This project implements a bias-aware machine learning model that analyzes employer data to identify and reduce gender-based discrimination in hiring and employment practices. The model leverages [Fairlearn](https://fairlearn.org/)'s GridSearch with DemographicParity constraints to mitigate bias while maintaining predictive accuracy. It uses scikit-learn's LogisticRegression as the base estimator and evaluates fairness through MetricFrame, providing comprehensive insights into model performance across different demographic groups.

Key capabilities include:
- Training both standard (unmitigated) and bias-mitigated models for comparison
- Evaluating fairness metrics including selection rates and demographic parity
- Exporting models in both PyTorch (.pth) and ONNX (.onnx) formats for flexible deployment

## Features

- **Bias Detection**: Analyzes employer data for gender-based discrimination patterns
- **Bias Mitigation**: Implements Fairlearn's GridSearch with DemographicParity constraints to reduce bias
- **Model Export**: Provides both PyTorch (.pth) and ONNX (.onnx) model formats
- **Fair Predictions**: Maintains model accuracy while improving fairness metrics
- **Comparative Analysis**: Evaluates both standard and bias-mitigated models side-by-side

## Requirements

- Python 3.13+
- PyTorch
- pandas
- scikit-learn
- fairlearn
- skl2onnx (for converting scikit-learn models to ONNX)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd devnotsummit25
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch pandas scikit-learn fairlearn skl2onnx
```

## Usage

Run the main script to train and evaluate the bias-mitigated model:

```bash
python src/hr_fairness_gender.py
```

The script will:
1. Load employer data from `src/input/employers_data.csv`
2. Train a bias-aware neural network
3. Evaluate fairness metrics
4. Export trained models to `src/output/` directory

## Model Output

The trained models are saved in two formats:

- **PyTorch Model** (`bias_mitigated_model.pth`): Native PyTorch format for Python applications
- **ONNX Model** (`bias_mitigated_model.onnx`): Cross-platform format for deployment

**Note**: The `src/input/` and `src/output/` directories are excluded from version control via `.gitignore` as they contain data files and generated models.

## Data Format

The input CSV file should contain employer-related features with the following considerations:
- Include relevant employment features (education, experience, skills, etc.)
- Include protected attribute (gender) for bias analysis
- Include target variable (hiring decision, promotion, etc.)

## Sample Dataset

- Source: Employer Data (Kaggle) — https://www.kaggle.com/datasets/gmudit/employer-data

## Fairness Metrics

The model evaluates fairness using:
- Demographic parity
- Equal opportunity
- Equalized odds
- Disparate impact ratio

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PyTorch for deep learning capabilities
- Implements adversarial debiasing techniques from fairness ML research
- Uses [Fairlearn](https://fairlearn.org/) - an open-source toolkit for assessing and improving fairness in AI systems, including GridSearch for bias mitigation and MetricFrame for fairness evaluation
- Supports ONNX for cross-platform deployment

## References

- [Fairlearn Documentation](https://fairlearn.org/)
- [Fairlearn GitHub Repository](https://github.com/fairlearn/fairlearn)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ONNX Documentation](https://onnx.ai/)