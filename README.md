# Vibration Fault Detection - Machine Learning for Power BI

This project provides tools to analyze vibration data for bearing fault detection using machine learning techniques, with results optimized for visualization in Power BI.

## Overview

The system is designed to detect and classify three types of bearing conditions:
- **Normal**: Healthy bearing without defects
- **Inner Race Fault**: Defects on the inner ring of the bearing
- **Outer Race Fault**: Defects on the outer ring of the bearing

## Project Components

- **main_processing.py**: Main script for data processing and model training
- **power_bi_guide.md**: Guide for visualizing results in Power BI
- **generate_example_files.py**: Creates example output files for demonstration purposes

## Getting Started

### Prerequisites

This project requires Python 3.6+ and the following packages:
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- seaborn
- joblib

Install them using:
```
pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib
```

### Data Format

The system expects vibration data in CSV files with three columns representing X, Y, and Z acceleration axes. Files should be named according to this pattern:
- **XYZ_N(i).csv**: For normal condition samples
- **XYZ_IR(i).csv**: For inner race fault samples
- **XYZ_OR(i).csv**: For outer race fault samples

Where (i) is any identifier or number.

Alternatively, you can organize your data into subfolders named:
- Normal
- Inner Race Fault
- Outer Race Fault

### Running the Analysis

1. Run the main processing script:
   ```
   python main_processing.py
   ```

2. When prompted, enter the path to your vibration data directory.

3. The script will process the data and generate output files in the `power_bi_data` folder.

4. Use these files in Power BI according to the guide in `power_bi_guide.md`.

### Generating Example Files

To create example output files (without real data) for demonstration purposes:

```
python generate_example_files.py
```

This will generate example CSV files in the `power_bi_example_files` folder that show the expected structure of the output files.

## Feature Extraction

The system extracts the following time-domain features from vibration signals:
- **RMS (Root Mean Square)**: Measure of the signal energy
- **Standard Deviation**: Measure of signal variability
- **Peak-to-Peak**: Difference between the maximum and minimum values
- **Skewness**: Measure of asymmetry of the signal distribution
- **Kurtosis**: Measure of the "tailedness" of the signal distribution

These features are calculated for each axis (X, Y, Z).

## Machine Learning Models

The system supports two machine learning models:
- **Random Forest**: Ensemble method using multiple decision trees
- **Support Vector Machine**: Powerful classification method

## Power BI Integration

For details on using the generated data in Power BI, see `power_bi_guide.md`.