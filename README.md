# NuDIT v1.0

## Transforming Numerical Data to Images for Deep Networks.

Brief description of the project, its purpose, and its features.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Programming Language:** MATLAB

## Usage

Follow these steps to run the project:

### Phase #1: Initialize the NuDIT

1. **Prepare the Tabular Dataset**

   Place your dataset in the `TabularDatasets` directory and name it `RiceMSCDataset.csv`.

   ```matlab
   tabularFile = 'TabularDatasets\RiceMSCDataset.csv';

2. **Create an Instance of the NuDIT**

   Initialize the NuDIT object with your dataset file.

   ```matlab
   nudit = NuDIT(tabularFile);

### Phase #2: Transform Numerical Data to Images
Transform tabular data to images

1. **Transform numerical data into images suitable for deep learning models.**

   ```matlab
   nudit.numToImgTransform();

### Phase #3: Apply k-fold Cross-Validation
Set Parameters

1. **Define the image size and number of folds for cross-validation. For example, set the image size to 32 pixels and use 5 folds.**

   ```matlab
   imageSize = 32;
   kfold = 5;
   Prepare Data for Cross-Validation

2. **Resize images and split the data into k folds for cross-validation.**

   ```matlab
   nudit.prepareData(imageSize, kfold);

### Phase #4: Run DAG-Net
Train the Network

1. **Execute the DAG-Net network with a specified number of epochs.**

   ```matlab
   epochs = 10;
   result = nudit.runNetwork(epochs);

### Results
The results will be saved in the output directory specified in your script. Check this directory for the results and any log files.

### Troubleshooting
If you encounter issues, verify the following:

- The dataset file exists in the `TabularDatasets` directory.
- All necessary packages and toolboxes are installed and properly configured.

### Contributing
For contributions, please follow the guidelines in the CONTRIBUTING.md file.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact
For further assistance, please contact:

- **Email:** aelen@bandirma.edu.tr
- **GitHub:** [abdullahelen](https://github.com/abdullahelen)
