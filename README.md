# Brain Tumor Challenge: Survival Prediction

In this project, we aim to predict the survival of patients with brain tumors using machine learning techniques.

## Dataset

We used the dataset from the challenge BraTS 2020 (https://www.med.upenn.edu/cbica/brats2020/data.html), it has both segmentation and survival prediction task inside. 

***Segmentation***: The dataset contains several model scans with NIFTI files (.nii.gz) and describe a) native(**T1**) and b) post-contrast T1-weighted (**T1Gd**), c) T2-weighted (**T2**), and d) T2 Fluid Attenuated Inversion Recovery (**T2-FLAIR**) volumns. All the imaging datasets have been segmented manually.

***Survival***: The overall survival (OS) data, defined in days, are included in a comma-separated value (.csv) file with correspondences to the pseudo-identifiers of the imaging data. The .csv file also includes the age of patients, as well as the resection status. N

## Methodology

We plan to use machine learning algorithms such as logistic regression, random forests, and support vector machines (SVM) to build predictive models that can accurately predict the survival of patients with glioblastoma multiforme. We will use Python and scikit-learn, a popular machine learning library, to preprocess the data, train the models, and evaluate their performance using cross-validation.

## Results

We will evaluate the performance of our models using various metrics such as accuracy, precision, recall, and area under the receiver operating characteristic (ROC) curve. We will also compare our models with the baseline models provided by the Brain Tumor Challenge.

## Conclusion

The ability to accurately predict the survival of patients with brain tumors can have a significant impact on their treatment and care. By applying machine learning techniques to the Brain Tumor Challenge dataset, we hope to contribute to the development of more effective and personalized treatments for patients with glioblastoma multiforme.
