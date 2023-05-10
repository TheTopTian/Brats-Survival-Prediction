# Brain Tumor Challenge: Survival Prediction

In this project, we aim to predict the survival of patients with brain tumors using machine learning techniques.

## Dataset

We used the dataset from the challenge BraTS 2020 (https://www.med.upenn.edu/cbica/brats2020/data.html), it has both segmentation and survival prediction task inside. 

***Segmentation***: The dataset contains several model scans with NIFTI files (.nii.gz) and describe a) native(**T1**) and b) post-contrast T1-weighted (**T1Gd**), c) T2-weighted (**T2**), and d) T2 Fluid Attenuated Inversion Recovery (**T2-FLAIR**) volumns. All the imaging datasets have been segmented manually.

<p align="center">
  <img src="./images/Brain and label.png" alt="brain and label" title="brain and label" width="700" height="auto">
</p>
  
***Survival***: The overall survival (OS) data, defined in days, are included in a comma-separated value (.csv) file with correspondences to the pseudo-identifiers of the imaging data. The .csv file also includes the age of patients, as well as the resection status.

## Features Extraction

Correct extraction of features is the basis for accurate prediction. We used two different methods to extract the features:

***PyRadiomics***: PyRadiomics is a package which can automatically extract radiomics features in medical images. The lack of standardization of feature definitions has been shown to have a substantial impart on the reliability of radiomic data and this package was designed to solve it. The features extracted with it could be divided into 7 different categories: the distribution of voxel intensities (First Order), 3D features of size and shape (Shape Features), gray level co-occurrence matrix features (GLCM), gray level size zone matrix (GLSZM), gray level run length matrix features (GLRLM), neighbouring gray tone difference matrix features (NGTDM) and gray level dependence matrix features (GLDM).

***Radiomic features***: On the other hands, we take the volumn ratio of necrosis, edema and active tumor relative to the whole brain, the position from the center of the brain and their relative coordinates to the center of the brain were selected respectively. In addition, we also added the parameters of tumor's surface. We think that the uneven surface would increase the difficulty of the operation and affect the survival time of the patient.

## Features Selection

After combining the extracted features based on radiomics and the manual defined method, there were 126 features for one single case. Many of them actually didn’t have high relationship with the survival days, most of them were just noise in the final regression. We used the SpearmanR to show the relationship between the features and survival day. Set a threshold to choose the parameter and lower the outliers.

<p align="center">
  <img src="./images/Feature.png" alt="feature selection" title="feature selection" width="700" height="auto">
</p>
  
## Regression Method

There are many papers have already proved that the simple machine learning regression methods like random forest or MLP have better performance compared with the deep learning methods. We do a comparison among all the normal regression methods and find that the pipeline which is combined with standard scaler and SGD regressor has the best performance among all of them.

<p align="center">
  <img src="./images/Regression_Methods.png" alt="regression methods" title="regression methods" width="700" height="auto">
</p>
  
## Framework of Survival

The whole framework of the survival prediction task looks like that:

<p align="center">
  <img src="./images/Framework.png" alt="framework" title="framework" width="700" height="auto">
</p>
  
## Results

We tried different threshold of the filter. It was clear that the threshold $>|0.10|$ showed the best result ($0.621$) among all of them. After filtering, the features of each case were reduced from 126 to 28 which also sped up the computation. The accuracy was far higher than another two related paper also extracted radiomic features in BraTS 2020.


|**Different Thresholds**|**Accuracy**|**MSE**|**SpearmanR**|
| --- | --- | --- | --- |
|without|0.379|$1.079\times10^{10}$|0.335|
|$>\|0.08\|$|0.586|$2.056\times10^8$|0.518|
|$>\|0.10\|$|**0.621**|$3.641\times10^7$|0.502|
|$>\|0.12\|$|0.517|$5.775\times10^7$|0.417|
|$<-0.01$|0.552|$5.167\times10^7$|0.480|
|$<-0.05$|0.379|$1.516\times10^8$|0.050|
|$<-0.10$|0.586|$1.406\times10^5$|0.537|

We also tried to use different input of the images since there were 4 types of images in the training dataset. It was also shown that the input of $T_1$ image had better result of accuracy compared with other images.

|**Different Input Modalities**|**Accuracy**|**MSE**|**SpearmanR**|
| --- | --- | --- | --- |
|$T_1$|**0.621**|$3.641\times10^7$|0.502|
|$T_{1ce}$|0.448|$2.977\times10^8$|0.433|
|$T_2$|0.483|$1.118\times10^5$|0.233|
|$T_{2Flair}$|0.345|$2.470\times10^5$|0.127|
|Combination of 4 modalities|0.517|$7.183\times10^6$|0.494|


## Conclusion

In the survival part, we achieved a promising accuracy ($0.621$). It was clear that the filter depended on the Spearman's Rank had an outstanding performance. The framework without the filter could only reach the accuracy of $0.379$ which was far lower. The reason why we always kept the negative part in the threshold was that the most import feature was age which showed $-0.419$ SpearmanR value with the survival days. And also due to some obvious common sense, such as tumor size and tumor surface area were all negatively correlated with survival time. According to the result, the accuracy with threshold $>|0.10|$ ($0.621$) was higher than the accuracy with threshold only $<-0.10$ ($0.586$), because some of features were also positively correlated with survival time like the tumor's distance and offset from the brain center. Setting the threshold is very important for a good result. It also needs researchers to have relevant medical knowledge of tumors and as we learn more medically about tumors, perhaps we can manually add more important parameters.

Different input modalities also leaded to different accuracy, it was clear that extract features only on $T_1$ image had far higher accuracy. We thought the reason was the $T_1$ image had stronger color contrast compared with the other three images, so that it was much more suitable to do the automatic features extraction. 

BraTS 2020 is a challenge with small dataset, only 118 cases can be used for the survival prediction. And the distribution of short-term (44 cases), mid-term (28 cases) and long-term (46 cases) patients is not even. So during training our regression model, the weight of the mid-term cases might be lower compared with the other two. The best prediction result we got were also mostly categorized as short-term and long-term. Therefore we thought that if the BraTS challenge could adjust the number of these 3 groups closer in the training dataset, it might improve the final accuracy.

We also tried to input all of the cases which were provided the feature of survival days. Despite the fact that half of them didn't belong to GTR, we could have 235 training cases for training dataset, far more than 118 cases at first. But at last we received a far worse result (only 0.414 for accuracy). It proved that the type of disease did have a huge impact on how long the patients could survive. Correct classification of the diseases at the beginning would have great help in survival task.

There was another paper proved that the survival days of the patients were not only closely related to the tumors' condition, but also had a great relationship with the patients' functional status at diagnosis. Hope that in the future the BraTS challenges can simultaneously provide other physical information of patients to achieve more accurate and robust survival predictions.

Our future work is to try this method on other medical dataset to prove that it is a robust way with high accuracy and using the SpearmanR-based filter can improve the prediction results. If it does have robustness on other dataset, we can specially study which features are closely related to the death based on the SpearmanR value, so as to build specific databases and models. In the future, only the relevant features need to be input to automatically determine the survival.
