# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained to predict whether an individual's annual income exceeds $50,000 based on demographic and employment features from the U.S. Census Bureau
## Intended Use
The model is intended to demonstrate an end-to-end MLOps pipeline — training, evaluation, slice-based validation, API deployment, and CI/CD integration.

## Training Data
The model was trained on the UCI Adult Census Income dataset, publicly available with 
Features: 14 input features — 8 categorical, 6 continuous
Label: salary — binary: <=50K or >50K
and 32562 entries

## Evaluation Data
The model was evaluated on the held-out 20% test split (6,513 samples), using the same random seed as the train/test split to ensure reproducibility. The test set was processed using the encoder and label binary coder fitted on the training data only,
## Metrics
_Please include the metrics used and your model's performance on those metrics._
Metrics
The following classification metrics were used to evaluate the model. All metrics use the positive class (>50K) as the target.
MetricValue
Precision 0.7388
Recall 0.6470
F1 Score 0.6899

Slice-based evaluation was also performed.Full results are available in slice_output.txt

## Ethical Considerations
This model is trained on demographic data including race, sex, and
native-country and hence needs to be assessed for any possible bias in classification.
## Caveats and Recommendations
- The dataset reflects U.S. Census data from 1994 and does not represent currentincome distributions or labor market conditions.
- Class imbalance exists in the dataset (~76% <=50K, ~24% >50K), which
contributes to the gap between precision and recall.
- A more thorough hyperparameter search could improve performance
