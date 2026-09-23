# Finding Donors for CharityML

A supervised learning project comparing three classifiers to identify people earning
more than $50,000 annually, using income as a proxy for potential charitable donors.

## Project resources

- [Executed notebook](finding_donors/finding_donors.ipynb): analysis, model comparisons, and answers to all eight project questions.
- [HTML report](finding_donors/report.html): download and open locally to view the complete report.
- [Reproduction instructions](finding_donors/README.md).
- [Computed metrics](finding_donors/analysis_metrics.json).
- [Python requirements](finding_donors/requirements.txt).
- [Optional competition predictions](finding_donors/kaggle_submission.csv): generated locally, not submitted or scored.

## Approach

1. Explore 45,222 labeled census records and establish naive baselines.
2. Demonstrate the required 103-column one-hot encoding and encode income as a binary label.
3. Fit imputation, scaling, and modeling schemas on training data only.
4. Compare Gaussian Naive Bayes, a decision tree, and gradient boosting on 1%, 10%, and 100% of training data.
5. Select the model family using training-only cross-validation and tune it with GridSearchCV.
6. Compare the full model with a model using its five most important features.

The assignment-visible encoding contains 103 columns. The separate model preprocessing
learns 102 columns from this training split because one rare category occurs only in the
holdout. Cross-validation refits preprocessing inside each fold.

## Results

The selected model is GradientBoostingClassifier with 200 trees and maximum tree depth 3.
Results use an 80/20 stratified split with random_state=0.

| Holdout metric | Untuned | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.863903 | 0.869099 |
| F0.5 | 0.746076 | 0.754315 |
| F1 | 0.692788 | 0.710230 |
| Precision | 0.786402 | 0.786876 |
| Recall | 0.619090 | 0.647190 |

F0.5 emphasizes precision to help reduce unnecessary mailings. F1 is also reported to
match the supplied rubric. Income predictions do not establish donation propensity,
donation revenue, or suitability for deployment on contemporary California residents.

## Run locally

Tested with Python 3.12.14. Create and activate a virtual environment, then run:

```sh
cd finding_donors
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 finding_donors.ipynb
python -m jupyter nbconvert --to html --output report.html finding_donors.ipynb
```

Fixed seeds and pinned packages support reproducibility. Timings vary between runs;
written timing comparisons describe the saved execution.

## Attribution

Based on the [Udacity Supervised Learning starter project](https://github.com/udacity/cd0025-supervised-learning).
The original `visuals.py` remains unmodified. The upstream license is preserved in
[LICENSE.txt](LICENSE.txt); no replacement license is applied to the starter materials.
