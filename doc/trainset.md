# ML Sets

```python3 src/test_val_train.py```
```root_dir = '/media/igofed/SSD_1T/AI4CI/FULLDATASET/FULLDATASET'```
```csv_file = "annotation.csv"```
By default it is split to 80% -train 10% val, 10% test

<img src="train_test_val.png" width="400">

## The training Set

It is the set of data that is used to train and make the model learn the hidden features/patterns in the data. In each epoch, the same training data is fed to the NN architecture repeatedly, and the model continues to learn the features of the data. The training set should have a diversified set of inputs so that the model is trained in all scenarios and can predict any unseen data sample that may appear in the future.

## The validation Set

The validation set is a set of data, separate from the training set, that is used to validate our model performance during training. This validation process gives information that helps us tune the model’s hyperparameters and configurations accordingly. It is like a critic telling us whether the training is moving in the right. The validation set is a set of data, separate from the training set, that is used to validate our model performance during training. This validation process gives information that helps us tune the model’s hyperparameters and configurations accordingly.

## The test Set

The test set is a separate set of data used to test the model after completing the training.

### Variant 1

|Dataset     |Class 0| Class 1 | Train |Test | Validation |
|------------| :-- | :-- | :-- |   :-- |   :-- |   :-- |
|desktop     |     |     |     |       |       |       |
|mobile      |     |     |     |       |       |       |

### Variant 2

|Dataset     |Class 0| Class 1 | Train |Test | Validation |
|------------| :-- | :-- | :-- |   :-- |   :-- |   :-- |
|desktop     |     |     |     |       |       |       |
|mobile      |     |     |     |       |       |       |
|berlin      | 100 |     |     |       |       |       |
|munich      | 100 |     |     |       |       |       |
|mainz       | 100 |     |     |       |       |       |
|zurich      | 100 |     |     |       |       |       |

### Variant 3

|Dataset     |Class 0| Class 1 | Train |Test | Validation |
|------------| :-- | :-- | :-- |   :-- |   :-- |   :-- |
|desktop     |     |     |     |       |       |       |
|mobile      |     |     |     |       |       |       |
|roadway     |     |     |     |       |       |       |
|berlin      | 100 |     |     |       |       |       |
|munich      | 100 |     |     |       |       |       |
|mainz       | 100 |     |     |       |       |       |
|zurich      | 100 |     |     |       |       |       |

### Variant 4

|Dataset     |Class 0| Class 1 | Train |Test | Validation |
|------------| :-- | :-- | :-- |   :-- |   :-- |   :-- |
|desktop     |     |     |     |       |       |       |
|mobile      |     |     |     |       |       |       |
|roadway     |     |     |     |       |       |       |
|eu2013      |     |     |     |       |       |       |
|berlin      | 100 |     |     |       |       |       |
|munich      | 100 |     |     |       |       |       |
|mainz       | 100 |     |     |       |       |       |
|zurich      | 100 |     |     |       |       |       |
