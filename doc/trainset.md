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

### Variant 1 (2493 images)

Full dataset **desktop**, **mobile**

|Dataset     |Train Class0|Train Class1| Test Class0 |Test Class1 | Val Class0 | Val Class1 |
|------------| :-- | :-- | :-- |   :-- |   :-- |   :-- |
|Dataset 1_0     | 218    | 1776    |  22   |  228     |  22     |   227    |
|Dataset 1_1     | 214    | 1780    |  20   |  230     |  28     |   221    |

### Variant 2 (2893 images)

Full dataset **desktop**, **mobile**, **berlin**, **munich**, **mainz**, **zurich**

Full dataset **desktop**, **mobile**

|Dataset     |Train Class0|Train Class1| Test Class0 |Test Class1 | Val Class0 | Val Class1 |
|------------| :-- | :-- | :-- |   :-- |   :-- |   :-- |
|Dataset 2_0     | 525    | 1789    |  63   |  227     |  74     |   215    |
|Dataset 2_1     | 534    | 1780    |  62   |  228     |  66     |   223    |

### Variant 3 (3334 images)

Full dataset **desktop**, **mobile**, **roadway**, **berlin**, **munich**, **mainz**, **zurich**

|Dataset     |Train Class0|Train Class1| Test Class0 |Test Class1 | Val Class0 | Val Class1 |
|------------| :-- | :-- | :-- |   :-- |   :-- |   :-- |
|Dataset 3_0     | 521    | 2146    |  70   |  264     |  71     |   262    |
|Dataset 3_1     | 547    | 2120    |  60   |  274     |  55     |   278    |

### Variant 4 (4882 images)

Full dataset **desktop**, **mobile**, **roadway**, **eu2013**, **berlin**, **munich**, **mainz**, **zurich**

|Dataset     |Train Class0|Train Class1| Test Class0 |Test Class1 | Val Class0 | Val Class1 |
|------------| :-- | :-- | :-- |   :-- |   :-- |   :-- |
|Dataset 4_0     | 528    | 3378    |  70   |  419     |  64     |   424    |
|Dataset 4_1     | 520    | 3386    |  75   |  414     |  67     |   421    |
