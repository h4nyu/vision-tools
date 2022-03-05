```mermaid
flowchart LR
train.csv --> read_annotations((read_annotations))
read_annotations --> train_annotations
train_annotations --> cleansing((cleansing)) --> cleaned_annotations
pred-r7-st.csv --> read_csv((read_csv)) --> train_box_annotations
cleaned_annotations --> create_croped_dataset0((create_croped_dataset)) --> train_croped_annotations
train_box_annotations --> create_croped_dataset0
train-fold0.csv --> read_csv0((read_csv)) --> fold_0_train
val-fold0.csv --> read_csv1((read_csv)) --> fold_0_val
train_croped_annotations --> train0((train_convnext_fold_0))
fold_0_train --> train0
fold_0_val --> train0
pred-test.csv --> read_csv2((read_csv)) --> test_box_annotations
test_box_annotations --> create_croped_dataset1((create_croped_dataset)) --> test_croped_annotations
```
