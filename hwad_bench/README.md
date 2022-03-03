```mermaid
flowchart LR
train.csv --> read_annotations((read_annotations))
read_annotations --> train_annotations
train_annotations --> cleansing((cleansing)) --> cleaned_annotations
```
