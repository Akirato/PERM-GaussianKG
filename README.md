## Implementation for the paper: 

**Probabilistic Entity Representation Model for Reasoning over Knowledge Graphs**, Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian and Chandan Reddy, NeurIPS 2021.

## Requirements
```
torch==1.2.0
tensorboardX==1.6
```

## Run
To reproduce the results on FB15k-237, DRKG and NELL, the hyperparameters are set in `example.sh`.
```
bash run.sh
```

## Arguments:
```
--do_train : Boolean that indicates if model should be trained
--cuda : Boolean that indicates if cuda should be used
--do_valid : Boolean that indicates if model should use validation
--do_test : Boolean that indicates if model should be tested to log metrics
--data_path : Folder that contains train, test and validation 
--model : Use 2-dimensions or one dimension for the model
-n : Number of negative samples per positive sample
-b : Batch size for training
-d : Dimension of embeddings (should be equal to semantic vector dimensions)
-lr : Learning rate of the model
--max_steps : Max number of epochs
--cpu_num : number of CPUs
--test_batch_size : Batch size for testing
--center_reg : Regularization factor for center updates
--geo : Gaussian embeddings or Vec embeddings
--task : Tasks for training
--stepsforpath : Same as number of epochs
--offset_deepsets : Aggregation methods for offsets
--center_deepsets : Aggregation methods for centers
--print_on_screen : Output should print on screen
```

## Code details
```
dataloader.py - File to load data for the PERM models
model_gaussian.py - File with the model definition for the PERM model and baselines
main_gaussian.py - File to run the model for different experiments
```
