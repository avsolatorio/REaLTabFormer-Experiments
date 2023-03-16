# Review Experiments

We document here the planned experiments that we will undertake for the ICML review comments.

## Additional experiments

First, we will need to train the models using the following variations:
1. Set the target mask rate to 0, 0.05, 0.1 (done), and 0.2
2. Remove the use of overfitting auto-detection.
3. Remove the use of overfitting auto-detection and set the target mask rate to 0.

We generate sub-versions of experiment 0.0.6 for all of these configurations.

## Data copying measure

Report on the number of data points that may have been generated exactly from the training data if we did not implement the target masking and the overfitting condition.

