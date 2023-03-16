# Review Experiments

We document here the planned experiments that we will undertake for the ICML review comments.

## Additional experiments

First, we will need to train the models using the following variations:
1. Set the target mask rate to:
   1. (0.0.6.1) mask_rate=0
   2. (0.0.6.2) mask_rate=0.05
   3. (0.0.6) mask_rate=0.1 (done)
   4. (0.0.6.3) mask_rate=0.2
2. (0.0.6.4) Remove the use of overfitting auto-detection.
3. (0.0.6.5) Remove the use of overfitting auto-detection and set the target mask rate to 0.

We generate sub-versions of experiment 0.0.6 for all of these configurations.

## Data copying measure

Report on the number of data points that may have been generated exactly from the training data if we did not implement the target masking and the overfitting condition.

