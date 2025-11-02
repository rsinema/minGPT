# Attention wasn't all we needed

## Training Specifications

I implemented all of the modifications and tried several different configurations of hyperparameters to test and see about differences in the learning. 
I used a subset of 1 million rows from the pile to train, with a learning rate of 5e-6, for 1 hour per modification.  

I also tried 5e-4 as a learning rate, a 3 hour time limit, and the whole dataset, but those modifications did not seem to change how the model would over fit on the data.
When running with the whole pile datatset the process was killed with a OOM exception. 

## Modification Evalutations

From looking at the loss curves for the different runs, it doesn't appear that the modifications had a significant effect on the model's training trajectory.
With a different model architecture or with a larger dataset these modifications would have made a larger difference in the model's loss.

The warmup iterations with the learning rate schedulers (linear and cosine annealing) did show a difference in the loss curve, because of the scheduled
learning rates for the first *n* interations (I used 1000). 

## Final Loss Values

Base model: 0.6355953216552734
Learning rate warmup: 0.7916161417961121
Cosine Scheduler: 0.7008657455444336
RMS Norm: 0.673073410987854
RoPE: 0.7690846920013428
SwiGLU: 0.08615359663963318
All Modifications: 0.6614674925804138

### Validation Set Final Loss Values

Base model: 0.579835756903603
Learning rate warmup: 0.5298536505017962
Cosine Scheduler: 0.6378134441754174
RMS Norm: 0.5798523397672743
RoPE: 0.5784605083957551
SwiGLU: 0.08543122530220047
All Modifications: 0.4862024452951219

## Loss Curves

![alt text](output/experiments/all_features/training_loss.png)

![alt text](output/experiments/base/training_loss.png)

![alt text](output/experiments/cos_scheduler/training_loss.png)

![alt text](output/experiments/lw_scheduler/training_loss.png)

![alt text](output/experiments/rms_norm/training_loss.png)

![alt text](output/experiments/rope/training_loss.png)

![alt text](output/experiments/swiglu/training_loss.png)