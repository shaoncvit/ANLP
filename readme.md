1. Neural Network Language Model
We use different hyperparameters combinations to test the NNLM. Out of these, a lower leaning rate of 1e-3, batch size 32 and Adams optimizer with default regularozation gives the best performance.
Test Perpexlity Score - 341.2452
![Train loss curve](lm1.png)
```
python lm1.py
```

2. RNN-based Language Model
Test Perplexity - 287
![](lm2_perplexity.png)

```
python lm2.py --use_wandb
```

3. Tranformer-
We see that the Perplexity loss in train and val mostly have their median values around 300. However, some of the short sequences, give a high perplexity score which is why our average value becomes high.
Train Perplexity = 63.72
Test Perplexity = 1049.15

```
python lm3.py --use_wandb
```
4. ondrive link for model https://iiitaphyd-my.sharepoint.com/:f:/g/personal/shaon_b_research_iiit_ac_in/EiVuqcoVmMBOvGOkcNiDLY4BDtN2W5VLq3MAxI0iezZPBQ?e=InnCu4