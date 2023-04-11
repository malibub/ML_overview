# Bagging Predictors (Leo Breimann, 1996)
## Intro
- We have a learingn data set $$\mathcal{L}=:\left\{(y_n, \boldsymbol{x_n}), n=1,\cdots N \right\}$$ where $y_n$ is either a numerical or a class variable.
- Let $\varphi(\boldsymbol{x}, \mathcal{L})$ be a predictor that etsimates $y$ for given input $\boldsymbol{x}$ and learning dataset $\mathcal{L}$.
- Not if we have a sequence of learnign datasets $\mathcal{L}_k$ with the same underlying distribution as in $\mathcal{L}$ we can use $\mathcal{L}_k$ to create a better predictor than $\mathcal{L}$ using only the sequence of predictors $\varphi(\boldsymbol{x}, \mathcal{L}_k)$.
- For a numerical repsonse varibale we simply use the avarage of $\varphi(\boldsymbol{x}, \mathcal{L}_k)$ over $k$ , $$\varphi_A(\boldsymbol{x})= E_{\mathcal{L}} \varphi(\boldsymbol{x}, \mathcal{L})$$ where $A$ denotes aggregation and $E_{\mathcal{L}}$ denotes excpectation over ${\mathcal{L}}$. (bedinger Erwartungswert???) For predictin a class of $j \in \{1,\cdots,J\}$ letting the predictors vote gives the prediction, so $$\varphi_A(\boldsymbol{x})= argmax_j  N_j,$$ where $Nj = \#\{k :\quad \varphi(\boldsymbol{x}, \mathcal{L}_k)=j\}.$
- Here now boosting (bootstrap aggregatin) comes into play as in most cases we do not have the luxury of having replicates of $\mathcal{L}$. We sample $N$ times from $\mathcal{L}$ with replacement to get our $B$-th bootstrap sample $\mathcal{L}^B$. Any data point may appear multiple times in $\mathcal{L}^B$ or not at all. For Details see Efron and Tibshirani, 1993.
- Now, how does this improve the performance? Or does it even?
- We see improvement in $\varphi_B$ compared to $\varphi$ if the latter is a unstable predictor, i.e.  if small changes in the learnign data $\mathcal{L}$ lead to big changes in the predictor. Unstable procedures are for example neuronal nets, classification and regression trees and some linear regression while $k$-nrearest neighbour methods are stable (see Breimann, 1994).

## Why does boosting work?

##### Numeric Prediction
- Each $(y, \boldsymbol{x})$ is drawn independantly from an (unknown) distribution $P$. Ozr aggregated predictor is given by  $$\varphi_A(\boldsymbol{x})\quad =  E_{\mathcal{L}}[\varphi(\boldsymbol{x}, \mathcal{L})]\quad = P(\boldsymbol{X} = \boldsymbol{x}) \cdot \sum_{b=1}^B \varphi(\boldsymbol{x}, \mathcal{L}).$$ (??? how is this the average over L and why is it the expected value) -> Paper "Analyzing Bagging"