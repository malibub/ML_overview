# Random Forests
### **2. Characterizing the accuracy of random forests**
### 2.1 Why do renadom forests converge? 
#### **Definition** *Random forests*
A random forest is a classifier consisting of an ensemble of tree-structured classifiers  $\{h(\boldsymbol{x}, \Theta_k):= h_k(\boldsymbol{x}), k=1,\cdots,K \}$ where $\Theta_k$ are $K$ iid relaizations of of a random variable $\Theta$ and each tree votes for the most popular class at input $\boldsymbol{x}$.

- The margin for an ensemble of classifiers an d a training set drawn from the random vectors $Y$, $\boldsymbol{X}$ is given by $$ \begin{align*} mg(Y, \boldsymbol{X}) &= av_k I(h_k(\boldsymbol{X})=Y)-\max_{j \ne Y} av_k I(h_k(\boldsymbol{X})=j) \\ &= \frac{1}{K} \sum_{k=1}^K I(h_k(\boldsymbol{X})=Y) - \max_{j \ne Y} \frac{1}{K} \sum_{k=1}^KI(h_k(\boldsymbol{X})=j)\end{align*}.$$
Note that if $mg(Y, \boldsymbol{X})<0$, on avarage the $K$ classifiers predict the wrong class for $\boldsymbol{X}$. The larger the margin, the more confidence in the classification.
- The generalization error is given by $$PE^* = P_{\boldsymbol{X},Y}\big(mg(Y, \boldsymbol{X})<0\big).$$

#### **Theorem** 
Let $\Theta$ be a renadom variable and $I \subseteq \mathbb{N}$ some  set of indices. As the numbers of trees increases, for almost surely all sequences $\Theta_i$, $i\in I$ i holds $$PE^*  \xrightarrow[a.s.]{K \rightarrow  \infty }  P_{\boldsymbol{X},Y}\left[ P_\Theta(h(\boldsymbol{X},\Theta) =Y)- \max_{j \ne Y} P_\Theta(h(\boldsymbol{X},\Theta) =j) <0\right].$$

*Proof*

TBA


- This theorem shows that  when adding more trees to the random forrest the resulting classifier's generalization error is limited. *How does this imply that random forests do not overfit?*

### 2.2 Strength an correlation

- The generelization error of a random forest can be derived from the accurancy of the individual tree / classifier and the dependence between them. (see Amit and Geman, 1997)

#### **Definition** *Margin of a random forest*
The margin of a random forest is $$ mr(\boldsymbol{X},Y) = P_\Theta(h(\boldsymbol{X},\Theta) =Y)- \max_{j \ne Y} P_\Theta(h(\boldsymbol{X},\Theta) =j)$$ and the strengh of the set of classifiers $\{h(\boldsymbol{x}, \Theta ) \}$ is $$ s= \mathbb{E}_{\boldsymbol{X}, Y}\left[ mr(\boldsymbol{X},Y)\right].$$

#### **Theorem** *Tschebycheff's inequality*
Let $X$ be a random variable with $\mathbb{E}[X^2]< \infty$. Then it holds $$P\left( \mid X -\mathbb{E}[X^2] \mid > \varepsilon\right) \le \frac{Var(X)}{\varepsilon^2} \qquad \forall \varepsilon >0.$$

- If $s>0$ we can follow from Tschebycheff's inequality that $$ \begin{split} PE^* &= P_{\boldsymbol{X},Y} \big( mr(\boldsymbol{X},Y) < 0 \big)  =  P_{\boldsymbol{X},Y} \big( s-mr(\boldsymbol{X},Y) > s \big) \\ &\le P_{\boldsymbol{X},Y} \big( \{s-mr(\boldsymbol{X},Y) > s\} \cup \{s-mr(\boldsymbol{X},Y) < -s\} \big) \\ &= P_{\boldsymbol{X},Y} \big( \mid mr(\boldsymbol{X},Y)-s\mid > s \big) \end{split} $$
so we get $$PE^* \le \frac{Var(mr(\boldsymbol{X},Y))}{s^2}.$$
- Now to find a more understandable approximation of $Var(mr(\boldsymbol{X},Y))$ let $$ \hat{j}(\boldsymbol{X},Y)= arg \max_{j\ne Y}P_\Theta(h(\boldsymbol{X},\Theta) =j).$$ Now we can wirte $$ \begin{split}mr(\boldsymbol{X},Y) &= P_\Theta(h(\boldsymbol{X},\Theta) =Y)- P_\Theta(h(\boldsymbol{X},\Theta) =\hat{j}(\boldsymbol{X},Y)) \\ &= \mathbb{E}_\Theta \left[ I(h(\boldsymbol{X},\Theta) =Y) -  I(h(\boldsymbol{X},\Theta) =\hat{j}(\boldsymbol{X},Y))\right]. \end{split}$$

#### **Definition** *Raw margin of a random forest*
The raw margin function is $$ rmg(\Theta, \boldsymbol{X},Y) = I(h(\boldsymbol{X},\Theta) =Y) -  I(h(\boldsymbol{X},\Theta) =\hat{j}(\boldsymbol{X},Y)). $$ So $mg(\boldsymbol{X},Y)$ is the expected valu of $rmg(\Theta, \boldsymbol{X},Y)$.

- Now let $\Theta$ and $\tilde{\Theta}$ be iid rv, then we have $$\mathbb{E}_\Theta \left[ f(\Theta) \right]^2 = \mathbb{E}_{\Theta, \tilde{\Theta}} \left[ f(\Theta) f(\tilde{\Theta}) \right].$$ So we get $$\begin{split} mr(\boldsymbol{X},Y)^2 &= \mathbb{E}_\Theta\left[ rmg(\Theta, \boldsymbol{X},Y)\right]^2 \end{split} = \mathbb{E}_{\Theta, \tilde{\Theta}}\left[ rmg(\Theta, \boldsymbol{X},Y)\cdot  rmg(\tilde{\Theta}, \boldsymbol{X},Y)\right].$$
- Now if $\mathbb{E}\left[mr(\boldsymbol{X},Y)\right]<\infty$ we have $$\begin{split}Var(mr(\boldsymbol{X},Y)) &= \mathbb{E}_{\boldsymbol{X},Y}\left[mr(\boldsymbol{X},Y)^2\right]-\mathbb{E}_{\boldsymbol{X},Y}\left[mr(\boldsymbol{X},Y)\right]^2 \\ &=\mathbb{E}_{\boldsymbol{X},Y}\left[\mathbb{E}_{\Theta, \tilde{\Theta}}\left[ rmg(\Theta, \boldsymbol{X},Y)\cdot  rmg(\tilde{\Theta}, \boldsymbol{X},Y)\right]\right]-\mathbb{E}_{\boldsymbol{X},Y} \left[ \mathbb{E}_{\Theta} \left[ rmg(\Theta, \boldsymbol{X},Y)\right]\right]^2 \\ &=\mathbb{E}_{\Theta, \tilde{\Theta}}\left[\mathbb{E}_{\boldsymbol{X},Y}\left[ rmg(\Theta, \boldsymbol{X},Y)\cdot  rmg(\tilde{\Theta}, \boldsymbol{X},Y)\right]\right]-\mathbb{E}_{\Theta} \left[ \mathbb{E}_{\boldsymbol{X},Y} \left[ rmg(\Theta, \boldsymbol{X},Y)\right]\right]\cdot \mathbb{E}_{\tilde{\Theta}} \left[ \mathbb{E}_{\boldsymbol{X},Y} \left[ rmg(\tilde{\Theta}, \boldsymbol{X},Y)\right]\right] \\ &=\mathbb{E}_{\Theta, \tilde{\Theta}}\left[ Cov_{\boldsymbol{X},Y} (rmg(\Theta, \boldsymbol{X},Y),  rmg(\tilde{\Theta}, \boldsymbol{X},Y))\right] \\ &= \mathbb{E}_{\Theta, \tilde{\Theta}}\left[ Corr_{\boldsymbol{X},Y} (rmg(\Theta, \boldsymbol{X},Y),  rmg(\tilde{\Theta}, \boldsymbol{X},Y))\cdot Std_{\boldsymbol{X},Y}(rmg(\Theta, \boldsymbol{X},Y))\cdot Std_{\boldsymbol{X},Y}(rmg(\tilde{\Theta}, \boldsymbol{X},Y))\right] \end{split}.$$
