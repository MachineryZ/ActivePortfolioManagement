# 基础知识

本章主要介绍 CAPM Capital Asset Pricing Model 资产定价模型，本章的主要观点：
1. 任何股票的收益都可以分解为两部分，系统部分和残余部分，这不是什么理论问题
2. 股票收益残差部分的期望值为 0
3. capm 模型易于理解和操作
4. capm 模型的隐含前提是市场非常有效
5. capm 模型人为消极性管理相对于积极性管理来说承担的风险要低很多
6. capm 关注的是预期收益而非风险
7. capm 模型提供了确定一致预期收益的途径，积极型管理者成功与否取决于其收益是否优于 capm 的一致预期收益

capm：
1. 一个投资组合 P 的超额收益为 $r_p$，市场组合 M 的超额收益为 $r_M$
2. 投资组合 P 的 $\beta$ 为 $\beta = \frac{Cov\{r_p, r_M\}}{Var(r_M)}$
3. $\beta$ 的初衷还是来源于投资组合超额收益与同期市场组合超额收益的简单线性回归
$
r_p(t) = \alpha_p + beta_p r_M(t) + \epsilon_o(t)
$
4. capm 表达式：任何一只股票或者投资组合的期望超额收益正比于改过的 $\beta$ 值
$$
E(r_p) - r_f = \beta_p * (E(r_M) - r_f)
$$
5. 投资组合 P 的市场收益和残余收益 $r_p - r_f = \beta_p (r_M - r_f) + \theta_p,\ \ E(\theta_p)=0$
6. 投资组合 P 的方差 $\sigma^2_p=\beta_p^2\sigma_M^2 + w_p^2$

有效市场理论：
1. 弱有效性：如果只依据历史股价和交易量数据，投资者不可能做的比市场好
2. 市场半强有效性：指根据公众都可获得的信息，历史股价、基本数据、公开发表的分析家推荐等，投资者不可能做的比市场好
3. 市场强有效性：投资者永远不可能比市场做得好

证券市场线 Security Market Line SML
1. 对于任何股票或者投资组合，他们的 $\beta$ 值以及对应的 capm 模型得出的预期收益图
2. 横坐标是 $\beta$ 纵坐标是期望收益图
3. 截距是无风险利率 $i_r$（因为无风险资产的$\beta$是0，它自然对应）
4. 斜率为市场超额期望收益 $\mu_M$
5. $\beta = 1$ 是市场组合（市场组合自己和本身的 cov 是 1）

## Homework
---
(1) 利用 capm 公式即可
$$
E(r_p) - r_f = \beta_p (E(r_M) - r_f)
$$
所以，$ -1.05 * 0.5\% = -0.525\%$

---

(2) 同理可得：$1.05 * 7\% = 7.35\%$

---

(3) 我们有投资组合的方差等式
$$
\sigma_p^2 = \beta_p^2 \sigma_M^2 + w_p^2
$$ 
有correlation等式：
$$
corr(r_A, r_B) = cov(r_A, r_B) / std(r_A) / std(r_B)
$$
有等式（这里有点奇怪，不知道咋来的）
$$
cov(r_A, r_B) = \beta_A\beta_B\sigma_M^2 + w_{A,B}
$$
带入相关性等式，可得：
$$
corr(r_A, r_B) = \beta_A \beta_B \sigma_M^2 / \sigma_A/\sigma_B=1.15 *0.95*0.2^2/0.35/0.33=0.3784
$$
对于残余波动率：
$$
w_P = \sqrt{\sigma_p^2 - \beta_P^2\sigma_M^2}
$$
所以得到：
$$
w_A = \sqrt{0.35^2 - 1.15^2*0.2^2}=0.2638
$$
$$
W_B = \sqrt{0.33^2 - 0.95^2 * 0.2^2} = 0.2698
$$
---

(4) 查表可得：通用电气的 $\beta = 1.3$，是全部表里面最高的，所以，当市场的组合的期望收益是正的，而且我们对于风险无偏好（也就是高风险和低风险对我们的效用函数无影响），在此情况下，我们会选择购买百分百的 通用电气的股票

---

(5) capm 模型提到，预期残余收益的期望是 0

## 技术附录

$h$ 代表风险资产的持有量，每个维度代表每种资产的百分比权重 \\
$f$ 用向量表示的预期超额收益 \\
$\mu$ 用向量表示的 capm 模型下的预期超额收益，也就是说，$f=\mu$ 时 capm 模型成立 \\
$V$ 风险资产超额收益的协方差矩阵，假设矩阵非奇异 \\
$\beta$ 向量表示的各类资产的 $\beta$ 值 \\
$e$ 单位向量 \\
风险定义为年超额收益的标准差

考虑单期模型，在投资期间不能调整投资组合，基本假设如下：
1. 存在一种无风险资产
2. 一阶矩和二阶矩都存在
3. 无法构造出一个无风险的充分投资组合
4. 方差最小的充分投资组合的预期超额收益为正

令 $a^T = \{a_1,a_2,...,a_N\}$ 为表示资产的属性或特征的向量，则投资组合 $h_p$ 在属性向量 $a$ 上的头寸 exposure 可记为 $a_p = \sum_n a_nh_{p, n}$


## 引理1：
1. 对于不为零的属性 a，一定存在唯一的投资组合 $h_a$，同时满足风险最小且属性 a 的头寸上为1，$h_a = V^{-1}a/(a^TV^{-1} a)$
2. 特征投资组合 $h_a$ 的方差为 $\sigma_a^2=h_a^TVh_a=1/(a^TV^{-1}a)$
3. 投资组合 $h_a$ 中所有资产的 $\beta$ 值等于 $a = V h_a/\sigma_a^2$
4. 考虑两个属性 a 和 d，他们对应的特征投资组合分别为 $h_a$ 和 $h_d$，令 $a_d$ 表示投资组合 $h_d$ 在属性 a 上的暴露头寸，$d_a$ 表示投资组合 $h_a$ 在属性 d 上的暴露头寸，则特征投资组合的方差满足：$\sigma_{a,d}=a_d\sigma^2_{a}=d_a\sigma^2_d$
5. 若 $\kappa$ 为一正标量，则对于 $\kappa a$ 的特征投资组合为 $h_a/\kappa$（线性）
6. 如果属性 a 是属性 d 和 f 的加权组合，那么 a 的特征组合也是 d 和 f 所对应的特征投资组合的加权组合，具体表示为
$$
a = \kappa_d d + \kappa_f f \\
h_a = (\kappa_d\sigma_a^2/\sigma_d^2)h_d + (\kappa_f\sigma_a^2/\sigma_f^2)h_f \\
1/\sigma_a^2 = (\kappa_d a_d/\sigma_d^2) + (\kappa_f a_f/\sigma_f^2)
$$

证明：
- (1) 用拉格朗日乘子法
$$
min\ h^TVh,\ \ s.t.\ h^Ta=1 \\
\frac{\partial (h^TVh-\theta(h^Ta-1))}{\partial h}\Rightarrow Vh -\theta a=0 \\
\theta = 1/(a^TV^{-1}a) \\
\Rightarrow h = \frac{V^{-1} a}{a^TV^{-1}a}
$$
得证第一条性质
- (2) 特征投资组合的方差定义即为持仓乘以协方差 $h_a^TVh_a$
- (3) 投资组合 $h_a$ 中所有资产的 $\beta$ 值等于 $a$，定义即为 $Vh_a/\sigma_a^2$
- (4) 考虑两个属性 a 和 d
$$
\sigma_{a,d} = h_a^T Vh_a=(\sigma_a^2 a)^T h_a = \sigma_a^2 (a^Th_a) = \sigma_a^2 a_d
$$
另外一个方向可以同理得证
- (5) 头寸定义 $(\kappa a)^Th_a/\kappa = a^Th_a$
- (6) $a = \kappa_d d + \kappa_f f$ 那么我们可以有：
$$
\frac{Vh_a}{\sigma_a^2} = a = \kappa_d d + \kappa_f f \\
= \kappa_d \frac{V h_d}{\sigma_d^2} + \kappa_f \frac{V h_f}{\sigma_f^2} \\
\Rightarrow \frac{V h_a}{\sigma_a^2} = \frac{\kappa_d V h_d}{\sigma_d^2} + \frac{\kappa_f V h_f}{\sigma_f^2} \\
\Rightarrow h_a = \frac{\kappa_d \sigma_a^2}{\sigma_d^2}h_d + \frac{\kappa_f \sigma_a^2}{\sigma_f^2}h_f
$$
以及，我们有风险的公式：
$$
\frac{1}{\sigma_a^2} = a^TV^{-1}a=\frac{\kappa_d a_d}{\sigma_d^2} + \frac{\kappa_f a_f}{\sigma_f^2}
$$


## 引理2：
令 $q$ 为预期超额收益 f 的特征投资组合:
$$
h_q = \frac{V^{-1}f}{f^TV^{-1}f}
$$
则有
1. $ SR_q = max\{SR_p | P\} = (f^T V^{-1} f)^{-1/2}$
2. $f_q = 1$, $\sigma_q^2 = \frac{1}{f^TV^{-1}f}$
3. $f = \frac{V h_q}{\sigma_q^2}=\frac{Vh_q}{\sigma_q}SR_q$
4. 若用 $\rho_{P,q}$ 表示投资组合 P 与 q 的相关系数，则有：$SR_P = \rho_{P,q} SR_q$
5. 组合 q 投资于风险资产的比例为 $e_q = \frac{f_C\sigma_q^2}{\sigma_C^2}$

证明：
- (1) 首先加上约束条件转化为在预期超额收益为1的组合中找出风险最小的投资组合
$$
\sigma_q^2 = \frac{Vh_q}{q}=\frac{V}{f}\frac{V^{-1}f}{f^TV^{-1}f}=\frac{1}{f^TV^{-1}f} \\
SR_q = \frac{f_q}{\sigma_q} = \frac{1}{\sigma_q} = \sqrt{f^TV^{-1}f}
$$
- (2) 和 (1) 可以同时证明（因为只要约束了预期超额收益为1，那么我们就可以套用引理1的结论）
- (3) 
$$
f = \frac{Vh_q}{\sigma_q^2} (引理1) = \frac{Vh_1}{\sigma_q}\frac{1}{\sigma_q} = \frac{Vh_q}{\sigma_q}SR_q
$$
- (4)
$$
SR_p = \frac{f_p}{\sigma_p} = \frac{h_p^Tf}{\sigma_p} = \frac{h_p^T}{\sigma_p} \frac{Vh_q}{\sigma_q}SR_q = \frac{h_p^TVh_q}{\sigma_p\sigma_q} SR_q = \rho_{p, q}
$$
- (5) 
$$
\sigma_{q,C} = e_q\sigma_C^2=f_C\sigma_q^2
$$
