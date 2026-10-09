---
title: 一、单层神经网络：逻辑回归的前向传播与反向传播
date: 2026-10-09 21:00:00 +0800
categories: [DeepLearning]
tags: [深度学习]
---
# 单层神经网络：逻辑回归的前向传播与反向传播

单层神经网络做二分类时，就是逻辑回归。前向先算线性结果 $z$，再经 sigmoid 得到概率 $a$。训练用二元交叉熵。对这个损失求梯度时，中间项会消掉，反向误差只剩 $a-y$。本文把前向、损失、反向、向量化和伪代码按同一种形状约定写完整。

---

## 1. 模型

输入是特征向量，输出是「标签为 1」的概率。参数只有权重 $w$ 和偏置 $b$。

$$
z = w^{\mathsf T} x + b
$$

$$
a = \sigma(z) = \frac{1}{1+e^{-z}}
$$

$z$ 可以是任意实数。$a$ 落在开区间 $(0,1)$。

判决用阈值 $0.5$，它和 $z$ 的符号是同一件事：

$$
a \ge 0.5 \iff z \ge 0
$$

$$
\hat{y} =
\begin{cases}
1, & z \ge 0 \\
0, & z < 0
\end{cases}
$$

因此分界面是一张超平面：

$$
w^{\mathsf T} x + b = 0
$$

计算图只有一条链：

```text
x ──┐
w ──┼──► z = w^T x + b ──► a = sigmoid(z) ──► L(a, y)
b ──┘                                         ▲
                                              │
                                              y
```

参数更新走 $L \to a \to z \to w$ 和 $L \to a \to z \to b$。输入 $x$ 是数据，不是参数。$dx$ 只在这一层前面还接着别的层时才需要，见第 8 节。

---

## 2. 符号与形状

样本按列摆放。后面的矩阵乘法都依赖这个约定。

| 符号 | 含义 | 形状 |
|---|---|---|
| $n_x$ | 特征个数 | 标量 |
| $m$ | 样本个数 | 标量 |
| $x$ | 一个样本，列向量 | $(n_x,1)$ |
| $X$ | 全部样本，第 $i$ 列是 $x^{(i)}$ | $(n_x,m)$ |
| $y$ | 一个标签，只取 $0$ 或 $1$ | 标量 |
| $Y$ | 全部标签 | $(1,m)$ |
| $w$ | 权重 | $(n_x,1)$ |
| $b$ | 偏置 | 标量 |
| $z,Z$ | 线性输出 | 标量，或 $(1,m)$ |
| $a,A$ | sigmoid 输出 | 与 $z,Z$ 相同 |
| $\alpha$ | 学习率 | 标量 |

上标表示样本，下标表示特征，二者不混用：

$$
\text{第 } i \text{ 个样本：}\quad x^{(i)},\; y^{(i)},\; z^{(i)},\; a^{(i)}
$$

$$
\text{第 } j \text{ 个特征：}\quad x_j,\; w_j
$$

一个样本写成向量就是：

$$
x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_{n_x} \end{bmatrix}, \qquad
w = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_{n_x} \end{bmatrix}
$$

$$
z = w_1 x_1 + w_2 x_2 + \cdots + w_{n_x} x_{n_x} + b
$$

---

## 3. 前向传播

### 3.1 一个样本

$$
z = w^{\mathsf T} x + b
$$

$$
a = \frac{1}{1+e^{-z}}
$$

偏置加在点积外面。它和「先把 $b$ 加进每个特征再做点积」不是同一个模型：

$$
\begin{aligned}
\text{正确} &\quad z = w^{\mathsf T} x + b \\
\text{错误} &\quad z = w^{\mathsf T}(x+b) = w^{\mathsf T} x + b\sum_{j=1}^{n_x} w_j
\end{aligned}
$$

第二种把每个特征都平移了 $b$，再被全部权重加总。代码里必须写成点积完成之后再加 $b$。

### 3.2 $m$ 个样本

$$
Z = w^{\mathsf T} X + b, \qquad Z \in \mathbb{R}^{1 \times m}
$$

$$
A = \sigma(Z), \qquad A \in \mathbb{R}^{1 \times m}
$$

$w^{\mathsf T}$ 是 $(1,n_x)$，$X$ 是 $(n_x,m)$，乘积是 $(1,m)$。每一列对应一个样本。标量 $b$ 加到每一列上。

---

## 4. Sigmoid 的导数

反向传播要用 $\dfrac{da}{dz}$。设

$$
a = \frac{1}{1+e^{-z}}
$$

令

$$
u = 1+e^{-z}, \qquad a = u^{-1}, \qquad \frac{du}{dz} = -e^{-z}
$$

求导：

$$
\frac{da}{dz} = -u^{-2}\frac{du}{dz} = \frac{e^{-z}}{(1+e^{-z})^{2}}
$$

分子改写成「分母减 1」：

$$
e^{-z} = (1+e^{-z})-1
$$

$$
\begin{aligned}
\frac{da}{dz}
&= \frac{(1+e^{-z})-1}{(1+e^{-z})^{2}} \\
&= \frac{1}{1+e^{-z}} - \frac{1}{(1+e^{-z})^{2}} \\
&= a - a^{2} \\
&= a(1-a)
\end{aligned}
$$

结论：

$$
\sigma'(z) = a(1-a)
$$

$a$ 在 $(0,1)$ 内，导数恒正。$a=0.5$ 时导数最大，等于 $0.25$。$a$ 靠近 $0$ 或 $1$ 时导数靠近 $0$，这就是饱和。

两个后面化简要用的比值：

$$
\frac{\sigma'(z)}{a} = 1-a, \qquad \frac{\sigma'(z)}{1-a} = a
$$

---

## 5. 损失函数

### 5.1 一个样本：负对数似然

把 $a$ 当作伯努利分布的参数：

$$
P(y \mid x) = a^{y}(1-a)^{1-y}
$$

对数似然：

$$
\ln P(y \mid x) = y\ln a + (1-y)\ln(1-a)
$$

损失取负号，最小化损失就等于最大化似然：

$$
L(a,y) = -\, y\ln a - (1-y)\ln(1-a)
$$

两种标签下它各自退化成一项：

$$
\begin{aligned}
y = 1 &\quad L = -\ln a, \qquad \text{$a$ 越靠近 $1$，$L$ 越小} \\
y = 0 &\quad L = -\ln(1-a), \qquad \text{$a$ 越靠近 $0$，$L$ 越小}
\end{aligned}
$$

### 5.2 $m$ 个样本：代价

样本独立同分布时，总对数似然是各项之和。代价取平均，数值大小不随 $m$ 成比例膨胀，学习率更好选：

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m} L\!\left(a^{(i)}, y^{(i)}\right)
$$

$L$ 是一个样本的损失。$J$ 是全部样本的平均代价。梯度前有没有 $\dfrac{1}{m}$，取决于求导对象是 $L$ 还是 $J$。

---

## 6. 反向传播

下面先固定一个样本，省去上标 $(i)$。已知

$$
z = w^{\mathsf T} x + b, \qquad a = \sigma(z), \qquad L = -\, y\ln a - (1-y)\ln(1-a)
$$

### 6.1 对 $a$ 求导

$$
\frac{\partial L}{\partial a} = -\, y\cdot\frac{1}{a} - (1-y)\cdot\frac{1}{1-a}\cdot(-1) = -\frac{y}{a} + \frac{1-y}{1-a}
$$

记

$$
da \triangleq \frac{\partial L}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}
$$

这里的 $da$ 是导数，不是 $a$ 的增量。

### 6.2 对 $z$ 求导，化成 $a-y$

链式法则：

$$
\begin{aligned}
dz
&\triangleq \frac{\partial L}{\partial z}
= \frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z} \\
&= \left(-\frac{y}{a} + \frac{1-y}{1-a}\right) \cdot a(1-a)
\end{aligned}
$$

拆开，分母分别被消掉：

$$
\left(-\frac{y}{a}\right)\cdot a(1-a) = -y(1-a)
$$

$$
\frac{1-y}{1-a}\cdot a(1-a) = (1-y)\, a
$$

相加：

$$
\begin{aligned}
dz
&= -y(1-a) + (1-y)a \\
&= -y + ya + a - ya \\
&= a - y
\end{aligned}
$$

$ya$ 与 $-ya$ 抵消。因此

$$
dz = a - y
$$

另一条路不展开 $da$，用第 4 节的比值：

$$
\begin{aligned}
\frac{\partial L}{\partial z}
&= -y\cdot\frac{\sigma'(z)}{a} + (1-y)\cdot\frac{\sigma'(z)}{1-a} \\
&= -y(1-a) + (1-y)a \\
&= a - y
\end{aligned}
$$

两条路结果相同。

这个消元只对「sigmoid + 二元交叉熵」成立。如果损失改成平方误差，链式法则里的 $a(1-a)$ 消不掉：

$$
L_{\mathrm{sq}} = \frac{1}{2}(a-y)^{2}
$$

$$
\frac{\partial L_{\mathrm{sq}}}{\partial z} = (a-y)\, a(1-a)
$$

不能把平方误差的反向也写成 $a-y$。

实现时不要先算 $da$ 再乘 $a(1-a)$。$a$ 靠近 $0$ 或 $1$ 时，$-\dfrac{y}{a}$ 和 $\dfrac{1-y}{1-a}$ 会溢出，乘回去之后的极限才是有限的 $a-y$。代码直接使用化简结果。

### 6.3 对 $w$ 和 $b$ 求导

$$
z = w_1 x_1 + \cdots + w_{n_x} x_{n_x} + b
$$

$$
\frac{\partial z}{\partial w_j} = x_j, \qquad \frac{\partial z}{\partial b} = 1
$$

所以

$$
\frac{\partial L}{\partial w_j} = dz \cdot x_j, \qquad \frac{\partial L}{\partial b} = dz
$$

$dz$ 是标量，乘在列向量上：

$$
\frac{\partial L}{\partial w} = x\, dz, \qquad \frac{\partial L}{\partial w}\in\mathbb{R}^{n_x \times 1}
$$

$$
\frac{\partial L}{\partial b} = dz
$$

分量形式就是 $\dfrac{\partial L}{\partial w_j} = x_j(a-y)$。预测高于标签时 $a-y>0$，该特征为正就会把对应权重往下拉。

---

## 7. 批量梯度与向量化

### 7.1 代价的梯度

$$
J = \frac{1}{m}\sum_{i=1}^{m} L\!\left(a^{(i)}, y^{(i)}\right)
$$

$$
dz^{(i)} = a^{(i)} - y^{(i)}
$$

求和与乘常数可以和求导交换：

$$
\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m} x^{(i)}\, dz^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m} dz^{(i)}
$$

第 $j$ 个权重：

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m} x_j^{(i)}\left(a^{(i)} - y^{(i)}\right)
$$

左边这些是 $J$ 的梯度，不是某一个 $L$ 的梯度。单个 $L$ 的梯度没有前面的 $\dfrac{1}{m}$。

### 7.2 矩阵形式

$$
X \in \mathbb{R}^{n_x \times m}, \qquad
dz \in \mathbb{R}^{1 \times m}, \qquad
dz^{\mathsf T} \in \mathbb{R}^{m \times 1}
$$

$$
\frac{\partial J}{\partial w} = \frac{1}{m}\, X\, dz^{\mathsf T}, \qquad \frac{\partial J}{\partial w}\in\mathbb{R}^{n_x \times 1}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m} dz^{(i)}
$$

形状对得上，结果才能和 $w$、$b$ 相减：$(n_x,m)$ 乘 $(m,1)$ 得到 $(n_x,1)$。

它和逐样本求和是同一件事。$X\, dz^{\mathsf T}$ 的第 $j$ 行等于

$$
\sum_{i=1}^{m} X_{ji}\, dz^{(i)}
$$

再除以 $m$，就是上一节的 $\dfrac{\partial J}{\partial w_j}$。

若样本被摆成行，即矩阵形状是 $(m,n_x)$，转置位置要整段对调，不能只改其中一次乘法。

---

## 8. $dx$ 与 $dz$

更新 $w$ 和 $b$ 只用 $dz$：

$$
dz = a - y
$$

$$
dw = x\, dz, \qquad db = dz
$$

误差若还要传回输入，走另一条链：

$$
dx = \frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}\cdot\frac{\partial z}{\partial x} = w\, dz
$$

$$
dx = w\, dz, \qquad dx \in \mathbb{R}^{n_x \times 1}
$$

$$
dz = a - y
$$

$dx$ 不等于 $dz$，也不能拿去更新 $w$。单层逻辑回归里 $x$ 是观测值，训练步骤不出现 $dx$。

---

## 9. 参数更新

沿梯度下降，用减号：

$$
w := w - \alpha\, \frac{\partial J}{\partial w}, \qquad b := b - \alpha\, \frac{\partial J}{\partial b}
$$

三种用法对应三种梯度，不要混用系数。

批量梯度下降用整批样本的精确梯度，其中已经包含 $\dfrac{1}{m}$：

$$
dw = \frac{1}{m}\, X\, dz^{\mathsf T}, \qquad db = \frac{1}{m}\sum_{i=1}^{m} dz^{(i)}
$$

随机梯度下降随机抽一个样本 $i$，用的是 $\dfrac{\partial L}{\partial w}$，不要再除以 $m$：

$$
dw = x^{(i)}\, dz^{(i)}, \qquad db = dz^{(i)}
$$

它是 $\dfrac{\partial J}{\partial w}$ 的无偏估计：每个样本等概率抽取时，

$$
\mathbb{E}\!\left[\frac{\partial L}{\partial w}\right] = \frac{\partial J}{\partial w}
$$

小批量设这个小批有 $m_b$ 个样本：

$$
dw = \frac{1}{m_b}\, X_{\mathrm{batch}}\, dz_{\mathrm{batch}}^{\mathsf T}, \qquad db = \frac{1}{m_b}\sum dz_{\mathrm{batch}}
$$

漏掉批量公式里的 $\dfrac{1}{m}$，相当于用了 $m$ 倍的梯度，步长会被放大 $m$ 倍。随机梯度本身就不再除以 $m$，那不是漏项。

$J$ 关于 $(w,b)$ 是凸函数。初值可以取

$$
w = 0, \qquad b = 0
$$

这一层没有多个隐藏单元，不需要用随机初始化去打破对称。深层网络才需要。

---

## 10. 伪代码

约定：$w$、$x$ 为列向量，$X$ 的每一列是一个样本。

### 10.1 单个样本上走若干步

只有一个样本时，$m=1$，$\partial L$ 和 $\partial J$ 相同。

```text
输入:
    x          形状 (n_x, 1)
    y          标量, 0 或 1
    alpha      学习率
    num_steps  迭代步数

初始化:
    w = zeros(n_x, 1)
    b = 0

for step = 1 to num_steps:
    z  = dot(w^T, x) + b
    a  = 1 / (1 + exp(-z))

    dz = a - y
    dw = x * dz
    db = dz

    w  = w - alpha * dw
    b  = b - alpha * db
```

对应的数组写法：

```text
z  = np.dot(w.T, x) + b
a  = 1 / (1 + np.exp(-z))
dz = a - y
dw = x * dz
db = dz
w  = w - alpha * dw
b  = b - alpha * db
```

$b$ 在点积外面。`np.dot(w.T, x + b)` 是错的。

### 10.2 全部样本的批量梯度下降

```text
输入:
    X          形状 (n_x, m)
    Y          形状 (1, m)
    alpha
    num_iters

初始化:
    w = zeros(n_x, 1)
    b = 0

for iter = 1 to num_iters:
    Z  = dot(w^T, X) + b                  # (1, m)
    A  = 1 / (1 + exp(-Z))                # (1, m)

    dZ = A - Y                            # (1, m)
    dw = (1/m) * dot(X, dZ^T)             # (n_x, 1)
    db = (1/m) * sum(dZ)                  # 标量

    w  = w - alpha * dw
    b  = b - alpha * db
```

```text
Z  = np.dot(w.T, X) + b
A  = 1 / (1 + np.exp(-Z))
dZ = A - Y
dw = (1/m) * np.dot(X, dZ.T)
db = (1/m) * np.sum(dZ)
w  = w - alpha * dw
b  = b - alpha * db
```

每一轮可以记录代价，用来确认整体在下降：

$$
J = -\frac{1}{m}\sum\left( Y\ln A + (1-Y)\ln(1-A) \right)
$$

$A$ 极靠近 $0$ 或 $1$ 时对数会溢出。计算 $J$ 之前把 $A$ 裁到一个小的开区间，例如 $[10^{-15},\, 1-10^{-15}]$。梯度仍用 $dZ = A-Y$，不要改回 $da$ 那条不稳定的式子。

### 10.3 按样本循环的随机梯度下降

```text
初始化:
    w = zeros(n_x, 1)
    b = 0

for epoch = 1 to num_epochs:
    将样本下标 1..m 随机打乱
    for i in 打乱后的顺序:
        z  = dot(w^T, x^{(i)}) + b
        a  = 1 / (1 + exp(-z))
        dz = a - y^{(i)}
        dw = x^{(i)} * dz          # 不再乘 1/m
        db = dz

        w  = w - alpha * dw
        b  = b - alpha * db
```

### 10.4 预测

预测只有前向，没有损失，也没有参数更新：

$$
z = w^{\mathsf T} x + b, \qquad a = \frac{1}{1+e^{-z}}
$$

$$
\hat{y} =
\begin{cases}
1, & a \ge 0.5 \\
0, & a < 0.5
\end{cases}
$$

$a$ 是概率。交叉熵直接使用 $a$，不要先把 $a$ 变成 $0/1$ 再拿去算损失。$0/1$ 只用于给出类别。

---

## 11. 数值演算

### 11.1 一个样本：核对 $da\cdot a(1-a) = a-y$

$$
x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad
y = 1, \quad
w = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad
b = 0
$$

前向：

$$
z = 0, \qquad a = \frac{1}{1+e^{0}} = \frac{1}{2}
$$

先走未化简的式子：

$$
\begin{aligned}
da
&= -\frac{y}{a} + \frac{1-y}{1-a}
= -\frac{1}{1/2}
= -2 \\
a(1-a)
&= \frac{1}{2}\cdot\frac{1}{2}
= \frac{1}{4} \\
dz
&= da\cdot a(1-a)
= -2\cdot\frac{1}{4}
= -\frac{1}{2}
\end{aligned}
$$

再走化简式：

$$
dz = a - y = \frac{1}{2} - 1 = -\frac{1}{2}
$$

一致。参数梯度：

$$
dw = x\, dz = \begin{bmatrix} 1 \\ 2 \end{bmatrix}\left(-\frac{1}{2}\right) = \begin{bmatrix} -1/2 \\ -1 \end{bmatrix}, \qquad db = -\frac{1}{2}
$$

学习率 $\alpha = 0.1$ 时，一步更新为：

$$
w = \begin{bmatrix} 0 \\ 0 \end{bmatrix} - 0.1\begin{bmatrix} -1/2 \\ -1 \end{bmatrix} = \begin{bmatrix} 0.05 \\ 0.1 \end{bmatrix}
$$

$$
b = 0 - 0.1\left(-\frac{1}{2}\right) = 0.05
$$

标签是 $1$，当前概率只有 $0.5$，两个特征都是正数，所以 $w$ 和 $b$ 都增大。下一步 $z$ 变大，$a$ 会靠近 $1$。

用链式法则直接看 $w_1$，可以再核对一次符号和数值：

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w_1} = (-2)\cdot\frac{1}{4}\cdot 1 = -\frac{1}{2}
$$

与 $dw$ 的第一个分量相同。

### 11.2 两个样本：核对矩阵乘法和逐项求和

$$
x^{(1)} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad y^{(1)} = 1, \qquad
x^{(2)} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}, \quad y^{(2)} = 0
$$

$$
X = \begin{bmatrix} 1 & 3 \\ 2 & 0 \end{bmatrix}, \qquad
Y = \begin{bmatrix} 1 & 0 \end{bmatrix}, \qquad
w = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \qquad
b = 0, \qquad m = 2
$$

两个样本的线性输出都是 $0$：

$$
A = \begin{bmatrix} 1/2 & 1/2 \end{bmatrix}
$$

$$
dZ = A - Y = \begin{bmatrix} 1/2 - 1 & 1/2 - 0 \end{bmatrix} = \begin{bmatrix} -1/2 & 1/2 \end{bmatrix}
$$

矩阵形式：

$$
dZ^{\mathsf T} = \begin{bmatrix} -1/2 \\ 1/2 \end{bmatrix}
$$

$$
X\, dZ^{\mathsf T}
= \begin{bmatrix}
1\cdot(-1/2) + 3\cdot(1/2) \\
2\cdot(-1/2) + 0\cdot(1/2)
\end{bmatrix}
= \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

$$
dw = \frac{1}{2}\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 1/2 \\ -1/2 \end{bmatrix}
$$

$$
db = \frac{1}{2}\left(-\frac{1}{2}+\frac{1}{2}\right) = 0
$$

逐样本相加：

$$
\begin{aligned}
dw
&= \frac{1}{2}\left(
x^{(1)}\cdot\left(-\frac{1}{2}\right)
+ x^{(2)}\cdot\frac{1}{2}
\right) \\
&= \frac{1}{2}\left(
\begin{bmatrix} -1/2 \\ -1 \end{bmatrix}
+ \begin{bmatrix} 3/2 \\ 0 \end{bmatrix}
\right) \\
&= \frac{1}{2}\begin{bmatrix} 1 \\ -1 \end{bmatrix}
= \begin{bmatrix} 1/2 \\ -1/2 \end{bmatrix}
\end{aligned}
$$

两种算法相同。

---

## 12. 实现时对照

单个样本：

$$
\begin{aligned}
z &= w^{\mathsf T} x + b \\
a &= \sigma(z) \\
dz &= a - y \\
dw &= x\, dz \\
db &= dz
\end{aligned}
$$

全部样本：

$$
\begin{aligned}
Z &= w^{\mathsf T} X + b \\
A &= \sigma(Z) \\
dZ &= A - Y \\
dw &= \frac{1}{m}\, X\, dZ^{\mathsf T} \\
db &= \frac{1}{m}\sum dZ
\end{aligned}
$$

传回上一层，本层训练不用：

$$
dx = w\, dz
$$

容易写错的几处：

1. $b$ 必须写在点积外面。
2. $\dfrac{\partial L}{\partial w} = x\, dz$ 是一个样本的梯度；$\dfrac{\partial J}{\partial w} = \dfrac{1}{m} X\, dZ^{\mathsf T}$ 是平均代价的梯度。
3. 批量梯度漏掉 $\dfrac{1}{m}$，步长会放大 $m$ 倍。随机梯度使用单个样本的 $\dfrac{\partial L}{\partial w}$，本来就没有 $\dfrac{1}{m}$。
4. $dx = w\, dz$，而 $dz = a-y$。二者不是同一个量。
5. $dz = a-y$ 只属于 sigmoid 加二元交叉熵。
6. 样本按列存放时，$dZ$ 是 $(1,m)$，乘法是 $X\, dZ^{\mathsf T}$。把 $dZ$ 当成列向量再右乘，形状会错。
7. 计算 $J$ 时对 $A$ 做裁剪；反向仍写 $dZ = A-Y$。

---

## 13. 公式汇总

前向：

$$
z = w^{\mathsf T} x + b, \qquad a = \frac{1}{1+e^{-z}}, \qquad \sigma'(z) = a(1-a)
$$

一个样本：

$$
L = -\, y\ln a - (1-y)\ln(1-a)
$$

$m$ 个样本：

$$
J = \frac{1}{m}\sum_{i=1}^{m} L_i
$$

反向：

$$
\frac{\partial L}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}
$$

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a}\cdot a(1-a) = a - y
$$

$$
\frac{\partial L}{\partial w} = x(a-y), \qquad \frac{\partial L}{\partial b} = a-y
$$

批量：

$$
\frac{\partial J}{\partial w}
= \frac{1}{m}\sum_{i=1}^{m} x^{(i)}\left(a^{(i)}-y^{(i)}\right)
= \frac{1}{m}\, X\, dZ^{\mathsf T}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\left(a^{(i)}-y^{(i)}\right)
$$

更新：

$$
w := w - \alpha\frac{\partial J}{\partial w}, \qquad b := b - \alpha\frac{\partial J}{\partial b}
$$

上一层才使用：

$$
dx = w(a-y)
$$

预测：

$$
\hat{y} =
\begin{cases}
1, & w^{\mathsf T} x + b \ge 0 \\
0, & w^{\mathsf T} x + b < 0
\end{cases}
$$