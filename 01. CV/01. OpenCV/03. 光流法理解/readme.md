## 1. 前置条件

* 当像素从帧 1（记为 `I₁`）移动到帧 2（记为 `I₂`）且位移为`(dx, dy)`时，该像素的亮度（或 “强度”）保持不变，数学表达式为：
  $$I₁(x,y) = I₂(x+dx, y+dy)$$
* 相邻像素具有相似的运动（特性）。

## 2. 算法推导

考虑第一帧中的一个像素 `I(x,y,t)`（注意：此处新增了一个维度 ---- 时间。此前我们仅处理图像，因此无需引入时间维度）。该像素在间隔 `dt` 时间拍摄的下一帧中，移动了距离 `(dx, dy)`。由于这些像素本质上是同一个像素，且其亮度（或 “强度”）保持不变，因此我们可以得出：

$$I(x,y,t) = I(x+dx, y+dy, t+dt)$$

假设移动很小，使用泰勒公式可以表示成：

$$I(x+dx, y+dy, t+dt) = I(x, y, t) + \frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt + H.O.T.$$

$ H.O.T $ 是高阶无穷小, 此处忽略

由上面的假设和使用泰勒公式展开的式子可以得到：

$\frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt = 0$

除以 $ dt $ 得

$I_x u + I_y u + I_t = 0$

其中, $I_x = \frac{\partial I}{\partial x}; \, I_y = \frac{\partial I}{\partial y}; \, I_t = \frac{\partial I}{\partial t}; \, u = \frac{dx}{dt}; \, v = \frac{dy}{dt}$

上述方程称为光流方程，$ I_x $ 和 $ I_y $ 是图像的梯度，$ I_t $ 是沿时间的梯度。但是 `u` 和 `v` 是未知的，我们无法用一个方程解两个未知数，那么就有 Lucas-Kanade 方法来解决这个问题

我和可以假设在一个大小为`mxm`的窗口内，图像的光流是一个恒定值。那么就可以得到以下方程组:

$I_{x1} u + I_{y1} v = -I_{t1} \\$
$I_{x2} u + I_{y2} v = -I_{t2} \\$
$\quad\quad\quad\vdots \\$
$I_{xn} u + I_{yn} v = -I_{tn}$

为了求解以上过度约束的系统可以采用最小二乘法对以上的方程还是进行最小化，将以上方程采用矩阵形式进行表示：
$$
\begin{bmatrix}
I_{x1} & I_{y1} \\
I_{x2} & I_{y2} \\
\vdots & \vdots \\
I_{xn} & I_{yn}
\end{bmatrix}
\begin{bmatrix}
u \\
v
\end{bmatrix}
=
\begin{bmatrix}
-I_{t1} \\
-I_{t2} \\
\vdots \\
-I_{tn}
\end{bmatrix}
$$
记作$A\vec{V} = -b$, 最终使用最小二乘法求解$\vec{V} = (A^T A)^{-1} A^T (-b)$

## 3. 实现流程

1. 读取图像：获取连续两帧图像（作为输入的前后处理图像序列）。
2. 转换为灰度图：将输入图像转换为单通道灰度图，保留亮度信息。
3. 低通滤波（高斯平滑）：对两帧灰度图分别应用高斯滤波，去除噪声，得到平滑图像。
4. 计算梯度：
   - 用特定卷积核(例如sobel算子)计算 x 方向梯度（Ix）。
   - 用特定卷积核(例如sobel算子)计算 y 方向梯度（Iy）。
   - 用特定卷积核(例如sobel算子)计算时间梯度（It）：$I_t(x, y) \approx \frac{I_2(x, y) - I_1(x, y)}{\Delta t}$
5. 特征点检测：使用Harris角点检测算法在第一帧平滑图像上检测角点作为跟踪特征点。
6. 初始化光流向量：创建与图像尺寸一致的光流向量数组（u、v），初始值为 nan。
7. 计算光流向量：
   - 对每个特征点，提取其 3x3 邻域内的 Ix、Iy、It 值。
   - 构造矩阵并通过最小二乘法（计算伪逆等）求解光流约束方程，得到该特征点的光流向量（u、v）。

## 4. 代码实现

[opencv光流法](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)

