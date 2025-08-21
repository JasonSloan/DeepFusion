### 一. 爱因斯坦表示法

[参考链接1](https://stackoverflow.com/questions/26089893/understanding-numpys-einsum) 

[参考链接2](https://www.youtube.com/watch?v=CLrTj7D2fLM)

如果有两个矩阵A和B，如果我们想将矩阵做一个矩阵乘法，再沿着某一个维度求和， 比如

```python
A = np.array([0, 1, 2])

B = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
```

```python
>>> (A[:, np.newaxis] * B).sum(axis=1)
array([ 0, 22, 76])
```

而用爱因斯坦表示法来写：

```python
>>> np.einsum('i,ij->i', A, B)
array([ 0, 22, 76])
```

解释一下i和j：

A是一维矩阵用i表示，B是二维矩阵用ij表示，i在两个矩阵中重复出现代表要在这个维度上做element-wise相乘；j在B矩阵中出现，但是在箭头右侧没出现，代表要在这个j这个维度上求和。

在这里，i被称为dummy index，j被称为free index；

dummy index在箭头左侧是可以重复出现的，但是free index在左侧是不可以重复出现的， dummy index代表要在这个维度上做逐元素相乘。特别注意的是任何一个符号在箭头左侧不能出现超过2次。

再比如：

```python
'i,ij->ij'代表的意思是在i维度上做逐元素相乘，在j维度上不相加；

'i,ij->'代表的意思是在i维度上做逐元素相乘，在i维度上相加后在j维度上相加，得到一个标量；

'ij,jk->ik'代表的意思是在j维度上做逐元素相乘并求和，也就是代表两个二维矩阵做矩阵乘法；

'i->'代表的意思是在i维度上求和，得到一个标量；

'i,i->i'代表的意思是在i维度上做逐元素相乘；

'i,i->'代表的意思是在i维度上做逐元素相乘，然后求和得到一个标量；

'i,j->ij'代表的意思是两个一维向量做外积；

'ijkl, ijtl->iktl'代表的意思是两个矩阵在第1和2维度上做矩阵乘法， 实际例子如下：
    >>> a = torch.randn(2,3,4,5)
    >>> b = torch.randn(2,3,4,5)
    >>> torch.einsum('ijkl, ijtl->iktl',a,b)		# 虽然a矩阵和b矩阵在1和2维度上是相等的，但是因为是要在这两个维度上做矩阵乘法，所以在两个表达式中也必须写为不同的符号，即jk和jt，不允许都写成jk
```

## 二. 多维高斯分布

[参考链接](https://blog.csdn.net/weixin_38468077/article/details/103508072?ops_request_misc=&request_id=&biz_id=102&utm_term=2d%E9%AB%98%E6%96%AF&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-103508072.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187)



