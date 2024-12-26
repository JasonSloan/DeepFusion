# 代码比较难理解,debug看一下吧

# 代码执行过程:先执行上面的分治递归,一直递归到最深,
# 然后再执行下面的合并代码,由内而外,合并结束一层一层向外return

def recursion(l):
    # 设置递归退出条件,即空列表和只有一个元素的列表返回自己
    if len(l) < 2:
        return l

    # 分治
    left = 0
    right = len(l)
    mid = (left + right) // 2

    # 开始递归
    left = recursion(l[left:mid])
    right = recursion(l[mid:right])

    # 计算左右表长度
    len_left = len(left)
    len_right = len(right)

    # 初始化左右表起始索引坐标
    i, j = 0, 0

    # 初始化result
    result = []
    # 合并左右
    while i < len_left and j < len_right:
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # 如果两个表长度不一样,再将两个表剩余部分加到result中
    result += left[i:]
    result += right[j:]

    return result


if __name__ == '__main__':
    l = [6, 5, 4, 3, 2, 1]
    result = recursion(l)
    print(result)
