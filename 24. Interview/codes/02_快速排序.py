# 多看两遍视频,动笔模拟一下过程能理解

def quick(l, start=0, end=None):
    # 第一次进入函数,设置end值
    if end is None:
        end = len(l)
    # 设置递归退出条件
    if end - start < 1:
        return

    # 排序
    # 设置基准值索引
    basic = start
    # 设置遍历起始点为基准值后偏移一个位置
    # 其中j为遍历基准值后所有值用
    i = j = basic + 1
    # 开始遍历
    while j < end:
        if l[j] < l[basic]:
            l[i], l[j] = l[j], l[i]
            i += 1
        j += 1
    # 最终再将基准值与i索引前面那一个值交换
    l[basic], l[i - 1] = l[i - 1], l[basic]

    quick(l, start, i - 1)
    quick(l, i, end)


if __name__ == '__main__':
    l = [2, 3, 1, 5, 4, 6, 7]
    quick(l)
    print(l)
