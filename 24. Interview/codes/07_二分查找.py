"""
二分查找法适用于已排序好的数组,下面这段代码只适用于升序数组
NOTE:使用分治法代码出现错误,一般是边界设置不正确.
二分查找法边界设置规则,因为比较数值大小用的是索引,所以初始右边界为len(l)-1,防止越界
因为求中间索引的时候用的是短除会向下取整,所以在查找右侧区域的时候要使右侧区域的左边界为mid+1而不是mid
--> 例如 search[2, 5] -->右侧区域查找 search[3, 5] --> search[4, 5] --> search[4, 5]跳不出递归了
"""

def binary_search(l, target):
    # 特殊情况, 只有一个值的时候直接返回
    if len(nums) == 1:
        if nums[0] == target:
            return 0
    left = 0
    # 因为比较数值大小用的是索引,所以初始右边界为len(l)-1,防止越界
    right = len(l)-1
    def binary_search_inner(left, right):
        mid = (left + right) // 2
        if target == l[mid]:
            return mid
        elif right == left:
            return "未找到,不存在"
        elif target < l[mid]:
            return binary_search_inner(left, mid)
        else:
            # 因为求中间索引的时候用的是短除会向下取整,所以在查找右侧区域的时候要使右侧区域的左边界为mid+1而不是mid
            return binary_search_inner(mid+1, right)
    return binary_search_inner(left, right)