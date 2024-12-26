"""
视频讲解地址：https://www.bilibili.com/video/BV1mY411D7f6/?vd_source=2fa3840975cc19817a9a15ddf8a1a81b
在哔哩哔哩-->收藏-->经典数据结构与算法中已收藏

46. 全排列
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案

示例 1：
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

示例 2：
输入：nums = [0,1]
输出：[[0,1],[1,0]]

示例 3：
输入：nums = [1]
输出：[[1]]
"""
from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        # 计算数组长度
        n = len(nums)
        # 初始化结果为空
        res = []
        # 初始化待排列的列表
        path = [0] * n

        # traceback函数代表从第i个索引值开始，剩余未排列的数组为s
        def traceback(i, s):
            # 如果索引值i已经等于长度，说明排列完了
            # 将结果添加即可，注意要将path进行一个copy
            if i == n:
                res.append(path.copy())
                return
            # 如果索引值不等于长度，那么说明没排列完
            # 那么遍历剩余未排列的元素s，对于每一次遍历，都将当前遍历的值排列进path中，然后对剩余的值继续使用递归排列
            else:
                for x in s:
                    path[i] = x
                    traceback(i + 1, s - {x})

        # 初始值从0开始，初始未排列数组为整个数组，转成set是为了在traceback(i+1, s - {x})中将已排列元素剔除方便
        traceback(0, set(nums))
        return res


if __name__ == '__main__':
    Solution().permute([1, 2, 3])
