"""
47. 全排列 II
给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列

示例 1：
输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]

示例 2：
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
"""

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        path = [0] * n
        def traceback(i, s):
            if i == n:
                if path not in res:
                    res.append(path.copy())
                return
            else:
                for x in s:
                    path[i] = x
                    # 这里要将s拷贝一下，要不在最外层大循环的第二次循环s就被remove空了
                    t = s.copy()
                    t.remove(x)
                    traceback(i+1, t)
        traceback(0, nums)
        return res