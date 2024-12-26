"""
45. 跳跃游戏 II
给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。
每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:
0 <= j <= nums[i]
i + j < n
返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。

示例 1:
输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
示例 2:
输入: nums = [2,3,0,1,4]
输出: 2
"""


class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 特殊情况,只有一个元素的列表,直接返回0
        if len(nums) == 1:
            return 0

        # 初始化步数step,初始索引位置i
        step = 0
        i = 0
        n = len(nums)

        def jump_inner(i, step):
            # 当前索引i能到达的最大索引位置为nums[i] + i
            current_max_index = nums[i] + i
            # 如果当前能到达的最大索引位置超过列表最大索引,那么就成功了
            if current_max_index >= n - 1:
                return step + 1
            else:
                # 否则,决定下一步要走多长
                # 走多长使用贪心策略,即下一步应该走的位置满足的条件为:下一步所在位置能到达的最大索引最大


                # 初始化下一步位置
                _j = i + 1
                # 初始化下一步所在位置能到达的最大索引
                j_max_index = nums[j] + j

                # 从当前位置i出发,遍历能到达的最近位置和能到达的最远位置-->
                # 得到最近位置和最远位置之间所有位置能到达的最远位置,即为当前位置i应该走的下一步位置
                for j in range(i + 1, current_max_index + 1):
                    j_current_index = j + nums[j]
                    if j_current_index > j_max_index:
                        j_max_index = j_current_index
                        _j = j

                # 更新从当前位置i出发,应该走到的下一步位置
                i = _j
                step += 1
                return jump_inner(i, step)

        return jump_inner(i, step)
