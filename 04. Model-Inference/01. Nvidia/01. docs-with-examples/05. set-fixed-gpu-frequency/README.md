[参考链接](https://stackoverflow.com/questions/64701751/can-i-fix-my-gpu-clock-rate-to-ensure-consistent-profiling-results/64729952#64729952)

```bash
# 设置成persistence mode
nvidia-smi -i 0 -pm 1    # -i代表第几号卡，-pm代表persistence mode
# 查询支持的memory graphics设置值（只能从这些“对”中选择，不能自己组合）
nvidia-smi -i 0 --query-supported-clocks=mem,gr --format=csv 
# 设置application定频
nvidia-smi -i 0 -ac 877,1215  # 将877和1215设置成上面命令查询到的“对”值
# --------------------------------参数解释--------------------------------
-ac   --applications-clocks= Specifies <memory,graphics> clocks as a
                                pair (e.g. 2000,800) that defines GPU's
                               speed in MHz while running applications on a GPU.
-rac  --reset-applications-clocks
                           Resets the applications clocks to the default values.
# ------------------------------------------------------------------------
# 设置gpu定频
nvidia-smi -i 0 -lgc 1215,1215  # 将1215设置成上面命令查询到的第二个值
# --------------------------------参数解释--------------------------------
-lgc  --lock-gpu-clocks=    Specifies <minGpuClock,maxGpuClock> clocks as a
                                pair (e.g. 1500,1500) that defines the range
                                of desired locked GPU clock speed in MHz.
                                Setting this will supercede application clocks
                                and take effect regardless if an app is running.
                                Input can also be a singular desired clock value
                                (e.g. <GpuClockValue>).
-rgc  --reset-gpu-clocks
                            Resets the Gpu clocks to the default values.
# ------------------------------------------------------------------------
```

