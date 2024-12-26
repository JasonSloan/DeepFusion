```bash
# 首先使用gdb调试可执行文件
gdb ./your_program

# 进入到与gdb交互界面后, 输入run
(gdb) run

# 程序崩溃到segmentation fault的位置
# 输入generate-core-file生成core文件, 会在当前文件夹下生成一个core文件(通常名字就叫core)
(gdb) generate-core-file

# CTRL+Z退出gdb
# 再次调试, 连同生成的core文件
gdb ./your_program core

# 程序崩溃到segmentation fault的位置
# 输入bt, 显示实际崩溃的代码信息
(gdb) bt
```

