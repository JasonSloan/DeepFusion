假如想在一个.c文件中调用如下的.cpp的代码中的某个函数, 需要将.cpp代码的函数声明和定义使用一段代码包裹上, 具体如下:

```c++
//test.cpp
 
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
 
int test_func(void)
{
    printf("test\n");
}
 
#ifdef __cplusplus
}
#endif /* __cplusplus */
 
 
 
//test.h
 
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
 
int test_func(void);      //在cpp文件中定义，在c文件中要调用的函数
                        
#ifdef __cplusplus
}
#endif /* __cplusplus */
 
这样才能在c文件中调用cpp文件的函数，要不编译c文件时出现未定义的错误
```

