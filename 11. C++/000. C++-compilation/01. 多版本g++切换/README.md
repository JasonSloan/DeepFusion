**步骤一: 安装编译器**

```bash
sudo apt update
sudo apt install build-essential
sudo apt -y install g++-8 g++-9 g++-10
```

**步骤二：使用update-alternatives工具创建多个g++编译器选项**

```C++
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
```

**步骤三：使用update-alternatives工具改变默认编译器**

```bash
sudo update-alternatives --config g++
There are 3 choices for the alternative g++ (providing /usr/bin/g++).

  Selection    Path            Priority   Status
------------------------------------------------------------
* 0            /usr/bin/g++-9   9         auto mode
  1            /usr/bin/g++-10  10         manual mode
  2            /usr/bin/g++-8   8         manual mode
  3            /usr/bin/g++-9   9         manual mode

Press  to keep the current choice[*], or type selection number:
```

**步骤四：查看版本**

```bash
gcc --version
```



