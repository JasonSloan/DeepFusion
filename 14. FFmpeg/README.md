编译安装ffmpeg

```bash
apt update && apt install nasm
git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg
cd ffmpeg
git checkout n7.0  # 切换到7.0版本的
./configure --prefix=/path/to/your/install/dir \
            --enable-shared \
            --disable-static \
            --extra-ldflags=-Wl,-rpath=\$\$ORIGIN/../lib
make ${nproc} install
```

一些简单用法

```bash
# 截取保存视频片段
ffmpeg -i input.mp4 -ss 00:01:00 -to 00:02:00 -c:v libx264 -c:a aac output.mp4

# 指定一个时间第3分19秒, 获取该时间所在整个视频的第几帧(大约)
ffprobe -read_intervals "%+03:19" -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 input.mp4

# 使用ffplay播放视频流
ffplay -rtsp_transport tcp rtsp://192.168.103.241:1935/live/724-2
```

