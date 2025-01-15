#!/bin/bash
# 上传到少林派命令
scp -r ../yolov8_seg_Xray/ linaro@192.168.10.1:/data

netstat -tlnp | grep 6666
kill 6666

sudo vim /etc/rc.local

/home/linaro/yolov8_opencv_tcp_server.sh

# 后台运行python文件
nohup python3 -u yolov8_opencv_tcp_server.py > yolov8_opencv_tcp_server.log 2>&1 &

# 查看运行python文件
ps -ef | grep python

cd  /data/yolov8_seg_Xray/python/
python3 yolov8_opencv_tcp_server.py
python3 /data/yolov8_seg_Xray/python/yolov8_opencv_tcp_server.py
# 创建软链接
sudo ln -s /lib/systemd/system/rc.local.service /etc/systemd/system/