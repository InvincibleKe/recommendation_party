#FROM python:3.7
#ADD requirements.txt /
#RUN pip install -r requirements.txt
#ADD . /app
#WORKDIR /app
#EXPOSE 8000
#CMD ["python", "fserver.py"]
# 指定基础镜像
FROM python:3.7
# 设置工作目录
WORKDIR /root/recommendation_party
ADD requirements.txt /
# 安装项目依赖项
RUN pip install -r requirements.txt
# 为启动脚本添加执行权限
RUN chmod 755 start.sh
# 容器启动时要执行的命令
ENTRYPOINT ["./start.sh"]
# 暴露端口
EXPOSE 8000