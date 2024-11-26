# gunicorn_config.py
bind = "0.0.0.0:8888"  # 定义Gunicorn监听的端口
workers = 4  # 工作进程数