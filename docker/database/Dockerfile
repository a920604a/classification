FROM mysql:latest
LABEL maintainer="a920604a@gmail.com"
ENV MYSQL_ROOT_PASSWORD root
ENV MYSQL_DATABASE devops
ENV MYSQL_USER yuan
ENV MYSQL_PASSWORD yuan12345
# 用本地的mysql.cnf配置来替换mysql.cnf镜像里的默认配置
COPY ./mysql.cnf /etc/mysql/conf.d/mysql.cnf
