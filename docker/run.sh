#! /bin/bash
###
# @Author: yuan
# @Date: 2021-02-22 09:33:58
 # @LastEditTime: 2021-03-10 13:16:59
 # @FilePath: /yuan-algorithm/docker/run.sh
###
sudo docker-compose up -d
# sudo docker-compose run --name yuan yuan-algorithm bash
sudo docker-compose logs -f