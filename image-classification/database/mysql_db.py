'''
Author: yuan
Date: 2021-03-03 11:54:28
LastEditTime: 2021-03-03 11:57:12
FilePath: /classification/image-classification/database/mysql_db.py
'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author         :  james
@Version        :
------------------------------------
@File           :  BaseModel.py
@Description    :  ORM封装
@CreateTime     :  2020/3/8 15:13
------------------------------------
@ModifyTime     :
"""
from sqlalchemy import func,  Column, Integer, String, Float, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData

Base = declarative_base()


class BaseModel(Base):
    __abstract__ = True  # 声明当前类为抽象类，被继承，调用不会被创建
    id = Column(Integer, primary_key=True, autoincrement=True)
    create_by = Column(String(64), comment="创建者")
    created_at = Column(TIMESTAMP(True), comment="创建时间",
                        nullable=False, server_default=func.now())
    update_by = Column(String(64), comment="更新者")
    updated_at = Column(TIMESTAMP(True), comment="更新时间", nullable=False, server_default=func.now(),
                        onupdate=func.now())
    remark = Column(String(500), comment="备注")


class Detection(BaseModel):
    """
    预测记录
    """
    __tablename__ = "t_detection"
    
    model_name = Column(String(100), unique=True, comment="模型名称")
    product_name = Column(String(100), unique=True, comment="产品名称")
    site_name = Column(String(100), unique=True, comment="站点名称")
    job_name = Column(String(100), comment="")
    layer_name = Column(String(100), comment="")
    lot_name = Column(String(100), comment="批号")
    panel_id = Column(String(100), comment="")
    serial_number = Column(String(100), comment="序号")
    process_time = Column(TIMESTAMP(
        True), comment="检测时间", nullable=False, server_default=func.now())
    image_name = Column(String(500), comment="图片名称")
    source_path = Column(String(500), comment="原图片路径")
    reference_path = Column(String(500), comment="标准图片路径")
    detection_path = Column(String(500), comment="检测图片路径")
    detection_class = Column(String(50), comment="检测类别")
    true_label = Column(String(50), comment="真实标签")
    confidence = Column(Float, comment="置信度")

    description = Column(String(100), comment="描述")
    status = Column(Integer, default=1, comment="状态（1正常 2停用）")
