CREATE TABLE `t_detection` (
    `id` INT(11) NOT NULL AUTO_INCREMENT COMMENT '检测ID',
    `model_name` VARCHAR(100) NULL COMMENT '模型名称',
    `product_name` VARCHAR(100) NULL COMMENT '产品名称',
    `site_name` VARCHAR(100) NULL COMMENT '站点名称',
    `job_name` VARCHAR(100) NULL COMMENT '',
    `layer_name` VARCHAR(100) NULL COMMENT '',
    `lot_name` VARCHAR(100) NULL COMMENT '批号',
    `panel_id` VARCHAR(100) NULL COMMENT '',
    `serial_number` VARCHAR(100) NULL COMMENT '序号',
    `process_time` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP COMMENT '检测时间',
    `image_name` VARCHAR(500) NULL COMMENT '图片名称',
    `source_path` VARCHAR(500) NULL COMMENT '原图片路径',
    `reference_path` VARCHAR(500) NULL COMMENT '标准图片路径',
    `detection_path` VARCHAR(500) NULL COMMENT '检测图片路径',
    `detection_class` VARCHAR(50) NULL COMMENT '检测类别',
    `true_label` VARCHAR(50) NULL COMMENT '真实标签',
    `confidence` FLOAT NULL COMMENT '置信度',
    `description` VARCHAR(100) NULL COMMENT '描述',
    `status` INT(10) NOT NULL DEFAULT 1 COMMENT '状态（1正常 2停用）',
    `create_by` VARCHAR(64) NULL DEFAULT '' COMMENT '创建者',
    `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_by` VARCHAR(64) NULL DEFAULT '' COMMENT '更新者',
    `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `remark` VARCHAR(500) NULL DEFAULT NULL COMMENT '备注',
    PRIMARY KEY (`id`) USING BTREE
)
COMMENT='检测记录表'
COLLATE='utf8_general_ci'
ENGINE=InnoDB
AUTO_INCREMENT=1;