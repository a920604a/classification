'''
Author: yuan
Date: 2021-02-22 10:06:21
LastEditTime: 2021-03-10 11:10:07
FilePath: /yuan-algorithm/image-classification/database/op_db.py
'''
import datetime
from contextlib import contextmanager

import pymysql
from conf import MYSQL
from struction import db_data

from database.mysql_db import Detection


def connectDb(dbName):
    try:
        mysqldb = pymysql.connect(
            host=MYSQL['HOST'],
            user=MYSQL['USER'],
            passwd=MYSQL['PASSWD'],
            database=MYSQL['DB'],)
        return mysqldb
    except Exception as e:
        print('###################connectDb############################')
        print(e)
    return None


@ contextmanager
def write_db(data: db_data, Session, conn):
    detector = Detection()
    detector.model_name = data.model_name
    detector.site_name = data.site_name
    detector.product_name = data.product_name
    detector.job_name = data.job_name
    detector.layer_name = data.layer_name
    detector.lot_name = data.lot_name
    detector.panel_id = data.panel_id
    detector.serial_number = data.serial_number
    detector.image_name = data.image_name
    detector.process_time = data.process_time
    detector.detection_path = data.detection_path
    detector.source_path = data.source_path
    detector.reference_path = data.reference_path
    detector.true_label = data.true_label
    detector.description = data.description
    detector.detection_class = data.detection_class
    detector.confidence = data.confidence
    # detector.create_at = data.create_at,
    detector.create_by = data.create_by,
    # detector.update_at = datetime.datetime.now(),
    detector.update_by = data.create_by

    session = Session(bind = conn)
    try:
        yield
    except Exception as e:
        print(e)
        session.rollback()
    else:
        session.add(detector)
        session.commit()
    finally:
        session.close()
