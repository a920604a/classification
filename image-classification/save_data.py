'''
Author: yuan
Date: 2021-02-25 15:33:32
LastEditTime: 2021-03-03 09:37:58
FilePath: /aidc-algorithm/image-classification/save_data.py
'''
import os
import shutil
import datetime
from database.op_db import write_db
from struction import db_data
from utils.utils import get_folder_struct, abs2relat_path
from contextlib import contextmanager


class Save_data(object):
    def __init__(self, model_name, job_name, job: dict, description, user, img_path, ref_path):
        # self.session = session
        self.model_name = model_name
        self.job_name = job_name
        self.site_name = job.get('site_name')
        self.product_name = job.get('product_name')
        self.description = description
        self.user = user

        self.img_path = img_path
        self.ref_path = ref_path

    def save_on_images(self, preds, probs, files_path, references_path, delta_t):
        ret = []
        for p, b, f, r in zip(preds, probs, files_path, references_path):
            ret.append(self.save_on_image(p, b, f, r, delta_t))
        return ret

    def save_on_image(self, pred, prob, file_path, reference_path, delta_t):
        img_dir, img_name = os.path.split(file_path)
        _, layer_name, lot_name, pannel_id = get_folder_struct(
            img_dir, self.job_name)

        result_img_folder = os.path.join(self.img_path, pred)
        result_ref_folder = os.path.join(self.img_path, pred)
        data_db = db_data(
            model_name=self.model_name,
            site_name=self.site_name,
            product_name=self.product_name,
            job_name=self.job_name,
            layer_name=layer_name,
            lot_name=lot_name,
            panel_id=pannel_id,
            serial_number="I don't know",
            image_name=img_name,
            process_time=datetime.datetime.now() + delta_t,
            source_path=abs2relat_path(result_img_folder),
            reference_path=abs2relat_path(result_ref_folder),
            detection_path=abs2relat_path(result_img_folder),
            detection_class=pred,
            true_label=pred,
            description=self.description[pred],
            confidence=prob,
            create_by=self.user,
            # create_at=datetime.datetime.now(),
            update_by=self.user
        )
        return data_db
        # with write_db(data_db, self.session):
        #     print("{} is write in database".format(data_db.image_name))

        # shutil.copy(file_path, os.path.join(result_img_folder, img_name))

        # shutil.copy(reference_path, os.path.join(result_ref_folder, img_name))

        # os.remove(file_path)
        # os.remove(reference_path)
