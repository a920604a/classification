'''
Author: yuan
Date: 2021-02-24 09:44:41
LastEditTime: 2021-03-10 18:02:28
FilePath: /classification/image-classification/prediction_v2.py
'''

import multiprocessing as mp
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import create_engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import sessionmaker

from conf.cfgnode import get_cfg
from conf.config import MYSQL
from database.op_db import connectDb, write_db
from predictor import VisualizationDemo
from save_data import Save_data
from struction import Config
from utils.loader import get_test_dataloader
from utils.tricks import batch_filter_ts
from utils.utils import (delta_time, get_config, get_job_names,
                         get_pannel_folders, make_labels_path,
                         make_result_path, mount2local, relat2abs_path)

# we create the engine
engine = create_engine(
    'mysql+pymysql://{}:{}@{}:{}/{}'.format(
        MYSQL['USER'],
        MYSQL['PASSWD'],
        MYSQL['HOST'],
        MYSQL['PORT'],
        MYSQL['DB'],

    ), pool_size=5, echo=False)
# at the module level, the global sessionmaker,
# bound to a specific Engine
Session = sessionmaker(engine)
conn = engine.connect()


def task(demo, b_img, b_img_path, b_ref_path, focus_labels, code_convert):
    pres, probs = demo.run_on_image(b_img)
    pred, prob = batch_filter_ts(
        code_convert,  pres, probs, focus_labels)

    return pred, prob, b_img_path, b_ref_path


def run(demo, save_data,  img_f, ref_f,
        batch_size, img_size,
        focus_labels, code_convert, th):

    # session = Session(bind=conn)

    dataloader = get_test_dataloader(
        img_f,
        ref_f,
        batch_size=batch_size,
        num_workers=0,
        size=img_size,
    )

    # for b_img, b_img_path, b_ref_path in dataloader:
    #     pred, prob, b_img_path, b_ref_path = task(demo,
    #                                               b_img, b_img_path, b_ref_path,
    #                                               focus_labels, code_convert
    #                                               )
    #     data_db = save_data.save_on_images(
    #         pred, prob, b_img_path, b_ref_path, delta_time())

    #     for d in data_db:
    #         with write_db(d, Session) :
    #             print("{} is write in database".format(d.image_name))
    with ThreadPoolExecutor(max_workers=th) as pool:
        threads = [
            pool.submit(
                task, demo,
                b_img, b_img_path, b_ref_path,
                focus_labels, code_convert
            )
            for b_img, b_img_path, b_ref_path in dataloader]

        for thr in as_completed(threads):
            pred, prob, b_img_path, b_ref_path = thr.result()
            data_db = save_data.save_on_images(
                pred, prob, b_img_path, b_ref_path, delta_time())
            for d in data_db:
                with write_db(d, Session, conn):
                    print("{} is write in database".format(d.image_name))
            del pred, prob
        pool.shutdown()


def main():
    print('--------------start predictor services--------------------')
    config = get_config(relat2abs_path('./config.json'))

    cfg = Config(config)

    user = cfg.create_by
    description = cfg.description
    th = cfg.thread_pool
    processors = len(cfg.gpus)
    gpus = cfg.gpus
    print(config)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
    #     [str(g) for g in config.get('gpus')])

    result_img_path, result_ref_path = make_result_path(cfg.result_path)
    make_labels_path(description.keys(), result_img_path, result_ref_path)
    while 1:
        img_path, ref_path = mount2local(cfg.mount_folder_path,
                                         cfg.input_folder_path)
        print('--------------mount2local finished--------------------')

        job_names = sorted(get_job_names(img_path))

        print(job_names)
        multiple_results = []
        with mp.Pool(processes=processors) as pool:
            while len(job_names) >= 1:
                job_name = job_names.pop(0)
                model, job = cfg.get_model_config(job_name)
                config_node = get_cfg(model)  # for multi-process pickle
                code_convert = {i: c for i, c in enumerate(model.labels)}
                focus_labels = model.focus_labels
                save_data = Save_data(model.net,
                                      job_name, job,
                                      description, user,
                                      result_img_path, result_ref_path)

                imgs = get_pannel_folders(os.path.join(img_path, job_name))
                refs = get_pannel_folders(os.path.join(ref_path, job_name))

                assert len(imgs) == len(
                    refs), f'please {len(imgs) and {len(refs)}}'
                print(len(imgs), imgs)
                while len(imgs) >= 1 and len(refs) >= 1:
                    for i in gpus:
                        #  for one gpu
                        # run(
                        #     VisualizationDemo(
                        #         config_node, gid=i), save_data,
                        #     imgs.pop(0), refs.pop(0),
                        #     cfg.batch_size, model.image_size,
                        #     focus_labels, code_convert, th)

                        multiple_results.append(pool.apply_async(run, (
                            VisualizationDemo(config_node, gid=i), save_data,
                            imgs.pop(0), refs.pop(0),
                            cfg.batch_size, model.image_size,
                            focus_labels, code_convert, th)
                        )
                        )
                    [res.get() for res in multiple_results]

                shutil.rmtree(os.path.join(img_path, job_name))
                shutil.rmtree(os.path.join(ref_path, job_name))


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    inspector = Inspector.from_engine(engine)
    if ('t_detection' in inspector.get_table_names()) is False:
        print('--------------------------------create table--------------------')

        # get a connection
        with connectDb('t_detection') as conn:
            print(dir(conn))
            print(conn)
            if conn is not None:
                # use it by cursor
                cur = conn.cursor()
            if cur is not None:
                # Note:  the current path is /workspace
                # create table
                sql = open(
                    relat2abs_path('./database/create.sql'), 'r').read()
                cur.execute(sql)
                print('# create table')
    main()
