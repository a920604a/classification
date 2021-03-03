'''
Author: yuan
Date: 2021-02-24 09:44:41
LastEditTime: 2021-03-03 10:46:58
FilePath: /aidc-algorithm/image-classification/prediction_v2.py
'''

import os
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from conf.config import MYSQL
from conf.cfgnode import get_cfg
from struction import Config
from database.op_db import write_db, connectDb
from utils.utils import relat2abs_path, make_result_path, \
    mount2local, get_config, get_job_names, delta_time
from utils.tricks import filter_ts
from utils.loader import get_test_dataloader
from save_data import Save_data
from predictor import VisualizationDemo


def get_all_files(path):
    ret = []
    for root, dirname, files in os.walk(path):
        for f in files:
            ret.append(os.path.join(root, f))
    return sorted(ret)


def get_pannel_folders(path):
    ret = set()
    for root, dirname, files in os.walk(path):
        for f in files:
            ret.add(root)
            break
    return sorted(list(ret))  # using order


def make_labels_path(labels: list, img_path, ret_path):

    for l in labels:
        os.makedirs(os.path.join(img_path, l), exist_ok=True)
        os.makedirs(os.path.join(ret_path, l), exist_ok=True)


def batch_filter_ts(code_convert: dict,  pres: list, probs: list, focus_labels: dict):
    ret_pre = []  # List[str]
    ret_prob = []  # List[numpy.float32]

    for p, b in zip(pres, probs):
        pre, prob = filter_ts(code_convert,  int(p), b, focus_labels)
        ret_pre.append(pre)
        ret_prob.append(prob)
    return ret_pre, ret_prob


def task(demo, b_img, b_img_path, b_ref_path, focus_labels, code_convert):
    pres, probs = demo.run_on_image(b_img)
    pred, prob = batch_filter_ts(
        code_convert,  pres, probs, focus_labels)

    return pred, prob, b_img_path, b_ref_path


def run(demo, save_data,  img_f, ref_f,
        batch_size, img_size,
        focus_labels, code_convert, th):

    engine = create_engine(
        'mysql+pymysql://{}:{}@{}:{}/{}'.format(
            MYSQL['USER'],
            MYSQL['PASSWD'],
            MYSQL['HOST'],
            MYSQL['PORT'],
            MYSQL['DB'],

        ), echo=False)
    Session = sessionmaker(bind=engine)

    dataloader = get_test_dataloader(
        img_f,
        ref_f,
        batch_size=batch_size,
        num_workers=0,
        size=img_size,
    )
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
                with write_db(d, Session):
                    print("{} is write in database".format(d.image_name))
        pool.shutdown()
        del pred, prob
    # for b_img, b_img_path, b_ref_path in dataloader:
    #     task(demo, save_data,
    #          b_img, b_img_path, b_ref_path,
    #          focus_labels, code_convert
    #          )


def main():
    print('--------------start predictor services--------------------')
    config = get_config(relat2abs_path('./config.json'))

    cfg = Config(config)

    user = cfg.create_by
    description = cfg.description
    th = cfg.thread_pool

    print(config)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
    #     [str(g) for g in config.get('gpus')])

    result_img_path, result_ref_path = make_result_path(cfg.result_path)
    make_labels_path(description.keys(), result_img_path, result_ref_path)
    # while 1:
    img_path, ref_path = mount2local(cfg.mount_folder_path,
                                     cfg.input_folder_path)

    print('--------------mount2local finished--------------------')

    # dist = DistributedPredictorModel(gpu_list=config['gpus'])

    job_names = sorted(get_job_names(img_path))

    while len(job_names) >= 1:
        # for gid in range(2):
        job_name = job_names.pop(0)
        model, job = cfg.get_model_config(job_name)
        config_node = get_cfg(model)
        code_convert = {i: c for i, c in enumerate(config_node.LABLE)}
        focus_labels = model.focus_labels
        save_data = Save_data(config_node.NET,
                              job_name, job,
                              description, user,
                              result_img_path, result_ref_path)

        imgs = get_pannel_folders(os.path.join(img_path, job_name))
        refs = get_pannel_folders(os.path.join(ref_path, job_name))

        # model_predictor_list = dist.distribute(config_node)
        while len(imgs) > 0 and len(refs) > 0:
            # for i in range(1):
            # demo = VisualizationDemo(config_node, gid=i)
            # run(VisualizationDemo(config_node, gid=0), save_data,
            #     img_folder_list[-1], ref_folder_list[-1],
            #     cfg.batch_size, config_node.IMAGE_SIZE,
            #     focus_labels, code_convert)

            with mp.Pool(processes=1) as pool:
                multiple_results = [pool.apply_async(run, (
                    VisualizationDemo(config_node, gid=i), save_data,
                    imgs.pop(), refs.pop(),
                    cfg.batch_size, config_node.IMAGE_SIZE,
                    focus_labels, code_convert, th)
                )
                    for i in range(1)]
                [res.get() for res in multiple_results]

        # shutil.rmtree(os.path.join(img_path, job_name))
        # shutil.rmtree(os.path.join(ref_path, job_name))


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    engine = create_engine(
        'mysql+pymysql://{}:{}@{}:{}/{}'.format(
            MYSQL['USER'],
            MYSQL['PASSWD'],
            MYSQL['HOST'],
            MYSQL['PORT'],
            MYSQL['DB'],

        ), echo=False)

    inspector = Inspector.from_engine(engine)
    if ('t_detection' in inspector.get_table_names()) is False:
        print('--------------------------------create table--------------------')
        # create table
        with connectDb('t_detection') as conn:
            print(dir(conn))
            if conn is not None:
                cur = conn.cursor()
            if cur is not None:
                # Note:  the current path is /workspace
                ff = open(
                    relat2abs_path('./database/create.sql'), 'r').read()
                sql = ff
                cur.execute(sql)
                print('# create table')
    # Session = sessionmaker(bind=engine)

    main()
