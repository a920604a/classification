import datetime
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymysql
import torch
from sqlalchemy import create_engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import sessionmaker

from conf.config import MYSQL
from database.op_db import connectDb, write_db
from struction import Child_Cfg, NN_model, db_data
from utils.allocate_gpu import occumpy_mem
from utils.tricks import filter_ts
from utils.utils import (abs2relat_path, delta_time, get_config,
                         get_folder_struct, get_job_names, make_result_path,
                         mount2local, relat2abs_path)

engine = create_engine(
    'mysql+pymysql://{}:{}@{}:{}/{}'.format(
        MYSQL['USER'],
        MYSQL['PASSWD'],
        MYSQL['HOST'],
        MYSQL['PORT'],
        MYSQL['DB'],

    ), pool_size=100, echo=False)
Session = sessionmaker(engine)


def task(loader, net, n_iter, image,
         j_name, code_convert, description, user, job, result_path):
    conn = engine.connect()
    bz = loader.batch_size
    batch_img_path = [
        f for f, _ in loader.dataset.samples[n_iter*bz:(1+n_iter)*bz]]
    delta_t = delta_time()
    # print("iteration: {}\ttotal {} iterations".format(
    #     n_iter, len(loader)))
    with torch.no_grad():
        net.eval()

        if job.model.gpu:
            image = image.cuda()
        output = net(image)
        probs = torch.sigmoid(output)
        _, preds = output.topk(k=1, dim=1, largest=True, sorted=True)

        for p, m, b in zip(preds, batch_img_path, probs):
            img_dir, img_name = os.path.split(m)
            # img_dir = ./datasets/ketmac/img/ycmb18a/CO/test/6480
            if j_name == 'default':
                job_name, layer_name, lot_name, pannel_id = get_folder_struct(
                    img_dir, )
            else:
                job_name, layer_name, lot_name, pannel_id = get_folder_struct(
                    img_dir, j_name)
            pre = int(p.cpu().numpy())

            prob = b.cpu().numpy()
            # print('pre:{},prod:{}'.format(pre, prob))
            pred, prob = filter_ts(code_convert, pre, prob,
                                   job.model.focus_labels)

            reference_path = os.path.join(
                img_dir.replace('img', 'ref'),
                img_name.replace('ins', 'ref'))

            if os.path.exists(reference_path):

                # copy from ketmac to result path
                result_img_folder = os.path.join(result_path[0], job_name,
                                                 layer_name, lot_name,
                                                 pannel_id)
                result_ref_folder = os.path.join(result_path[1], job_name,
                                                 layer_name, lot_name,
                                                 pannel_id)
                os.makedirs(result_img_folder, exist_ok=True)
                os.makedirs(result_ref_folder, exist_ok=True)
                result_img_path = os.path.join(result_img_folder, img_name)
                result_ref_path = os.path.join(result_ref_folder, img_name)
                shutil.copy(m, result_img_path)
                shutil.copy(m, result_ref_path)

                # wirte db fun
                data_db = db_data(
                    model_name=job.model.net,
                    site_name=job.site_name,
                    product_name=job.product_name,
                    job_name=job_name,
                    layer_name=layer_name,
                    lot_name=lot_name,
                    panel_id=pannel_id,
                    serial_number="I don't know",
                    image_name=img_name,  # abs2relat_path(result_img_path),
                    process_time=datetime.datetime.now() + delta_t,
                    source_path=abs2relat_path(result_img_folder),
                    reference_path=abs2relat_path(result_ref_path),
                    detection_path=abs2relat_path(result_img_folder),
                    detection_class=pred,
                    true_label=pred,
                    description=description[pred],
                    confidence=prob,
                    create_by=user,
                    # create_at=datetime.datetime.now(),
                    update_by=user
                )
                with write_db(data_db,  Session, conn):
                    print("{} is write in database".format(data_db.image_name))

                # remove this image if it exists
                os.remove(m)
                os.remove(reference_path)
            else:
                print("Don't find base on reference file:{}".format(
                    relat2abs_path(reference_path)))
    return n_iter * bz


def predict(loader, net, job, user, j_name, description, th, result_path):

    code_convert = {i: c for i, c in enumerate(job.model.labels)}
    print('code_convert', code_convert)
    with ThreadPoolExecutor(max_workers=th) as pool:
        threads = [
            pool.submit(
                task, loader, net, n_iter, image, j_name,
                code_convert, description, user, job, result_path
            )
            for n_iter, (image, _) in enumerate(loader)]
        for thr in as_completed(threads):
            _ = thr.result()
        pool.shutdown()

    if job.model.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')


def main():
    print('--------------start predictor services--------------------')
    config = get_config(relat2abs_path('./config.json'))
    print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        [str(g) for g in config.get('gpus')])
    # occumpy_mem(','.join(
    #     [str(g) for g in config.get('gpus')]), 0.8)
    result_path = make_result_path(config.get('result_path'))
    while 1:
        img_path, ref_path = mount2local(config.get('mount_folder_path'),
                                         config.get('input_folder_path'))
        print('--------------mount2local finished--------------------')

        job_names = get_job_names(img_path)
        cfg = Child_Cfg(config)

        print('job_names', img_path, job_names)
        for job_name in job_names:
            # for data_loader
            print(job_name)
            loader_path = os.path.join(img_path, job_name)
            # avoid ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)
            works = 0
            # find job object using job name from cfg
            if cfg.jobs.get(job_name):
                job = cfg.jobs.get(job_name)
                model = job.model
                nn_model = NN_model(model, cfg.batch_size, loader_path, works)
            else:  # job_name : use default pattern

                job = cfg.jobs.get('default')
                model = job.model
                nn_model = NN_model(model, cfg.batch_size, loader_path, works)

            network = nn_model.get_net()
            test_loader = nn_model.get_loader()
            user = cfg.create_by
            description = cfg.description
            th = cfg.thread_pool

            print('**********start prediction**************')

            predict(test_loader, network, job, user,
                    job_name, description, th, result_path)
            shutil.rmtree(loader_path)
            shutil.rmtree(os.path.join(ref_path, job_name))

    time.sleep(0.5)


if __name__ == '__main__':

    inspector = Inspector.from_engine(engine)
    if ('t_detection' in inspector.get_table_names()) is False:
        print('--------------------------------create table--------------------')
        # create table
        with connectDb('t_detection') as conn:
            # print(dir(conn))
            if conn is not None:
                cur = conn.cursor()
            if cur is not None:
                # Note:  the current path is /workspace
                ff = open(
                    relat2abs_path('./database/create.sql'), 'r').read()
                sql = ff
                cur.execute(sql)
                print('# create table')
    Session = sessionmaker(bind=engine)

    main()
