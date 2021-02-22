from sqlalchemy.engine.reflection import Inspector
import os
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from conf import MYSQL
import shutil
from collections import namedtuple
import torch
import datetime
import sqlalchemy
from struction import db_data, Child_Cfg, Cfg, NN_model
import pymysql
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import filter_ts
import time
from database.op_db import write_db


def get_config(path):
    with open(path) as f:
        model = json.loads(f.read())
    return model


def copy_img_list(src_path, des_path):
    ret_ins = list()
    ret_ref = list()
    for (root, _, files) in os.walk(src_path):
        for f in files:
            if f.endswith('_ins.bmp'):
                img_path = os.path.join(des_path, 'img')
                ins = os.path.join(root, f)
                ret_ins.append(ins)
                des = ins.replace(src_path, img_path)
                os.makedirs(os.path.split(des)[0], exist_ok=True)
                # shutil.copy(ins, des)
                shutil.move(ins, des)

            elif f.endswith('_ref.bmp'):
                ref_path = os.path.join(des_path, 'ref')

                ref = os.path.join(root, f)
                ret_ref.append(ref)
                des = ref.replace(src_path, ref_path)
                os.makedirs(os.path.split(des)[0], exist_ok=True)
                # shutil.copy(ref, des)
                shutil.move(ref, des)

    return ret_ins, ret_ref


def mount2local(src_path, des_path):
    os.makedirs(des_path, exist_ok=True)
    img_path = os.path.join(des_path, 'img')
    ref_path = os.path.join(des_path, 'ref')
    print(img_path)
    if not os.path.exists(img_path):
        print('***************copy image file****************')
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(ref_path, exist_ok=True)

    ret_ins, ret_ref = copy_img_list(src_path, des_path)
    print('ret_ins, ret_ref ', ret_ins, ret_ref)
    assert len(ret_ins) == len(
        ret_ref), 'please check number of insepection fiels and reference files'

    return img_path, ref_path


def get_folder_struct(img_dir: str, job_name: str):
    stru = img_dir.split('/')
    s = stru.index(job_name)

    stru = stru[s:]
    assert len(
        stru) == 4, "don't found  job_name / layer_name / lot_name / pannel_id / "
    # job_name / layer_name / lot_name / pannel_id /
    return stru


def delta_time():
    return datetime.timedelta(hours=8)


def task(loader, net, n_iter, image,
         j_name, code_convert, description, user, job, result_path):
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
                with write_db(data_db, Session):
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


def get_job_names(path='./datasets/ketmac'):
    return os.listdir(relat2abs_path(path))


def relat2abs_path(path: str) -> str:
    current_path = '/workspace/yuan-algorithm/image-classification/'
    # current_path = os.getcwd()  # '/workspace/yuan-algorithm/image-classification'
    print('current_path', current_path)
    return path.replace('./', current_path)


def abs2relat_path(path: str) -> str:
    current_path = os.getcwd()  # '/workspace/yuan-algorithm/image-classification'
    return path.replace(current_path, '.')


def get_result_path(result_path):
    os.makedirs(result_path, exist_ok=True)
    result_img_path = os.path.join(result_path, 'img')
    result_ref_path = os.path.join(result_path, 'ref')
    os.makedirs(result_img_path, exist_ok=True)
    os.makedirs(result_ref_path, exist_ok=True)
    return result_img_path, result_ref_path


def main():
    print('--------------start predictor services--------------------')
    config = get_config(relat2abs_path('./config.json'))
    print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        [str(g) for g in config.get('gpus')])

    result_path = get_result_path(config.get('result_path'))
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


if __name__ == '__main__':
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
        conn = connectDb('t_detection')
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
