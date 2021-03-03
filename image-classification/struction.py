from collections import namedtuple
import torch
from utils.loader import get_valid_dataloader, get_test_dataloader
from utils.network import get_network
db_data = namedtuple('db_data', ['model_name',
                                 'site_name',
                                 'product_name',
                                 'job_name',
                                 'layer_name',
                                 'lot_name',
                                 'panel_id',
                                 'serial_number',
                                 'process_time',
                                 'image_name',
                                 'source_path',
                                 'reference_path',
                                 'detection_path',
                                 'detection_class',
                                 'true_label',
                                 'description',
                                 'confidence',
                                 'create_by',
                                 #  'create_at',
                                 'update_by'
                                 ])


# class Model:
#     def __init__(self, model):
#         self.image_size = tuple(model['image_size'])
#         self.model_file = model['model_file']
#         self.labels = model['labels']
#         self.focus_labels = model['focus_labels']
#         self.net = model['name']

#         self.gpu = torch.cuda.is_available()

# class Job:
#     def __init__(self, job):

#         self.job_name = job['job_name']
#         self.product_name = job['product_name']
#         self.site_name = job['site_name']
#         # self.model = Model(job['model'])
#         body = {
#             'image_size': tuple(job['model']['image_size']),
#             'model_file': job['model']['model_file'],
#             'labels': job['model']['labels'],
#             'focus_labels': job['model']['focus_labels'],
#             'net': job['model']['name'],
#             'gpu': torch.cuda.is_available()
#         }

#         self.model = type('Model', (object,), body)


class Cfg():
    def __init__(self, config):

        self.gpus = config.get('gpu')
        self.thread_pool = config.get('thread_pool')
        self.process_pool = config.get('process_pool')
        self.batch_size = config.get('batch_size')
        self.result_path = config.get('result_path')
        self.mount_folder_path = config.get('mount_folder_path')
        self.input_folder_path = config.get('input_folder_path')
        self.create_by = config.get('create_by')
        self.description = config.get('description')


class Child_Cfg(Cfg):
    model = namedtuple('model',
                       ['image_size', 'model_file',
                        'labels', 'focus_labels', 'net', 'gpu'
                        ])

    job = namedtuple('job',
                     ['job_name', 'site_name',
                      'product_name', 'model'
                      ])

    def __init__(self, config):
        super(Child_Cfg, self).__init__(config)
        self.jobs = dict()
        for j in config['jobs']:
            self.jobs[j['job_name']] = Child_Cfg.job(
                job_name=j['job_name'],
                site_name=j['site_name'],
                product_name=j['product_name'],
                model=Child_Cfg.model(
                    image_size=j['model']['image_size'],
                    model_file=j['model']['model_file'],
                    labels=j['model']['labels'],
                    focus_labels=j['model']['focus_labels'],
                    net=j['model']['name'],
                    gpu=torch.cuda.is_available()
                )

            )
        # [print(j['job_name']) for j in config['jobs']]
        # self.jobs = [Job(j) for j in config['jobs']]


class Config(Cfg):
    model = namedtuple('model',
                       ['image_size', 'model_file',
                        'labels', 'focus_labels', 'net', 'gpu'
                        ])

    def __init__(self, config):
        super(Config, self).__init__(config)

        self.jobs = config.get('jobs')

        self.job_names = [j.get('job_name') for j in config.get('jobs')]

        self.default = [j for j in self.jobs if j.get('job_name') == 'default']

    def get_job_config(self, job_name):
        for j in self.jobs:
            if j['job_name'] == job_name:
                return j

        else:
            return self.default[0]

    def get_model_config(self, job_name):

        ret = self.get_job_config(job_name)

        m = ret.get('model')

        model = Child_Cfg.model(
            image_size=m.get('image_size'),
            model_file=m.get('model_file'),
            labels=m.get('labels'),
            focus_labels=m.get('focus_labels'),
            net=m.get('name'),
            gpu=torch.cuda.is_available()
        )

        return model, ret


class NN_model():
    """
    one img_path mapping to one data_loader
    """

    def __init__(self, model, batch_size, img_path, works):
        self.model = model
        self.net = get_network(self.model)
        self.test_loader = get_valid_dataloader(
            root_path=img_path,
            num_workers=works,   # multi-threading
            batch_size=batch_size,
            shuffle=False,
            size=tuple(model.image_size)
        )
        # class_to_idx = {l: i for i, l in enumerate(self.model.labels)}
        # classes = self.model.labels,
        # self.test_loader = get_test_dataloader(
        #     root_path=img_path,
        #     num_workers=works,   # multi-threading
        #     batch_size=batch_size,
        #     shuffle=False,
        #     size=tuple(model.image_size),
        #     classes=classes,
        #     class_to_idx=class_to_idx
        # )

    def get_net(self):
        self.net.load_state_dict(torch.load(self.model.model_file))
        return self.net

    def get_loader(self):
        return self.test_loader
