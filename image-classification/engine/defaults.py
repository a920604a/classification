'''
Author: yuan
Date: 2021-02-24 16:16:25
LastEditTime: 2021-03-02 09:06:25
FilePath: /aidc-algorithm/image-classification/engine/defaults.py
'''
from utils.network import get_network
import copy
import torch
import torchvision.transforms as transforms
from collections import namedtuple


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """
    net_cfg = namedtuple('net_cfg', ['net', 'gpu'])

    def __init__(self, cfg, gid=0):
        # self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg = copy.deepcopy(cfg)
        self.gid = gid
        netcfg = DefaultPredictor.net_cfg(net=self.cfg.NET, gpu=self.cfg.GPU)
        self.model = get_network(netcfg)  # build_model(self.cfg)
        self.model.load_state_dict(torch.load(self.cfg.MODEL_FILE))
        self.model = self.model.to(torch.device('cuda:{}'.format(self.gid)))
        self.model.eval()
        # if len(cfg.DATASETS.TEST):
        #     self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load(cfg.MODEL.WEIGHTS)

        # self.aug = T.ResizeShortestEdge(
        #     [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        # )
        # self.aug = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize(self.cfg.IMAGE_SIZE),
        #     transforms.ToTensor(),
        # ])

        self.input_format = "BGR"  # cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                # original_image = original_image[:, :, ::-1]
                original_image = original_image[:, :, :, ::-1]  # N, H, W, C
            # height, width = original_image.shape[1:3]
            # image = self.aug.get_transform(
            #     original_image).apply_image(original_image)
            # image = self.aug(original_image)
            image = original_image
            # image = image.transpose(2, 0, 1)
            # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            # inputs = {"image": image, "height": height, "width": width}

            # image = image.unsqueeze(0)  # add axis to replace batch
            predictions = self.model(image.cuda().to(
                torch.device('cuda:{}'.format(self.gid))))
            return predictions
