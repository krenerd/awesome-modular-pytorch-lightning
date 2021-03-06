from torchvision import models
from utils.models import drop_layers_after


def torchvision_feature_extractor(model_id, drop_after=None, *args, **kwargs):
    """
    Load model(and pretrained-weights) implemented in `torchvision.models`. Although some of our custom
    architecture implementation is also sort of based on torchvision, we implement this method to support more
    models and access to pretrained checkpoints.

    Model catalog can be found in: https://pytorch.org/vision/stable/models.html#id1
    TODO: support `torchvision` models for other tasks. e.g. video.

    Parameters
    ----------
    model_id: str
        Exact name of model to use. We look for `torchvision.models.{model_id}`.
    drop_after: str
        Remove all the layers including and after `drop_after` layer. For example, in the example below of `resnet50`,
        we want to set `drop_after` = "avgpool" to drop the final 2 layers and output feature map.
        ```
                ...
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
            )
            (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
            (fc): Linear(in_features=512, out_features=1000, bias=True)
        )
        ```
    Returns
    -------
    nn.Module
        feature_extractor network that can be used in multiple subtasks by plugging in different downstream heads.
    """
    # find model with same id & create model
    model = getattr(models, str(model_id))(*args, **kwargs)
    # detach final classification head(make it feature extractor)
    if drop_after:
        model = drop_layers_after(model, drop_after)
    return model
