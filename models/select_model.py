"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']  # one input: L
    # 通过配置获取模型
    if model == 'plain':
        from models.model_plain import ModelPlain as M

    elif model == 'plain2':  # two inputs: L, C
        from models.model_plain2 import ModelPlain2 as M

    elif model == 'plain4':  # four inputs: L, k, sf, sigma
        from models.model_plain4 import ModelPlain4 as M

    elif model == 'gan':  # one input: L
        from models.model_gan import ModelGAN as M

    elif model == 'vrt':  # one video input L, for VRT
        from models.model_vrt import ModelVRT as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))
    # 初始化模型 通过opt设置模型
    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
