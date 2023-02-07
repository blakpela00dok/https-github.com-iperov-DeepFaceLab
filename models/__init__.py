from .ModelBase import ModelBase

def import_model(model_class_name, use_bn=False):
    module = __import__('Model_'+model_class_name, globals(), locals(), [], 1)
    if use_bn:
        return getattr(module, 'Model_bn')
    return getattr(module, 'Model')
