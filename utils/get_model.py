def GetModel(modelname):
    module = __import__('models.{0}'.format(modelname.lower()))
    Model = eval('module.{0}.{1}'.format(modelname.lower(), modelname.upper()))
    return Model