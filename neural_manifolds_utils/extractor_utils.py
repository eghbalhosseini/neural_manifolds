from mftma.utils.activation_extractor import extractor
class mftma_extractor(object):
    def __init__(self,model=None, exm_per_class=50, nclass=50, data=None):
        self.extractor=extractor