from mmseg.models import UDA
from mmseg.models.uda.uda_decorator import UDADecorator


@UDA.register_module()
class TestUDA(UDADecorator):

    def __init__(self, **cfg):
        super(TestUDA, self).__init__(**cfg)
