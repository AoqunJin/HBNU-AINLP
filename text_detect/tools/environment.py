from .detection_models import basic_detection, dl_detection
from .functions import get_obs_image, get_folder_image


class Environment():
    def __init__(self, cfg) -> None:
        if cfg.use_obs:
            self.data_collector = get_obs_image(cfg)
        else:
            assert cfg.image_folder is not None
            self.data_collector = get_folder_image(cfg)
            
        if cfg.use_dl:
            self.detection_model = dl_detection(cfg)
        else:
            self.detection_model = basic_detection()
        
        self.cfg = cfg
    
    def __call__(self):
        if self.cfg.use_obs:
            data_list = self.data_collector()
        else:
            data_list = self.data_collector(self.cfg.image_folder)
        
        return [self.detection_model(data) for data in data_list]
    