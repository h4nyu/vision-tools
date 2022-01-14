import sys
from omegaconf import OmegaConf
from vision_tools.yolox import YOLOX, Criterion, Inference
sys.path.append('/kaggle/input/tensorflow-great-barrier-reef')
sys.path.append('/kaggle/input/tensorflow-great-barrier-reef/greatbarrierreef/')
import greatbarrierreef
# import competition
# env = greatbarrierreef.make_env()

# help('modules')


# def predict():
#     cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config/yolox.yaml"))
#     return model.predict(data)

# if __name__ == '__main__':
#     predict()
