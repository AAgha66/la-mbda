import clearml
<<<<<<< HEAD

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

=======
>>>>>>> 31d93dc5ab5a225b88294a4bad91cbf462213610
import train_utils as train_utils
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
<<<<<<< HEAD
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
=======
>>>>>>> 31d93dc5ab5a225b88294a4bad91cbf462213610

if __name__ == "__main__":
    config = train_utils.make_config(train_utils.define_config())
    if not config.local:
        task = clearml.Task.init()
        task_logger = task.get_logger()
        task_params = task.get_parameters_as_dict(cast=True)
        print(task_params)
        d = task_params["internal"]
        print(d)
        config.log_dir = d["log_dir"]
        config.environment = d["environment"]
        config.total_training_steps = d["total_training_steps"]
        config.safety = d["safety"]

    from la_mbda.la_mbda import LAMBDA

    train_utils.train(config, LAMBDA)
