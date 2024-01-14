import os
import clearml

import train_utils as train_utils

if __name__ == "__main__":
    os.environ["LD_LIBRARY_PATH"] = (
        os.environ["LD_LIBRARY_PATH"] + ":/home/.mujoco/mujoco200/bin"
    )

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
