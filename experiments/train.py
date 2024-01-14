import argparse
import train_utils as train_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = train_utils.define_config()
    for key, value in config.items():
        if type(value) == bool:
            assert not value, "Default bool params should be set to false."
            parser.add_argument("--{}".format(key), action="store_true")
        else:
            parser.add_argument(
                "--{}".format(key),
                type=type(value) if value is not None else str,
                default=value,
            )
    config = parser.parse_args()

    from la_mbda.la_mbda import LAMBDA

    train_utils.train(config, LAMBDA)
