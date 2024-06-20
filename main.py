import argparse
from pathlib import Path
from train.training_model import main, CONFIG_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training model.")
    parser.add_argument('-t', '--training_config_file', type=Path,
                        default=CONFIG_PATH.joinpath("train_config.json"),
                        help="Path to the training configuration file.")
    parser.add_argument('-d', '--data_config_file', type=Path, 
                        default=CONFIG_PATH.joinpath("data_config.json"),
                        help="Path to the data configuration file.")

    args = parser.parse_args()

    train_manager_obj = main(training_config_file=args.training_config_file,
                             data_config_file=args.data_config_file)