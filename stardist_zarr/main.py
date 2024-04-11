import argparse
from pathlib import Path
import yaml
import json
from stardist_zarr.pipeline import stardist2D_stacked


def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist"

    if args.config.suffix == '.yaml':
        with open(args.config, 'r') as f:
            return yaml.safe_load(f)

    elif args.config.suffix == '.json':
        with open(args.config, 'r') as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported config file format {args.config.suffix}")


def main():
    config = get_config()
    assert 'input_files' in config, "Key 'input_files' missing in config"
    assert 'pipeline' in config, "Key 'pipeline' missing in config"
    for file_infos in config['input_files']:
        stardist2D_stacked(file_infos, **config['pipeline'])


if __name__ == '__main__':
    main()