import os
import glob
import yaml

def read_yaml_file():
    path_to_yaml = "config.yaml"
    try:
        with open(path_to_yaml, "r") as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        print("Error reading the config file")


def validate_raw_data_file_exist(config):
        try:
            validation_status = None

            file_names = sorted(glob.glob(os.path.join("data","raw","*.parquet")))
            for file in file_names:
                prefix_file_name = file.split('/')[-1].split('_')[0]
                if prefix_file_name == 'complaints':
                    validation_status = True
                    with open(config['data_validation']['STATUS_FILE'], 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = False
                    with open(config['data_validation']['STATUS_FILE'], 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e
        

if __name__ == "__main__":
    config = read_yaml_file()
    print(config['data_validation']['STATUS_FILE'])
    validate_raw_data_file_exist(config)

