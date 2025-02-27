import subprocess
import yaml

# def run_script(script_name, args):
#     command = ['python3', script_name] + args
#     result = subprocess.run(command, capture_output=True, text=True)
#     print(result.stdout)
#     if result.stderr:
#         print(result.stderr)

def run_script(script_name, config_path):
    command = ['python3', script_name, '--config', config_path]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

# def parse_args(config, section):
#     args = []
#     for key, value in config[section].items():
#         args.append(f'--{key}')
#         args.append(str(value))
#     return args

if __name__ == "__main__":
    # with open('./create_input/config.yml', 'r') as f:
    #     config = yaml.safe_load(f)

    # preprocess_args = parse_args(config, 'preprocess')
    # make_zero_padding_args = parse_args(config, 'make_zero_padding')
    # make_input_args = parse_args(config, 'make_input')

    config_path = './create_input/config.yml'

    #run_script('create_input/preprocess.py', preprocess_args)
    #run_script('create_input/make_zero_padding.py', make_zero_padding_args)
    run_script('./create_input/make_input.py', config_path)