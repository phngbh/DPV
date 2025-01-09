import subprocess
import yaml

def run_script(script_name, config_path):
    command = ['python', script_name, '--config', config_path]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

if __name__ == "__main__":
    config_path = 'config.yml'
    
    run_script('generate_synthetic_data/LSTM_VAE.py', config_path)
    run_script('generate_synthetic_data/make_example_csv.py', config_path)