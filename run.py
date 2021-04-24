import yaml
import argparse
from src.trainer import Training
from src.colnet import ColNet


def load_config(config_file, model_checkpoint=None):
    learning_rate = 0.0001
    num_workers = 4
    models_dir = './model/'
    img_out_dir = './out/'

    with open(config_file, 'r') as conf:
        y = yaml.load(conf)
        
        if 'learning_rate' in y:
            learning_rate = y['learning_rate']

        if 'model_checkpoint' in y:
            model_checkpoint = y['model_checkpoint']

        if 'num_workers' in y:
            num_workers = y['num_workers']

        if 'models_dir' in y:
            models_dir = y['models_dir']
        
        if 'img_out_dir' in y:
            img_out_dir = y['img_out_dir']


        train = Training(batch_size=y['batch_size'],
                         epochs=y['epochs'],
                         img_dir_train=y['img_dir_train'],
                         img_dir_val=y['img_dir_val'],
                         img_dir_test=y['img_dir_test'],
                         learning_rate=learning_rate,
                         model_checkpoint=model_checkpoint,
                         num_workers=num_workers,
                         models_dir=models_dir,
                         img_out_dir=img_out_dir)

        return train



if __name__ == "__main__":
    short_desc = 'Loads network configuration from YAML file.\n'
    parser = argparse.ArgumentParser(description=short_desc,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', metavar='config', help='Path to .yaml config file')
    parser.add_argument('--model', help='Path to pretrained .pt model')
    args = parser.parse_args()
    t = load_config(args.config, args.model)
    t.run()
    t.plot_losses()
    
