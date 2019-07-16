from audioNet.utils import read_csv, get_class_dist
from audioNet.utils import build_data
from audioNet.parameters import Config



if __name__ == "__main__":

    file_path = '.'
    config = Config(file_path=file_path, mode='conv')

    # Read csv file
    df = read_csv(config.file_path, 'instruments.csv')
    print(df.head())

    classes, class_dist = get_class_dist(df, config.clean_data_path)
    # normalize the class dist.
    prob_dist = class_dist / class_dist.sum()
    # cover the data set twice
    n_samples = 2 * int(df['length'].sum() / 0.1)

    config.classes = classes
    config.class_dist = class_dist
    config.prob_dist = prob_dist
    config.n_samples = n_samples

    print(f'\nclasses: \n{classes}')
    print(f'\nclass_dist: \n{class_dist}')
    print(f"\nn_samles: {n_samples}")
    print(f"\nprob_dist: \n{prob_dist}")


    # prepare data for the model as MFCC using clean data
    build_data(df, config)
