import os
import pandas


def csv_maker(data_source_dir, destination_csv_dir):
    if os.path.isfile(destination_csv_dir):
        print('csv exists. overwrite!')
    #     os.remove(destination_csv_dir)

    file_names_list = os.listdir(data_source_dir)
    file_names_list = [data_source_dir + r'/' + filename for filename in file_names_list]

    data_frame = pandas.DataFrame({'file_name': file_names_list})
    data_frame.to_csv(destination_csv_dir, index=False)


if __name__ == "__main__":
    csv_maker('D:/Jupyter/AgiProbot/model_trainer/data/date_set/Dataset3_merge',
              'D:/Jupyter/AgiProbot/model_trainer/data/dataset.csv')
