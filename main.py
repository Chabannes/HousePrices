from helper import *
import warnings
warnings.filterwarnings('ignore')


def main():

    data_train, data_test = import_data('train.csv', 'test.csv')
    data_train = removing_outliers(data_train)
    data_train_prep, data_test_prep, prepro_pred = preprocess_data(data_train, data_test)

    feature_list = list(data_train.columns)
    feature_list_test = list(data_train.drop('SalePrice', axis=1).columns)

    training_set, testing_set = train_test_datasets(data_train_prep, feature_list, feature_list_test)

    features = training_set[feature_list_test]
    labels = training_set["SalePrice"].values
    feature_test = testing_set[feature_list_test]

    model = build(dropout='True')
    compile_model(model, optimizer='adam')
    fit_plot_learning_history(model, features, labels, epochs=30, batch_size=16)
    plot_final_prediction(model, feature_test, testing_set, prepro_pred, feature_list)



if __name__ == '__main__':
    main()
