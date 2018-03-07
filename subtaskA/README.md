In order to run the program: 

I) run `feature_generator_TaskA.ipynb` to generate all the features and save it to `train_feats_taskA.npy`
II) run `RFECV_feature_ranking.ipynb` to do feature selection using RFECV and save the indices of the top features in `indices`
III) finally run `train_test` to train and test the model based on the top features from `indices`