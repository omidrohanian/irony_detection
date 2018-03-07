In order to run the program: 

1. run `feature_generator_TaskA.ipynb` to generate all the features and save it to `train_feats_taskA.npy`
2. run `RFECV_feature_ranking.ipynb` to do feature selection using RFECV and save the indices of the top features in `indices`
3. finally run `train_test` to train and test the model based on the top features from `indices`

When running the last script, you should have the [Twitter Word2vec model](http://yuca.test.iminds.be:8900/fgodin/downloads/word2vec_twitter_model.tar.gz) and pre-trained [emoji2vec embeddings](https://github.com/uclmr/emoji2vec/blob/master/pre-trained/emoji2vec.bin) and set their path in the script. 