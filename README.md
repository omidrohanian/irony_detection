## Irony Detection in English Tweets 
Code and the data used with regard to experiments in the paper `WLV at SemEval-2018 Task 3:  Dissecting Tweets in Search of Irony`.

Dependencies:

* Ekphrasis (`pip install ekphrasis`)
* Stanford CoreNLP 
* Pycore NLP (`pip install pycorenlp`)
* Sklearn / scikit-learn (`pip install scikit-learn`)
* NLTK (`pip install nltk`)
	* nltk.download('wordnet')
	* nltk.download('averaged\_perceptron\_tagger')
	* nltk.download('sentiwordnet')
* Gensim (`pip install gensim`)

If you use the code for your project, please cite the following paper (<a href="http://www.aclweb.org/anthology/S18-1090">link</a> to PDF):

```
@inproceedings{rohanian2018wlv,
  title={WLV at SemEval-2018 Task 3: Dissecting Tweets in Search of Irony},
  author={Rohanian, Omid and Taslimipoor, Shiva and Evans, Richard and Mitkov, Ruslan},
  booktitle={Proceedings of The 12th International Workshop on Semantic Evaluation},
  pages={553--559},
  year={2018}
}
```

Run <b>Install_project.ipynb</b> for a guided process for the installation.