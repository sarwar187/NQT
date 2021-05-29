# Multi-task Transformer

This is a pytorch implementation of the transformer model borrowed from the github repository of Samuel Lynn Evans. If you'd like to understand the model, or any of the code better, please refer to <a href=https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec>a tutorial by Samuel Lynn Evans</a>.

We used this model to train our neural query translation approach proposed in ACL '19 short paper titled as "A Multi-Task Architecture on Relevance-based Neural Query Translation" 

# Usage

Please refer to the jupyter notebook transformer_train.ipynb to check how we trained our model. The validation of the model is on cross-lingual IR task. Hence, validation requires retrieval with the validation set queries. You will need the queries, relevance judgements and search index. We have developed a .jar file that can take a query file, run it on the search index constructed from LATIMES corpus that we have discussed about in the paper, and finally evaluate the retrieval results using mean average precision. However, we have not included that in the repository. If you would really like to access the index, please contact me at smsarwar@cs.umass.edu. Otherwise please look at the transformer_train.ipynb jupyter notebook. It is sufficiently commented to understand the implementation of the model. 
