## EmbeddingBasedRecommendationSysem

#Introduction
This repository contain the code of a recommendation system that uses an artificial neural network and a learning-to-rank approach to generate the recommendations. The recommendation system produces a Top-N recommended products for a given product.
The recommendation system can be trained and tested on various embedding models that were produced from on a Knowledge Graph. 

The embedding model in this repository are generated using different embedding algorithm:
1.  Word2Vec
2.  RDF2Vec
3.  Pyke
4.  ConEx 
5.  HybridE (Combination of Word2Vec and ConEx)


To train and test the model, the data has to be generated by the class dataprocessor where you can specify the name of the embedding model
of choice. To build a centroid-based data, change centroid to "True" (not for HybridE).

```
public static void main(String[] args) throws Exception {

        modelinfo = new modelConstructor("conex"); // w2v(Word2Vec), r2v(RDF2Vec), pyke, conex, hybride

```

The training of the recommender is done in Class trainer for Word2Vec, RDF2Vec, Pyke, ConEx and in trainerHybridE for HybridE. (This two classes will be merged). The model can be saved at any training phase (epoch), where its specified in this line:
```
//save model
    if (savemodel == true) {
            String Path = "data/evaluation/" + modeler + "/" + modeler + "_epoch" + epochs + ".model";
            model.save(new File(Path), true);
        }
```
The Class loader can load a model and further train and test it. 
The Class Cheater generate centroid vectors for each product found in the test set and evaluate them.

The evaluation results are stored under data/evaluation/model_name.

# Requirement

Java 1.8
Maven 3.6.1



