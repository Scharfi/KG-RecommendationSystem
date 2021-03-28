public class modelConstructor {
    String name;
    int dimension;
    String embeddingsPath;
    String datasetPath;
    String productsNamesPath;
    String normalizedEmbeddingsPath;

    public modelConstructor(String name) {
        this.name = name;
        switch (name) {
            case "pyke":
                this.dimension = 50;
                break;
            case "w2v":
                this.dimension = 300;
                break;
            case "r2v":
                this.dimension = 100;
                break;
            case "conex":
                this.dimension = 40;
                break;
            case "hybride":
                this.dimension = 340;
                break;
        }
        this.embeddingsPath = "data/embeddings/" + name + ".csv";
        this.normalizedEmbeddingsPath = "data/normalizedEmbeddings/" + name + "_normalizedvectors.csv";
        this.datasetPath = "data/trainingData/" + name + "/" + name;
        this.productsNamesPath="data/processingdata/product_names.csv";

        System.out.println("The embedding model information are:"
            +"\nModel name: "+ name
            +"\nDimension: "+dimension
            +"\nEmbedding path: "+embeddingsPath
            +"\nnormalized embedding path"+ normalizedEmbeddingsPath
            +"\nThe number of total products: 15089");
    }

    public String getModelname(){
        return this.name;
    }
    public String getnormalizedEmbeddingsPath(){
        return this.normalizedEmbeddingsPath;
    }
    public String getEmbeddingsPath() {
        return this.embeddingsPath;
    }
    public String getDatasetPath(){
        return this.datasetPath;
    }
    public int getDimension(){
        return this.dimension;
    }
    public String getProductNamesPath(){
        return this.productsNamesPath;
    }
}
