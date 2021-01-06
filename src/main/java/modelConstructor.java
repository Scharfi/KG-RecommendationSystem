public class modelConstructor {
    String name;
    int dimension;
    String embeddingsPath;
    String datasetPath;
    String normalizedEmbeddingsPath;

    public modelConstructor(String name) {
        this.name = name;
        if (name == "pyke")
            this.dimension = 50;
        else if (name == "w2v")
            this.dimension = 300;
        else if (name == "r2v")
            this.dimension = 100;
        else if (name == "conex")
            this.dimension = 40;
        else if (name =="hybride")
            this.dimension = 340;

        this.embeddingsPath = "data/embeddings/" + name + ".csv";
        this.normalizedEmbeddingsPath = "data/normalizedEmbeddings/" + name + "_normalizedvectors.csv";
        this.datasetPath = "data/trainingData/" + name + "/" + name;
    }

    public String getModelname(){
        return this.name;
    }
    public String getnormalizedEmbeddingsPath(){
        return this.normalizedEmbeddingsPath;
    }
    public String getEmbeddingsPath(){ return this.embeddingsPath; }
    public String getDatasetPath(){
        return this.datasetPath;
    }
    public int getDimension(){
        return this.dimension;
    }



}
