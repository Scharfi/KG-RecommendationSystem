import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
        this.productsNamesPath = "data/processingdata/product_names.csv";

        System.out.println("The embedding model information are:"
                + "\nModel name: " + name
                + "\nDimension: " + dimension
                + "\nEmbedding path: " + embeddingsPath
                + "\nnormalized embedding path" + normalizedEmbeddingsPath
                + "\nThe number of total products: 15089");


    }

    public String getModelname() {
        return this.name;
    }

    public String getnormalizedEmbeddingsPath() {
        return this.normalizedEmbeddingsPath;
    }

    public String getEmbeddingsPath() {
        return this.embeddingsPath;
    }

    public String getDatasetPath() {
        return this.datasetPath;
    }

    public int getDimension() {
        return this.dimension;
    }

    public String getProductNamesPath() {
        return this.productsNamesPath;
    }

    public Map<Integer, String> getRecommendationList() {
        Map<Integer, String> list = new HashMap<>();
        String id;
        try {
            List<String> lines = IOUtils.readLines(new FileInputStream("data/processingdata/Recommended_10ids.csv"), StandardCharsets.UTF_8);
            for (String line : lines) {
                id = line.substring(0, line.indexOf(","));
                list.put(Integer.parseInt(id), line.substring(line.indexOf(",") + 1));
            }
            return list;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public INDArray list_embeddings() {
        try {
            INDArray embeds = Nd4j.create(new int[]{15089, dimension});
            List<String> lines = IOUtils.readLines(new FileInputStream(normalizedEmbeddingsPath), StandardCharsets.UTF_8);
            for (String line : lines) {
                int id = Integer.parseInt(line.substring(0, line.indexOf(",")));
                String[] parts = line.split(",");
                for (int j = 0; j < parts.length - 1; j++) {
                    double v = Double.parseDouble(parts[j + 1]);
                    embeds.putScalar(new int[]{id, j}, v);
                }
            }
            return embeds;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public INDArray list_embeddingsByName(String embeddingModelName) {
        int dimension = 0;
        try {
            switch (embeddingModelName) {
                case "pyke":
                    dimension = 50;
                    break;
                case "w2v":
                    dimension = 300;
                    break;
                case "r2v":
                    dimension = 100;
                    break;
                case "conex":
                    dimension = 40;
                    break;
            }
            String path = "data/normalizedEmbeddings/" + embeddingModelName + "_normalizedvectors.csv";
            INDArray embeds = Nd4j.create(new int[]{15089, dimension});
            List<String> lines = IOUtils.readLines(new FileInputStream(path), StandardCharsets.UTF_8);
            for (String line : lines) {
                int id = Integer.parseInt(line.substring(0, line.indexOf(",")));
                String[] parts = line.split(",");
                for (int j = 0; j < parts.length - 1; j++) {
                    double v = Double.parseDouble(parts[j + 1]);
                    embeds.putScalar(new int[]{id, j}, v);
                }
            }
            return embeds;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

}
