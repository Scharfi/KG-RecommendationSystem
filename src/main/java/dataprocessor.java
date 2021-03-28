import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class dataprocessor {
    static String modeler;
    static String embeddingPath;
    static modelConstructor embeddingModelinfo;

    private static Map<String, String> products;
    private static Map<String, String> embeddings;
    private static Map<String, String> recommends;

    public static void main(String[] args) throws Exception {
        embeddingModelinfo = new modelConstructor("conex"); // w2v(Word2Vec), r2v(RDF2Vec), pyke, conex, hybride
        modeler = embeddingModelinfo.getModelname();
        boolean centroid = false;

        System.out.println(" Data preparation of " + modeler);
        // Data preparation
        /*
        generateIds(embeddingModelinfo.getProductNamesPath()); //create new ids for all products
        products = getproductsInfo(); //return products {id,name}
        normalizeVectors(); //normalize data
        */

        // take normalized embeddings and generated training and test data
        embeddings = readEmbeddings();
        if ("hybride".equals(modeler))
            createTrainAndTestDataHybrid(embeddings); // Create Training and testing for HybridE
        else if (centroid)
            createCentroidTrainAndTestData(embeddings); // Create Training and testing for the centroid assumption
        else
            createTrainAndTestData(embeddings); //Create Training and testing for w2v(Word2Vec),r2v(RDF2Vec),pyke,conex
    }

    private static void normalizeVectors() {
        int count = 0;
        double magnitude, value;
        try {
            FileWriter file = new FileWriter("data/normalizedEmbeddings/" + modeler + "_normalizedvectors.csv");
            List<String> lines = IOUtils.readLines(new FileInputStream(embeddingPath), StandardCharsets.UTF_8);

            for (String line : lines) {
                String[] parts = line.split(",");
                double sum = 0;
                // Compute magnitude of vector
                for (int i = 1; i < parts.length; i++) {
                    sum = sum + Math.pow(Double.parseDouble(parts[i]), 2);
                }
                magnitude = Math.sqrt(sum);
                //normalize the vector
                if (magnitude != 0) {
                    String normalizedline = parts[0];
                    for (int i = 1; i < parts.length; i++) {
                        value = Double.parseDouble(parts[i]) / magnitude;
                        String v = String.valueOf(value);
                        if (v.contains("E")) { //check for scientific numbers
                            BigDecimal d = new BigDecimal(value);
                            normalizedline = normalizedline + "," + d.toPlainString().substring(0, 20);
                        } else if (!v.isEmpty() && v.contains("."))
                            normalizedline = normalizedline + "," + value;
                    }
                    // write the normalized embedding vector to a file
                    if (!normalizedline.contains("E") && !normalizedline.contains(",,")) {
                        file.append(normalizedline);
                        file.append("\n");
                    }
                } else
                    count++;
            }
            System.out.println("magnitude 0 for " + count + " products");

            //save file
            try {
                file.flush();
                file.close();
            } catch (IOException e) {
                System.out.println("Error while flushing/closing fileWriter !!!");
                e.printStackTrace();
            }
        } catch (Exception e) {
            e.printStackTrace();

        }
    }

    public static void createTrainAndTestData( Map<String, String> embeddings) {
        try {
            int productId;
            String id, recId, fileline, embed, recEmbed;
            String filepath = "data/trainingData/" + modeler + "/" + modeler;
            Map<Integer, String> recommendationlist = getRecommendationList();

            for (int fold = 0; fold < 1; fold++) {
                System.out.println("creating train and test data for " + modeler + " - Fold " + fold
                        +"\nThe number of total products: "+ recommendationlist.size());

                int testProducts = 0, trainProducts = 0, excludedProducts = 0;
                FileWriter trainfile = new FileWriter(filepath + "_train_F" + fold + ".csv");
                FileWriter testfile = new FileWriter(filepath + "_test_F" + fold + ".csv");

                for (int i=0;i<15090;i++) {
                    if (recommendationlist.containsKey(i)) {
                        String[] parts = recommendationlist.get(i).split(",");
                        id = String.valueOf(i);
                        embed = FindEmbeddingsById(embeddings, id);
                        if (!(id.isEmpty()) && !(embed.isEmpty())) {
                            productId = i;
                            //construct test data: {testproductId,embed}
                            if (productId % 5 == fold) {
                                String testline = productId + "," + embed + "\n";
                                testfile.append(testline);
                                testProducts++;
                            } else {
                                //construct train data: {train_embed,true_embed}
                                for (int j = 0; j < parts.length; j++) {
                                    recId = parts[j];
                                    recEmbed = FindEmbeddingsById(embeddings, recId);
                                    if (!recEmbed.isEmpty()) {
                                        fileline = embed + "," + recEmbed;
                                        trainfile.append(fileline).append("\n");
                                    } else {
                                        excludedProducts++;
                                    }
                                }
                                trainProducts++;
                            }
                        } else {
                            excludedProducts++;
                        }
                    } else
                        excludedProducts++;
                }

                System.out.println("Fold "+fold +
                        "\nThe number of testing products: "+ testProducts +
                        "\nThe number of training products: "+ trainProducts +
                        "\nThe number excludedProducts: " + excludedProducts);

                //save file
                try {
                    trainfile.flush();
                    trainfile.close();
                    testfile.flush();
                    testfile.close();
                } catch (IOException e) {
                    System.out.println("Error while flushing/closing fileWriter !!!");
                    e.printStackTrace();
                }
            }
        } catch (FileNotFoundException fileNotFoundException) {
            fileNotFoundException.printStackTrace();
        } catch (IOException ioException) {
            ioException.printStackTrace();
        }
    }


    public static void createTrainAndTestDataHybrid(Map<String, String> embeddings) {
        try {
            int productid;
            String id, recId, fileline, embed, recEmbed, embed2, recEmbed2;
            String filepath = "data/trainingData/" + modeler + "/" + modeler;
            Map<String, String> embeddingsConex = readEmbeddingsC();
            Map<Integer, String> recommendationlist = getRecommendationList();

            for (int fold = 0; fold < 5; fold++) {
                System.out.println(" creating train and test data for " + modeler + "Fold " + fold);
                int testProducts = 0, trainProducts = 0, excludedProducts = 0;

                FileWriter trainfile = new FileWriter(filepath + "_train_F" + fold + ".csv");
                FileWriter testfile = new FileWriter(filepath + "_test_F" + fold + ".csv");

                for (int i=0;i<recommendationlist.size();i++) {
                    if (recommendationlist.containsKey(i)) {
                        String[] parts = recommendationlist.get(i).split(",");
                        id = String.valueOf(i);
                    embed = FindEmbeddingsById(embeddings, id);
                    embed2 = FindEmbeddingsById(embeddingsConex,id);

                    if (!(id.isEmpty()) && !(embed.isEmpty())&& !(embed2.isEmpty())) {
                        productid = Integer.parseInt(id);
                        //construct test data
                        if (productid % 5 == fold) {
                            String testline = productid +","+embed2 + "," + embed + "\n";
                            testfile.append(testline);
                            testProducts++;
                        } else {
                            //construct train data
                            for (int j = 0; j < parts.length; j++) {
                                recId = parts[j];
                                recEmbed = FindEmbeddingsById(embeddings, recId);
                                recEmbed2 = FindEmbeddingsById(embeddingsConex, recId);
                                if (!recEmbed.isEmpty()&&!recEmbed2.isEmpty()) {
                                    fileline = embed2 + "," + embed +","+ recEmbed2+","+recEmbed;
                                    trainfile.append(fileline);
                                    trainfile.append("\n");
                                } else {
                                    excludedProducts++;
                                }
                            }
                            trainProducts++;
                        }
                    } else {
                        excludedProducts++;
                    }
                }
                }
                System.out.println("Fold "+fold +
                        "\nThe number of testing products: "+ testProducts +
                        "\nThe number of training products: "+ trainProducts +
                        "\nThe number excludedProducts: " + excludedProducts);

                //save files
                try {
                    trainfile.flush();
                    trainfile.close();
                    testfile.flush();
                    testfile.close();
                } catch (IOException e) {
                    System.out.println("Error while flushing/closing fileWriter !!!");
                    e.printStackTrace();
                }
            }
        } catch (FileNotFoundException fileNotFoundException) {
            fileNotFoundException.printStackTrace();
        } catch (IOException ioException) {
            ioException.printStackTrace();
        }
    }

    public static void createCentroidTrainAndTestData(Map<String, String> embeddings) {
        int productid;
        String id, recId, fileline, embed;
        try {
            INDArray embeds = list_embeddings();
            Map<Integer, String> recommendationlist = getRecommendationList();
            String filepath = "data/trainingData/" + modeler + "/" + modeler;

            for (int fold = 0; fold < 1; fold++) {
                System.out.println("Creating train and test data for " + modeler + " - Fold " + fold);
                int testProducts = 0, trainProducts = 0, excludedProducts = 0;

                FileWriter trainfile = new FileWriter(filepath + "_train_F" + fold + "Centroid.csv");
                FileWriter testfile = new FileWriter(filepath + "_test_F" + fold + "Centroid.csv");

                for (int i=0;i<recommendationlist.size();i++) {
                    if (recommendationlist.containsKey(i)) {
                        String[] parts = recommendationlist.get(i).split(",");
                        id = String.valueOf(i);
                        embed = FindEmbeddingsById(embeddings, id);
                        if (!(id.isEmpty()) && !(embed.isEmpty())) {
                            productid = Integer.parseInt(id);
                            //construct test data
                            if (productid % 5 == fold) {
                                String testline = productid + "," + embed + "\n";
                                testfile.append(testline);
                                testProducts++;
                            } else {
                                //construct train data
                                INDArray row = Nd4j.zeros(new long[]{embeddingModelinfo.getDimension()});
                                for (int j = 0; j < parts.length; j++) {
                                    recId = parts[j];
                                    row = row.addRowVector(embeds.getRow(Integer.parseInt(recId)));
                                }
                                row.divi(10);
                                if (!row.isEmpty()) {
                                    fileline = embed + "," + row.toStringFull()+"\n";
                                    trainfile.append(fileline);
                                } else {
                                    System.out.println("recEmbed is empty!");
                                    excludedProducts++;
                                }
                                trainProducts++;
                            }
                        } else {
                            excludedProducts++;
                        }
                    }
                }

                System.out.println("Fold "+fold +
                        "\nThe number of testing products: "+ testProducts +
                        "\nThe number of training products: "+ trainProducts +
                        "\nThe number excludedProducts: " + excludedProducts);

                //save files
                try {
                    trainfile.flush();
                    trainfile.close();
                    testfile.flush();
                    testfile.close();
                } catch (IOException e) {
                    System.out.println("Error while flushing/closing fileWriter !!!");
                    e.printStackTrace();
                }
            }
        } catch (FileNotFoundException fileNotFoundException) {
            fileNotFoundException.printStackTrace();
        } catch (IOException ioException) {
            ioException.printStackTrace();
        }
    }

    public static INDArray list_embeddings() {
        try{
            INDArray embeds = Nd4j.create(new int[]{15089, embeddingModelinfo.getDimension()});
            List<String> lines = IOUtils.readLines(new FileInputStream(embeddingModelinfo.getnormalizedEmbeddingsPath()), StandardCharsets.UTF_8);
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


    private static Map<String, String> readEmbeddings() {
        String path="";
        try {
            if ("hybride".equals(modeler))
                path = "data/normalizedEmbeddings/w2v_normalizedvectors.csv";
            else
                path = embeddingModelinfo.getnormalizedEmbeddingsPath();
            List<String> lines = IOUtils.readLines(new FileInputStream(path), StandardCharsets.UTF_8);
            Map<String, String> enums = new HashMap<>();
            for (String line : lines) {
                String id = line.substring(0, line.indexOf(","));
                String prodDisc = line.substring(line.indexOf(",") + 1);
                enums.put(id, prodDisc);
            }
            return enums;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static Map<String, String> readEmbeddingsC() {
        try {
            String path = "data/normalizedEmbeddings/conex_normalizedvectors.csv";
            List<String> lines = IOUtils.readLines(new FileInputStream(path), StandardCharsets.UTF_8);
            Map<String, String> enums = new HashMap<>();
            for (String line : lines) {
                String id = line.substring(0,line.indexOf(","));
                String prodDisc = line.substring(line.indexOf(",")+1);
                enums.put(id, prodDisc);
            }
            return enums;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // generate new ids to all the products
    private static void generateIds(String csvFileClasspath) {
        try {
            String fileline, productName;
            int i = 0;
            FileWriter file = new FileWriter("data/processingdata/products_newids.csv");
            List<String> lines = IOUtils.readLines(new FileInputStream(csvFileClasspath), StandardCharsets.UTF_8);

            for (String line : lines) {
                String[] parts = line.split(",");
                productName = parts[0].substring(0, parts[0].length() - 3);
                fileline = i + "," + productName;
                file.append(fileline).append("\n");
                i++;
            }

            //save file
            try {
                file.flush();
                file.close();
            } catch (IOException e) {
                System.out.println("Error while flushing/closing fileWriter !!!");
                e.printStackTrace();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Update the embedding file by changing the name of the product with its id
    public static void UpdateEmbeddings(String csvFileClasspath, Map<String, String> products) {

        try {
            FileWriter file = new FileWriter("data/processingdata/embeddingsId_updated.csv");
            List<String> lines = IOUtils.readLines(new FileInputStream(csvFileClasspath), StandardCharsets.UTF_8);
            String p_name, p_id, embedline;
            String fileline;

            for (String line : lines) {
                String[] parts = line.split(",");
                p_name = parts[0];
                p_id = FindProductId(products, p_name);
                if (!(p_id.isEmpty())) {
                    embedline = line.substring(line.indexOf(","));
                    fileline = p_id + embedline;
                    file.append(fileline);
                    file.append("\n");
                } else {
                    System.out.println("Embedding vector for product "+ p_name+" not found");
                }
            }
            try {
                file.flush();
                file.close();
            } catch (IOException e) {
                System.out.println("Error while flushing/closing fileWriter !!!");
                e.printStackTrace();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Map<Integer, String> getRecommendationList() {
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

    private static String FindProductId(Map<String, String> products, String name) {
        String id = null;
        String productName;
        try {
            for (Map.Entry<String, String> entry : products.entrySet()) {
                productName = entry.getValue();
                if (productName.equals(name)) {
                    id = entry.getKey();
                    break;
                }
            }
            return id;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static String FindEmbeddingsById(Map<String, String> embeddings, String productId) {
        String embeds=null;
        String localId;
        try {
            for (Map.Entry<String, String> entry : embeddings.entrySet()) {
                localId = entry.getKey();
                if (localId.equals(productId)) {
                    embeds = entry.getValue();
                    break;
                }
            }
            return embeds;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

    }

    private static Map<String, String> getproductsInfo() throws IOException {
        String p_name, p_id;
        try {
            products = new HashMap<>();
            List<String> lines = IOUtils.readLines(new FileInputStream(new File("data/processingdata/products_newids.csv")), StandardCharsets.UTF_8);
            for (String line : lines) {
                String[] parts = line.split(",");
                p_id = parts[0];
                p_name = parts[1];
                products.put(p_id, p_name);
            }
            return products;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}

