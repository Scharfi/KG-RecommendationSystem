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
    private static Map<String, String> products;
    static String modeler;
    static String embeddingPath;
    static modelConstructor modelinfo;
    private static Map<String, String> embeddings;
    private static Map<String, String> recommends;

    public static void main(String[] args) throws Exception {

        modelinfo = new modelConstructor("conex"); // w2v(Word2Vec), r2v(RDF2Vec), pyke, conex, hybride
        modeler = modelinfo.getModelname();
        boolean centroid = false;

        System.out.println(" pre-processing data for " + modeler);

        //--------Data pre-processing
        // create new ids for all products
        //generateIds("data/processingdata/product_names.csv");
        //products = getproductsInfo();

        // update list of the embeddings with the new ids
        //embeddingPath= modelinfo.getEmbeddingsPath(); // original embedding vectors
        //UpdateEmbeddings(embeddingPath, products);

        // normalize data
        //normalizeVectors(modeler);

        // take normalized embeddings and generated training and test data
        embeddings = readEmbeddings();
        if (modeler=="hybride")
            createTrainandTestDataHybrid(embeddings, modeler); // Create Training and testing for HybridE
        else if (centroid)
            createCentroidTrainandTestData(embeddings, modeler); // Create Training and testing for the centroid assumption
        else
            createTrainandTestData(embeddings, modeler); //Create Training and testing for w2v(Word2Vec),r2v(RDF2Vec),pyke,conex

    }

    private static void normalizeVectors(String model) {
        try {
            FileWriter file = new FileWriter("data/normalizedEmbeddings/" + model + "_normalizedvectors.csv");
            List<String> lines = IOUtils.readLines(new FileInputStream(new File(embeddingPath)), StandardCharsets.UTF_8);

            int count = 0, count2 = 0;
            double magnitude, value;
            for (String line : lines) {
                String[] parts = line.split(",");
                double sum = 0;
                // Compute magnitude of vector
                for (int i = 1; i < parts.length; i++) {
                    sum = sum + Math.pow(Double.parseDouble(parts[i]), 2);
                }
                magnitude = Math.sqrt(sum);
                //normalize
                if (magnitude != 0) {
                    String normalizedline = parts[0];
                    for (int i = 1; i < parts.length; i++) {
                        value = Double.parseDouble(parts[i]) / magnitude;
                        String v = String.valueOf(value);
                        if (v.contains("E")) {
                            BigDecimal d = new BigDecimal(value);
                            normalizedline = normalizedline + "," + d.toPlainString().substring(0, 20);
                        } else if (!v.isEmpty() && v.contains("."))
                            normalizedline = normalizedline + "," + value;
                    }
                    if (normalizedline.contains("E"))
                        count2++;
                    if (!normalizedline.contains("E") && !normalizedline.contains(",,")) {
                        file.append(normalizedline);
                        file.append("\n");
                    }
                } else {
                    System.out.println("magnitude 0 for product id = " + parts[0]);
                    count++;
                }
            }
            System.out.println("magnitude 0 for " + count + " products " + "e value " + count2);
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

    public static void createTrainandTestData( Map<String, String> embeddings, String modelname) {
        try {
            int productid;
            String id, recId, fline = "", embed, recEmbed;
            String filepath = "data/trainingData/" + modeler + "/" + modeler;
            String Recommendationlist_path= "data/processingdata/Recommended_10ids.csv";

            for (int f = 0; f < 5; f++) {

                int n = 0, p = 0, k = 0;
                System.out.println(" creating train and test data for " + modeler + "Fold " + f);

                FileWriter trainf = new FileWriter(filepath + "_train_F" + f + ".csv");
                FileWriter testf = new FileWriter(filepath + "_test_F" + f + ".csv");

                List<String> lines = IOUtils.readLines(new FileInputStream(new File(Recommendationlist_path)), StandardCharsets.UTF_8);// productId, labellist{}

                for (String line : lines) {
                    String[] parts = line.split(",");
                    id = parts[0];
                    embed = FindEmbeddingsById(embeddings, id);
                    if (!(id.isEmpty()) && !(embed.isEmpty())) {
                        productid = Integer.parseInt(id);
                        //construct test data
                        if (productid % 5 == f) {
                            String testline = productid + "," + embed + "\n";
                            testf.append(testline);
                            n++;
                        } else {
                            //construct train data
                            for (int i = 1; i < parts.length; i++) {
                                recId = parts[i];
                                recEmbed = FindEmbeddingsById(embeddings, recId);

                                if (!recEmbed.isEmpty()) {
                                    fline = embed +","+recEmbed;
                                    trainf.append(fline);
                                    trainf.append("\n");
                                } else {
                                    k++;
                                }
                            }
                            p++;
                        }
                    } else {
                        k++;
                    }
                }
                System.out.println("Finished");
                System.out.println("Fold: "+f + " nbr of testing produtcs = " + n + " nbr of training products =" + p + " productout =" + k);
                try {
                    trainf.flush();
                    trainf.close();
                    testf.flush();
                    testf.close();
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


    public static void createTrainandTestDataHybrid(Map<String, String> embeddings, String modelname) {
        try {
            int productid;
            String id, recId, fline = "", embed, recEmbed, embed2, recEmbed2;
            Map<String, String> embeddingsConex = readEmbeddingsC();
            String Recommendationlist_path= "data/processingdata/Recommended_10ids.csv";
            String filepath = "data/trainingData/" + modelname + "/" + modelname;

            for (int f = 0; f < 5; f++) {

                int n = 0, p = 0, k = 0;
                System.out.println(" creating train and test data for " + modeler + "Fold " + f);

                FileWriter trainf = new FileWriter(filepath + "_train_F" + f + ".csv");
                FileWriter testf = new FileWriter(filepath + "_test_F" + f + ".csv");

                List<String> lines = IOUtils.readLines(new FileInputStream(new File(Recommendationlist_path)), StandardCharsets.UTF_8);// productId, labellist{}

                for (String line : lines) {
                    String[] parts = line.split(",");
                    id = parts[0];
                    embed = FindEmbeddingsById(embeddings, id);
                    embed2 = FindEmbeddingsById(embeddingsConex,id);

                    if (!(id.isEmpty()) && !(embed.isEmpty())&& !(embed2.isEmpty())) {
                        productid = Integer.parseInt(id);
                        //construct test data
                        if (productid % 5 == f) {
                            String testline = productid +","+embed2 + "," + embed + "\n";
                            testf.append(testline);
                            n++;
                        } else {
                            //construct train data
                            for (int i = 1; i < parts.length; i++) {
                                recId = parts[i];
                                recEmbed = FindEmbeddingsById(embeddings, recId);
                                recEmbed2 = FindEmbeddingsById(embeddingsConex, recId);
                                if (!recEmbed.isEmpty()&&!recEmbed2.isEmpty()) {
                                    fline = embed2 + "," + embed +","+ recEmbed2+","+recEmbed;
                                    trainf.append(fline);
                                    trainf.append("\n");
                                } else {
                                    k++;
                                }
                            }
                            p++;
                        }
                    } else {
                        k++;
                    }
                }
                System.out.println("Fold: "+f + " nbr of testing produtcs = " + n + " nbr of training products =" + p + " productout =" + k);
                try {
                    trainf.flush();
                    trainf.close();
                    testf.flush();
                    testf.close();
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

    public static void createCentroidTrainandTestData(Map<String, String> embeddings, String modelname) {
        try {
            int productid;
            String id, recId, fline = "", embed;
            INDArray embeds = list_embeddings();
            String Recommendationlist_path= "data/processingdata/Recommended_10ids.csv";
            String filepath = "data/trainingData/" + modelname + "/" + modelname;
            for (int f = 0; f < 1; f++) {

                int n = 0, p = 0, k = 0;
                System.out.println(" creating centroid train and test data for " + modelname + "Fold " + f);

                FileWriter trainf = new FileWriter(filepath + "_train_F" + f + "_Centroid.csv");
                FileWriter testf = new FileWriter(filepath + "_test_F" + f + ".csv");

                List<String> lines = IOUtils.readLines(new FileInputStream(new File(Recommendationlist_path)), StandardCharsets.UTF_8);

                for (String line : lines) {
                    String[] parts = line.split(",");
                    id = parts[0];
                    embed = FindEmbeddingsById(embeddings, id);
                    if (!(id.isEmpty()) && !(embed.isEmpty())) {
                        productid = Integer.parseInt(id);
                        //construct test data
                        if (productid % 5 == f) {
                            String testline = productid + "," + embed + "\n";
                            testf.append(testline);
                            n++;
                        } else {
                            //construct train data
                            INDArray row = Nd4j.zeros(new long[]{40});
                            for (int i = 1; i < parts.length; i++) {
                                recId = parts[i];
                                row = row.addRowVector(embeds.getRow(Integer.parseInt(recId)));
                            }
                            row.divi(10);
                                if (!row.isEmpty()) {
                                    fline = embed +","+row.toStringFull();
                                    trainf.append(fline);
                                    trainf.append("\n");
                                } else {
                                    System.out.println("recEmbed is empty!");
                                    k++;
                                }

                            p++;
                        }
                    } else {
                        k++;
                    }
                }
                System.out.println("Fold: "+f + " nbr of testing produtcs = " + n + " nbr of training products =" + p + " productout =" + k);
                try {
                    trainf.flush();
                    trainf.close();
                    testf.flush();
                    testf.close();
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

    private static INDArray list_embeddings() {
        int col = 0;
        try {
            if (modeler == "w2v") {
                col = 300;
            } else if (modeler == "r2v") {
                col = 500;
            } else if (modeler == "pyke") {
                col = 50;
            } else
                col = 40;
            INDArray embeds = Nd4j.create(new int[]{15089, col});
            List<String> lines = IOUtils.readLines(new FileInputStream(new File(modelinfo.getnormalizedEmbeddingsPath())), StandardCharsets.UTF_8);
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
        try {
            String path="";
            if (modeler=="hybride")
                path = "data/normalizedEmbeddings/w2v_normalizedvectors.csv";
            else
                path = modelinfo.getnormalizedEmbeddingsPath();
            List<String> lines = IOUtils.readLines(new FileInputStream(new File(path)), StandardCharsets.UTF_8);
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
            List<String> lines = IOUtils.readLines(new FileInputStream(new File(path)), StandardCharsets.UTF_8);
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
            int i = 0;
            FileWriter file = new FileWriter("data/postprocessingdata/products_newids.csv");
            List<String> lines = IOUtils.readLines(new FileInputStream(new File(csvFileClasspath)), StandardCharsets.UTF_8);

            for (String line : lines) {
                String[] parts = line.split(",");
                String id = parts[1].substring(4, parts[1].length() - 4);
                String prodName = parts[0].substring(4, parts[0].length() - 3);
                String fileline = i + "," + prodName;

                i++;
                file.append(fileline);
                file.append("\n");
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

    // Update the embedding file by changing the name of the product with its id
    public static void UpdateEmbeddings(String csvFileClasspath, Map<String, String> products) {

        try {
            FileWriter file = new FileWriter("data/processingdata/embeddingsId_updated.csv");
            List<String> lines = IOUtils.readLines(new FileInputStream(new File(csvFileClasspath)), StandardCharsets.UTF_8);
            String p_name, p_id, embedline;
            String fileline;
            int i = 8;
            for (String line : lines) {
                String[] parts = line.split(",");
                p_name = parts[0];
                p_id = FindProductId(products, p_name);
                if (!(p_id.isEmpty())) {
                    embedline = line.substring(line.indexOf(","));
                    fileline = p_id + embedline;
                    file.append(fileline);
                    file.append("\n");
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

    private static Map<String, String> getRecommendationList() {
        Map<String, String> list = new HashMap<>();
        String id;
        try {
            List<String> lines = IOUtils.readLines(new FileInputStream(new File("data/processingdata/Recommended_10ids.csv")), StandardCharsets.UTF_8);

            for (String line : lines) {
                id = line.substring(0, line.indexOf(","));
                list.put(id, line.substring(line.indexOf(",") + 1));
            }
            return list;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static String FindProductId(Map<String, String> products, String name) {
        String id = "";
        String prodname = "";
        int i = 8;
        try {
            for (Map.Entry<String, String> entry : products.entrySet()) {
                prodname = entry.getValue();
                if (prodname.equals(name)) {
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

    private static String FindEmbeddingsById(Map<String, String> embeddings, String p_id) {
        String embeds = "";
        String localId = "";
        try {
            for (Map.Entry<String, String> entry : embeddings.entrySet()) {
                localId = entry.getKey();
                if (localId.equals(p_id)) {
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
        try {

            String p_name, p_id;
            products = new HashMap<>();
            List<String> lines = IOUtils.readLines(new FileInputStream(new File("data/processingdata/products_newids.csv")), StandardCharsets.UTF_8);
            for (String line : lines) {
                String[] parts = line.split(",");
                p_id = parts[0];
                p_name = parts[1];
                products.put(p_id, p_name);
            }
            //System.out.print(products.get(8));
            return products;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}

