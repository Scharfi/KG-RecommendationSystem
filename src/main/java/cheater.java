import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
/**
 * Experiment 3: Cheater recommender
 * A class that generate centroid vectors for each produtc found in the test set
 * and evaluate these vectors based on a ranking-based approach
 * @author Salma Charfi
 */
public class cheater {
    static String modeler;
    static String time;
    static INDArray embeds;

    public static void main(String[] args) throws Exception {

        modelConstructor modelinfo = new modelConstructor("conex");
        modeler = modelinfo.getModelname();
        embeds = list_embeddings(modelinfo.getnormalizedEmbeddingsPath());
        int dimension = modelinfo.dimension;

        System.out.println(" The embedding model used is: \n" + modeler +
                "\n Dimension:\n " + dimension+
                "\n embedding path:\n" + modelinfo.getnormalizedEmbeddingsPath());


        RecordReader rrt = new CSVRecordReader();
        rrt.initialize(new FileSplit(new File(modelinfo.getDatasetPath()+"_test_F" + 0 + ".csv")));
        DataSetIterator testData = new RecordReaderDataSetIterator.Builder(rrt, 3000).regression(0, 0).build();
        DataSet test = testData.next();
        INDArray productIds = test.getLabels();

        int productId;
        INDArray centroid_matrix = Nd4j.create(new long[]{2313,40});
        Map<Integer, String> reclist = getRecommendationList();

        for(int p=0;p< productIds.size(0);p++){
            String productid = productIds.getRow(p).toStringFull();
            productId = Integer.parseInt(productid.substring(1,productid.length()-1));
            String[] labels = reclist.get(productId).split(",");
            //System.out.println("Product: "+product+" reclist is: "+reclist.get(product));

            INDArray row = Nd4j.zeros(new long[]{40});
            //System.out.println(row.toStringFull());
            for (int l=0;l<labels.length;l++) {
                row = row.addRowVector(embeds.getRow(Integer.parseInt(labels[l])));
            }
            row.divi(10);
            centroid_matrix.putRow(p,row);
        }


            String info = modeler + " Fold 0 " + "\n";
            System.out.println(info);

            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH-mm-ss");
            LocalDateTime now = LocalDateTime.now();
            time = dtf.format(now);

            FileWriter file = new FileWriter("data/evaluation/" +modeler+ "/cheater_" + time + "_" + modeler + "_evaluationResults_F"+".csv");
            ranking_evaluation(centroid_matrix,productIds, file);

            try {
                file.flush();
                file.close();
            } catch (IOException e) {
                System.out.println("Error while flushing/closing fileWriter !!!");
                e.printStackTrace();
            }
        }


    private static void ranking_evaluation(INDArray centroid_matrix,INDArray productIds, FileWriter file) throws IOException {

        double similarity, sumT = 0, sumout = 0, sumin = 0, count;
        INDArray centroid_vec;
        int  product=0, toprec = 0, countout = 0;
        String fileline;
        int[] index;
        double[] similarities;
        int[] labellist = new int[10];
        int[] p_position = new int[100];

        try {
            System.out.println("Evaluation ...");

            AssociativeSort sortarray = new AssociativeSort();
            Map<Integer, String> reclist = getRecommendationList();

            int numOfSamples = (int) centroid_matrix.size(0);
            System.out.println("Nbr of samples " + numOfSamples);

            // loop over all predicted vectors and count correct samples
            for (int k = 0; k < numOfSamples; k++) {
                count = 0;
                similarities = new double[15089];
                index = new int[15089];

                centroid_vec = centroid_matrix.getRow(k);

                // loop over the embeddings vectors and calculate cosine similarity between vectors
                for (int e = 0; e < embeds.size(0); e++) {
                    INDArray row = embeds.getRow(e);
                    // Store  cosine similarities for all product
                    similarity = cosineSimilarity(centroid_vec.toDoubleVector(), row.toDoubleVector());
                    similarities[e] = similarity;
                    index[e] = e;
                }
                // Sort the predicted array as well as its indexes
                double[] transformed = sortarray.quickSort(similarities, index);

                // evaluate predicted ids with true labels
                String productid = productIds.getRow(k).toStringFull();
                product = Integer.parseInt(productid.substring(1,productid.length()-1));
                String line = "";
                String list = "";
                    // check Top 100 product with 10 label
                    if (reclist.containsKey(product)) {
                        list = reclist.get(product);
                        String[] labels = list.split(",");
                        for (int i = 0; i < 100; i++) { //top 100 reversed
                            int e = index[15088 - i];
                            line = line + e + ",";
                            for (int l = 0; l < labels.length; l++) { //label list
                                int label = Integer.parseInt(labels[l]);
                                if (e == label) {
                                    labellist[l] = labellist[l] + 1;
                                    p_position[i] = p_position[i] + 1;
                                    count++;
                                }
                            }
                        }
                    }

                    //sum total correct prediction
                    sumT = sumT + count;
                    if (count > 0)
                        sumin++;
                    else sumout++;

            }

            // Overall results and evaluation
            fileline = "Total correct samples " + sumT +
                    "\n" + "Nbr of products in " + sumin +
                    "\n" + "Nbr of products out  ------" + sumout +
                    "\n" + "Nbr of products not recognized " + countout +
                    "\n";
            System.out.println(fileline);
            file.append(fileline);

            fileline = "";
            for (int l = 0; l < 10; l++)
                fileline += labellist[l] + ",";
            System.out.println(fileline);
            file.append(fileline);file.append("\n");

            // calculate results
            calculeResults(numOfSamples,p_position,sumT,file);


            System.out.println("Evaluation ending ...");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void calculeResults(int numOfSamples, int[] p_position, double sumT, FileWriter file) {

        try {
            String fileline = "";
            double value, sum_10 = 0, sum_15 = 0, sum_20 = 0, sum_25 = 0, sum_50 = 0, sumw_10 = 0, sumw_15 = 0, sumw_20 = 0, sumw_25 = 0, sumw_50 = 0, sumw_100 = 0;
            for (int l = 0; l < 100; l++) {
                value = p_position[l];
                fileline = fileline + value + ",";
                if (l < 10) {
                    sumw_10 += ((double) value / (l + 1));
                    sum_10 += value;
                }
                if (l < 15)
                    sum_15 += value;
                if (l < 20)
                    sum_20 += value;
                if (l < 25)
                    sum_25 += value;
                if (l < 50)
                    sum_50 += value;
            }
            System.out.println(fileline);
            System.out.println("s50= " + sum_50);
            file.append(fileline);
            file.append("\n");

            // calculate results
            double numOfSamples_10 = numOfSamples * 10;
            double microrecall = sumT / (numOfSamples_10);
            double microprecision = sumT / (numOfSamples * 100);
            fileline = "Micro recall@100: " + sumT + "/" + numOfSamples_10 + "= " + microrecall + "\n Micro precision@100 = " + microprecision + "\n";
            System.out.println(fileline);
            file.append(fileline);

            // Calcule of average recall at different ranks
            double recall_10 = sum_10 / numOfSamples_10;
            double recall_15 = sum_15 / numOfSamples_10;
            double recall_20 = sum_20 / numOfSamples_10;
            double recall_25 = sum_25 / numOfSamples_10;
            double recall_50 = sum_50 / numOfSamples_10;
            fileline = "recall@10: " + recall_10
                    + "\n recall@15: " + recall_15
                    + "\n recall@20: " + recall_20
                    + "\n recall@25: " + recall_25
                    + "\n recall@50: " + recall_50
                    + "\n";
            System.out.println(fileline);
            file.append(fileline);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static INDArray list_embeddings(String path) {
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
            System.out.println("model" + modeler + "col" + col + " " + path);
            INDArray embeds = Nd4j.create(new int[]{15089, col});
            List<String> lines = IOUtils.readLines(new FileInputStream(new File(path)), StandardCharsets.UTF_8);
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

    private static int[] getfeatureasIds() {
        int[] list = new int[12];
        try {
            List<String> lines = IOUtils.readLines(new FileInputStream(new File("data/ids.csv")), StandardCharsets.UTF_8);
            for (int i = 0; i < lines.size(); i++) {
                list[i] = Integer.parseInt(lines.get(i));
            }
            return list;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static int searchinToprecommendation(String predictedlist) {
        int count = 0;
        String fileline = "";
        try {
            List<String> recs = IOUtils.readLines(new FileInputStream(new File("data/processingdata/toprecommended.csv")), StandardCharsets.UTF_8);
            String[] parts = predictedlist.split(",");
            for (int i = 0; i < parts.length; i++) {
                int id = Integer.parseInt(parts[i]);
                for (String r : recs) {
                    int rec = Integer.parseInt(r);
                    if (id == rec) {
                        fileline = fileline + id + ",";
                        count++;
                    }
                }
            }
            //if (k == 50 || k == 150 || k == 300 || k == 530 || k == 650 || k == 700) {
            //file.append(" list of products found in top recommended list: ");
            //file.append(fileline);
            //file.append("\n");
            //fileline = "Count of products in top recommended list: " + count + "\n";
            //file.append(fileline);
            //}
            return count;
        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    private static Map<Integer, String> getRecommendationList() {
        Map<Integer, String> list = new HashMap<>();
        String id;
        try {
            List<String> lines = IOUtils.readLines(new FileInputStream(new File("data/processingdata/Recommended_10ids.csv")), StandardCharsets.UTF_8);
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

    public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}