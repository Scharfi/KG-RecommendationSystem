import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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
 * A class to load a pre-trained model that can be used for more training or testing.
 * @author Salma Charfi
 */
public class loader {
    static String modeler;
    static INDArray embeds;
    static modelConstructor modelinfo;
    static String embeddingPath;
    static INDArray embedsConex;
    static INDArray embedsW2v;

    public static void main(String[] args) throws Exception {

        modelinfo = new modelConstructor("hybrid");
        modeler = modelinfo.getModelname();
        int dimension = modelinfo.getDimension();
        if(modeler!="hybrid")
            embeds = list_embeddings();
        else{
            embedsConex = list_embeddings("data/normalizedEmbeddings/conex_normalizedvectors.csv");
            embedsW2v = list_embeddings("data/normalizedEmbeddings/w2v_normalizedvectors.csv");
        }


        int labelIndexFrom = dimension;
        int labelIndexTo = (dimension*2)-1;
        int batchSize = 500;

        for (int fold = 0; fold < 1; fold++) {

            RecordReader rr = new CSVRecordReader();
            rr.initialize(new FileSplit(new File(modelinfo.datasetPath+"_train_F" + 0 + ".csv")));
            DataSetIterator trainData = new RecordReaderDataSetIterator.Builder(rr, batchSize).regression(labelIndexFrom, labelIndexTo).build();

            RecordReader rrt = new CSVRecordReader();
            rrt.initialize(new FileSplit(new File(modelinfo.datasetPath+"_test_F" + 0 + ".csv")));
            DataSetIterator testData = new RecordReaderDataSetIterator.Builder(rrt, 3000).regression(0, 0).build();

            //Configure neural network
            int epochs = 2;
            int Additional_epochs = 10;

            System.out.println("loading model....");
            String Pathl = "data/evaluation/" + modeler + "/" + modeler + "_epoch.model";
            MultiLayerNetwork model = MultiLayerNetwork.load(new File(Pathl),true);
            model.init();
            model.setListeners(new ScoreIterationListener(10));  //Print score every n parameter updates

            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH-mm-ss");
            LocalDateTime now = LocalDateTime.now();
            String time = dtf.format(now);

            FileWriter file = new FileWriter("data/evaluation/" + modeler + "/" + time + "-1-" + modeler + "_evaluation_Results_F"+fold+".csv");
            String info = modeler + " Fold" + fold + " Config = " + model.getLayer(1) + "\n" +" LOADER \n";
            System.out.println(info);

            /*for (int i=0;i<Additional_epochs;i++){
                while (trainData.hasNext())
                    model.fit(trainData.next());
                trainData.reset();
            }
            epochs+=Additional_epochs;*/

            System.out.println("output check and evaluation....");
                System.out.println("fit model....");
                System.out.println("output check and evaluation...." + (epochs));

                // get test data and predict
                DataSet test = testData.next();
                INDArray feature = test.getFeatures();
                INDArray products = test.getLabels();
                INDArray prediction = model.output(feature);

                System.out.println(" feature size "+feature.size(0));
                info = "============ epoch " + epochs + "\n" + " Nbr of samples " + feature.size(0) + " \n";
                file.append(info);


                // evaluate predicted rows
                if (modeler!="hybrid")
                    ranking_evaluation(prediction, products, file);
                else
                    ranking_evaluationH(prediction,products, file);

                // save model
                /*
                if(epochs==9) {
                    String Path = "data/evaluation/" + modeler + "/" + modeler +"_"+epochs+ ".model";
                    model.save(new File(Path), true);
                }*/


            try {
                file.flush();
                file.close();
            } catch (IOException e) {
                System.out.println("Error while flushing/closing fileWriter !!!");
                e.printStackTrace();
            }


        }

    }

    private static void ranking_evaluation(INDArray prediction, INDArray products, FileWriter file) throws IOException {

        double similarity, sumT = 0, sumout = 0, sumin = 0, count;
        INDArray predvec;
        int productId;
        String fileline;
        int[] index;
        double[] similarities;
        int[] labellist = new int[10];
        int[] p_position = new int[100];

        try {
            System.out.println("Evaluation ...");

            AssociativeSort sortarray = new AssociativeSort();
            Map<Integer, String> reclist = getRecommendationList();

            int numOfSamples = (int) prediction.size(0);
            System.out.println("Nbr of samples " + numOfSamples);

            // loop over all predicted vectors and count correct samples
            for (int k = 0; k < numOfSamples; k++) {

                count = 0;
                similarities = new double[15089];
                index = new int[15089];

                predvec = prediction.getRow(k);
                String productid = products.getRow(k).toStringFull();
                productId = Integer.parseInt(productid.substring(1, productid.length() - 1));

                // loop over the embeddings vectors and calculate cosine similarity between vectors
                for (int e = 0; e < embeds.size(0); e++) {
                    INDArray row = embeds.getRow(e);

                    // Store  cosine similarities for all product
                    similarity = cosineSimilarity(predvec.toDoubleVector(), row.toDoubleVector());
                    similarities[e] = similarity;
                    index[e] = e;
                }
                // Sort the predicted array as well as its indexes
                double[] transformed = sortarray.quickSort(similarities, index);

                // evaluate predicted ids with true labels
                String line = "";
                String list = "";
                // check Top 100 product with 10 label
                if (reclist.containsKey(productId)) {
                    list = reclist.get(productId);
                    String[] parts = list.split(",");
                    for (int i = 0; i < 100; i++) { //top 100 reversed
                        int e = index[15088 - i];
                        line = line + e + ",";
                        count = trainer.getCount(count, labellist, p_position, parts, i, e);
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
                    "\n";
            System.out.println(fileline);
            file.append(fileline);

            fileline = "";
            for (int l = 0; l < 10; l++)
                fileline += labellist[l] + ",";
            System.out.println(fileline);
            file.append(fileline);
            file.append("\n");

            // calculate results
            calculeResults(numOfSamples, p_position, sumT, file);

            System.out.println("Evaluation ending ...");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void calculeResults(int numOfSamples, int[] p_position, double sumT, FileWriter file) {

        try {
            String fileline = "";
            double value, sum_10 = 0, sum_15 = 0, sum_20 = 0, sum_25 = 0, sum_50 = 0;
            for (int l = 0; l < 100; l++) {
                value = p_position[l];
                fileline = fileline + value + ",";
                if (l < 10)
                    sum_10 += value;
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

    private static INDArray list_embeddings() {
        int col = 0;
        try {
            if (modeler == "w2v") {
                col = 300;
            } else if (modeler == "r2v") {
                col = 100;
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

    private static void ranking_evaluationH( INDArray prediction,INDArray products, FileWriter file) throws IOException {

        double similarity_conex, similarity_w2v, sumT = 0, sumout = 0, sumin = 0, count;
        INDArray predvec;
        int product;
        String fileline, productid;
        int[] index;
        double[] similarities;
        int[] labellist = new int[10];
        int[] p_position = new int[100];

        try {
            System.out.println("Evaluation ...");

            AssociativeSort sortarray = new AssociativeSort();
            Map<Integer, String> reclist = getRecommendationList();

            int numOfSamples = (int) prediction.size(0);
            System.out.println("Nbr of samples " + numOfSamples);

            // loop over all predicted vectors and count correct samples
            for (int k = 0; k < 2; k++) {
                count = 0;
                similarities = new double[15089];
                index = new int[15089];
                predvec = prediction.getRow(k);
                productid = products.getRow(k).toStringFull();
                product = Integer.parseInt(productid.substring(1, productid.length() - 1));

                double[] p1 = predvec.toDoubleVector();
                double[] p_conex = new double[40];
                double[] p_w2v = new double[300];
                for (int v = 0; v < p1.length; v++) {
                    if (v < 40)
                        p_conex[v] = p1[v];
                    else
                        p_w2v[v - 40] = p1[v];
                }

                // loop over the embeddings vectors and calculate cosine similarity between vectors
                for (int e = 0; e < embedsConex.size(0); e++) {
                    INDArray row_conex = embedsConex.getRow(e);
                    INDArray row_w2v = embedsW2v.getRow(e);

                    // Store  cosine similarities for all product
                    similarity_conex = cosineSimilarity(p_conex, row_conex.toDoubleVector());
                    similarity_w2v = cosineSimilarity(p_w2v, row_w2v.toDoubleVector());
                    similarities[e] = (similarity_conex + similarity_w2v) / 2;
                    index[e] = e;

                }

                // Sort the predicted array as well as its indexes
                double[] transformed = sortarray.quickSort(similarities, index);

                // evaluate predicted ids with true labels
                String line = "";
                String list = "";
                // check Top 100 product with 10 label
                if (reclist.containsKey(product)) {
                    list = reclist.get(product);
                    String[] parts = list.split(",");
                    for (int i = 0; i < 100; i++) { //top 100 reversed
                        int e = index[15088 - i];
                        line = line + e + ",";
                        count = trainer.getCount(count, labellist, p_position, parts, i, e);
                    }
                }

                //sum total correct prediction
                sumT = sumT + count;
                if (count > 0)
                    sumin++;
                else sumout++;

            }

            // Overall results and evaluation
            fileline = "Total correct samples: " + sumT +
                    "\n" + "Nbr of products in: " + sumin +
                    "\n" + "Nbr of products out: " + sumout +
                    "\n";
            System.out.println(fileline);
            file.append(fileline);

            fileline = "";
            for (int l = 0; l < 10; l++)
                fileline += labellist[l] + ",";
            System.out.println(fileline);
            file.append(fileline);
            file.append("\n");

            // calculate results
            calculeResults(numOfSamples, p_position, sumT, file);

            System.out.println("Evaluation ending ...");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static INDArray list_embeddings(String path) {
        int col = 0;
        try {
            if (path.contains("w2v")) {
                col = 300;
            } else if (path.contains("r2v")) {
                col = 500;
            } else if (path.contains("pyke")) {
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