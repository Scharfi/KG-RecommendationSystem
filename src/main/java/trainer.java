import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
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
 * A recommendation system that uses various embedding model as input
 * and an artificial neural network for learning these embedding models.
 * The testing of the results is based on a ranking approach
 * @author Salma Charfi
 */
public class trainer {
    static String modeler;
    static String embeddingPath;
    static String time;
    static INDArray embeds;

    public static void main(String[] args) throws Exception {

        modelConstructor modelinfo = new modelConstructor("conex"); // w2v(Word2Vec),r2v(RDF2Vec),pyke,conex,hybride
        modeler = modelinfo.getModelname();
        embeddingPath = modelinfo.getnormalizedEmbeddingsPath();
        embeds = list_embeddings();
        int dimension = modelinfo.getDimension();

        System.out.println(" The embedding model used is: \n" + modeler +
                "\n Dimension:\n " + dimension+
                "\n embedding path:\n" + embeddingPath);


        int labelIndexFrom = dimension;
        int labelIndexTo = (dimension * 2) - 1;
        int batchSize = 500;

        for (int fold = 0; fold < 1; fold++) {

            RecordReader rr = new CSVRecordReader();
            rr.initialize(new FileSplit(new File(modelinfo.getDatasetPath() + "_train_F" + fold + ".csv")));
            DataSetIterator trainData = new RecordReaderDataSetIterator.Builder(rr, batchSize).regression(labelIndexFrom, labelIndexTo).build();

            RecordReader rrt = new CSVRecordReader();
            rrt.initialize(new FileSplit(new File(modelinfo.getDatasetPath() + "_test_F" + fold + ".csv")));
            DataSetIterator testData = new RecordReaderDataSetIterator.Builder(rrt, 3000).regression(0, 0).build();

            //Configure neural network
            final int numInputs = modelinfo.dimension;
            int epochs = 1;
            int seed = 123;
            double learningRate = 0.5;
            int nOut = 9000;
            double l2 = 1e-7;


            System.out.println("building model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .updater(new Sgd(learningRate))
                    .l2(l2)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numInputs)
                            .activation(Activation.IDENTITY)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(new DenseLayer.Builder().nIn(numInputs).nOut(nOut)
                            .activation(Activation.RELU)
                            .build())
                    .layer(new OutputLayer.Builder()
                            .lossFunction(LossFunctions.LossFunction.MSE)
                            .activation(Activation.TANH) //IDENTITY nesterovs,0,9
                            .nIn(nOut).nOut(numInputs).build())
                    .build();

            System.out.println("init model....");
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(10));  //Print score every n parameter updates


            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH-mm-ss");
            LocalDateTime now = LocalDateTime.now();
            time = dtf.format(now);

            FileWriter file = new FileWriter("data/evaluation/" + modeler + "/" + time + "_" + modeler + "_evaluationResults_F" + fold + ".csv");

            String info = modeler + " Fold" + fold + " Config = " + model.getLayer(1) + "\n";
            System.out.println(info);
            file.append(info);

            System.out.println("fit model....");
            for (int i = 0; i < epochs; i++) {
                while (trainData.hasNext())
                    model.fit(trainData.next());
                trainData.reset();
            }

            System.out.println("output check and evaluation....");
            System.out.println("fit model....");
            System.out.println("epoch " + epochs);

            DataSet test = testData.next();
            INDArray feature = test.getFeatures();
            INDArray products = test.getLabels();
            System.out.println(" feature size ");
            System.out.println(feature.size(0));
            INDArray prediction = model.output(feature);

            info = "============ epoch " + epochs + "\n" + " Nbr of samples " + feature.size(0) + " \n";
            file.append(info);

            // evaluate predicted rows
            ranking_evaluation(prediction, products, file);

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
                        for (int l = 0; l < parts.length; l++) { //label list
                            int label = Integer.parseInt(parts[l]);
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
            List<String> lines = IOUtils.readLines(new FileInputStream(new File(embeddingPath)), StandardCharsets.UTF_8);
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