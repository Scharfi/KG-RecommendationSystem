import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

/**
 * A class to load a pre-trained model that can be used for more training and testing.
 * @author Salma Charfi
 */
public class loader {
    static String modeler;
    static modelConstructor embeddingModelInfo;
    static boolean saveModel = false;
    static Map<String, String> parameters = new HashMap<>();

    public static void main(String[] args) throws Exception {
        parameters = read_parameters(args);
        modeler = parameters.get("modeler");
        System.out.println(parameters.toString());

        embeddingModelInfo = new modelConstructor(modeler); // w2v(Word2Vec),r2v(RDF2Vec),pyke,conex,hybride

        int dimension = embeddingModelInfo.getDimension();
        int labelIndexFrom = dimension;
        int labelIndexTo = (dimension * 2) - 1;
        int batchSize = 500;

        for (int fold = 0; fold < 1; fold++) {

            RecordReader rr = new CSVRecordReader();
            rr.initialize(new FileSplit(new File(embeddingModelInfo.getDatasetPath() + "_train_F" + fold + ".csv")));
            DataSetIterator trainData = new RecordReaderDataSetIterator.Builder(rr, batchSize).regression(labelIndexFrom, labelIndexTo).build();

            rr = new CSVRecordReader();
            rr.initialize(new FileSplit(new File(embeddingModelInfo.getDatasetPath() + "_test_F" + fold + ".csv")));
            DataSetIterator testData = new RecordReaderDataSetIterator.Builder(rr, 3000).regression(0, 0).build();

            // load the neural network model
            System.out.println("loading model....(Using Loader Class)");
            MultiLayerNetwork model = MultiLayerNetwork.load(new File("data/models/" + parameters.get("modelName") + ".model"), true);
            model.init();
            model.setListeners(new ScoreIterationListener(10));  //Print score every n parameter updates

            // prepare file and save model related information
            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH-mm-ss");
            LocalDateTime now = LocalDateTime.now();
            FileWriter file = new FileWriter("data/evaluation/" + modeler + "/" + dtf.format(now) + "_" + modeler + "_evaluationResults_F" + fold + ".csv");
            String info = modeler + " - Fold: " + fold + "\nConfig = " + model.getLayer(1) + "\nEpochs: " + parameters.get("epochs") + "\n";
            System.out.println(info);
            file.append(info);

            System.out.println("fit model....");
            int previous_epochs = Integer.parseInt(parameters.get("modelName").replaceAll("\\D+", ""));
            int epochs = Integer.parseInt(parameters.get("epochs"));

            for (int i = 0; i < epochs; i++) {
                while (trainData.hasNext())
                    model.fit(trainData.next());
                trainData.reset();
            }

            // save model
            if (saveModel)
                model.save(new File("data/models/" + modeler + (epochs+previous_epochs) + ".model"), true);

            System.out.println("output check and evaluation....");
            DataSet test = testData.next();
            INDArray feature = test.getFeatures();
            INDArray products = test.getLabels();
            INDArray prediction = model.output(feature);

            // evaluate predicted rows
            ranking_evaluation(prediction, products, file);

            //save evaluation to file
            try {
                file.flush();
                file.close();
            } catch (IOException e) {
                System.out.println("Error while flushing/closing fileWriter !!!");
                e.printStackTrace();
            }
        }
    }



    private static void ranking_evaluation(INDArray prediction, INDArray products, FileWriter file) {
        double similarity, sumOfCorrectPrediction = 0, ProductHasNoCorrectPrediction = 0, ProductHasCorrectPrediction = 0, count;
        double[] similarities;

        int productTotest;
        int[] indexes;
        int[] correctPredictionByLabelPosition = new int[10];
        int[] listOfCorrectPredictionPosition = new int[100];

        INDArray predictedVec;
        StringBuilder fileline = new StringBuilder();
        try {
            System.out.println("Evaluation ...");

            AssociativeSort sortarray = new AssociativeSort();
            Map<Integer, String> reclist = embeddingModelInfo.getRecommendationList();

            int numOfSamples = (int) prediction.size(0);
            fileline = fileline.append("Number of samples: ").append(numOfSamples);
            System.out.println(fileline.toString());

            // loop over all predicted vectors and count correct samples
            for (int k = 0; k < numOfSamples; k++) {
                count = 0;
                similarities = new double[15089];
                indexes = new int[15089];

                predictedVec = prediction.getRow(k);
                String productId = products.getRow(k).toStringFull();
                productTotest = Integer.parseInt(productId.substring(1, productId.length() - 1));

                // loop over the embeddings vectors and calculate cosine similarity between vectors
                if ("hybride".equals(modeler)) {
                    double[] predictedVecToDouble = predictedVec.toDoubleVector();
                    double[] predictionOfConex = new double[40];
                    double[] predictionOfW2v = new double[300];
                    //separate predicted vector into two parts: conex part and word2vec part
                    for (int v = 0; v < predictedVecToDouble.length; v++) {
                        if (v < 40)
                            predictionOfConex[v] = predictedVecToDouble[v];
                        else
                            predictionOfW2v[v - 40] = predictedVecToDouble[v];
                    }
                    INDArray embedsConex = embeddingModelInfo.list_embeddingsByName("conex");
                    INDArray embedsW2v = embeddingModelInfo.list_embeddingsByName("w2v");
                    for (int e = 0; e < embedsConex.size(0); e++) {
                        INDArray row_conex = embedsConex.getRow(e);
                        INDArray row_w2v = embedsW2v.getRow(e);

                        // Store  cosine similarities for all product
                        if (row_w2v != null && row_conex != null) {
                            double similarity_conex = cosineSimilarity(predictionOfConex, row_conex.toDoubleVector());
                            double similarity_w2v = cosineSimilarity(predictionOfW2v, row_w2v.toDoubleVector());
                            similarities[e] = (similarity_conex + similarity_w2v) / 2;
                        } else
                            similarities[e] = 0;

                        indexes[e] = e;

                    }
                } else {
                    INDArray embeds = embeddingModelInfo.list_embeddings();
                    for (int i = 0; i < embeds.size(0); i++) {
                        INDArray row = embeds.getRow(i);
                        // Store  cosine similarities for all product
                        similarity = cosineSimilarity(predictedVec.toDoubleVector(), row.toDoubleVector());
                        similarities[i] = similarity;
                        indexes[i] = i;
                    }
                }

                // Sort the predicted products as well as their indexes
                sortarray.quickSort(similarities, indexes);

                // test the predicted products(ids) with the true labels: check Top 100 product with 10 label
                String labelsList;
                if (reclist.containsKey(productTotest)) {
                    labelsList = reclist.get(productTotest);
                    String[] labels = labelsList.split(",");
                    for (int i = 0; i < 100; i++) { //top 100 reversed
                        int predictedProduct = indexes[15088 - i];
                        count = getCount(count, correctPredictionByLabelPosition, listOfCorrectPredictionPosition, labels, i, predictedProduct);
                    }
                } else
                    System.out.println("Product " + productTotest + "has no recommendation list");

                sumOfCorrectPrediction += count;

                // count products that have correct/not correct predictions
                if (count > 0)
                    ProductHasCorrectPrediction++;
                else ProductHasNoCorrectPrediction++;
            }

            // Overall results and evaluation
            fileline = new StringBuilder("Total correctly predicted products: " + sumOfCorrectPrediction +
                    "\nNumber of products that the model predicted correct products: " + ProductHasCorrectPrediction +
                    "\nNumber of products that the model did not predicted correct products: " + ProductHasNoCorrectPrediction);
            System.out.println(fileline);
            file.append(fileline.toString());

            // display and save the correctly predicted products according to the true labels positions
            fileline = new StringBuilder();
            for (int l = 0; l < 10; l++)
                fileline.append(correctPredictionByLabelPosition[l]).append(",");

            System.out.println("The correctly predicted products according to the true labels positions: \n" + fileline);
            file.append(fileline.toString()).append("\n");

            // evaluate the results
            calculeResults(numOfSamples, listOfCorrectPredictionPosition, sumOfCorrectPrediction, file);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void calculeResults(int numOfSamples, int[] listOfCorrectPredictionPosition, double sumOfCorrectPrediction, FileWriter file) {
        double value, sum_10 = 0, sum_15 = 0, sum_20 = 0, sum_25 = 0, sum_50 = 0, sumw_10 = 0;
        StringBuilder fileline = new StringBuilder();
        try {
            for (int l = 0; l < 100; l++) {
                value = listOfCorrectPredictionPosition[l];
                fileline.append(value).append(",");
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
            System.out.println("The correctly predicted products according to the top predicted list positions:\n" + fileline);
            file.append(fileline.toString()).append("\n");

            // calculate results
            System.out.println("Evaluation Results");
            double numOfSamplesBy10 = numOfSamples * 10;
            double microRecall_100 = sumOfCorrectPrediction / (numOfSamplesBy10);
            double microprecision = sumOfCorrectPrediction / (numOfSamples * 100);
            fileline = new StringBuilder("Micro recall@100 = " + sumOfCorrectPrediction + "/" + numOfSamplesBy10 + " = " + microRecall_100
                    + "\nMicro precision@100 = " + microprecision + "\n");
            System.out.println(fileline);
            file.append(fileline.toString());

            // Calcule of average recall at different ranks
            double microRecall_10 = sum_10 / numOfSamplesBy10;
            double microRecall_15 = sum_15 / numOfSamplesBy10;
            double microRecall_20 = sum_20 / numOfSamplesBy10;
            double microRecall_25 = sum_25 / numOfSamplesBy10;
            double microRecall_50 = sum_50 / numOfSamplesBy10;
            fileline = new StringBuilder("recall@10: " + microRecall_10
                    + "\nrecall@15: " + microRecall_15
                    + "\nrecall@20: " + microRecall_20
                    + "\nrecall@25: " + microRecall_25
                    + "\nrecall@50: " + microRecall_50
                    + "\nrecall@100: " + microRecall_100);
            System.out.println(fileline);
            file.append(fileline.toString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    static double getCount(double count, int[] correctPredictionByLabelPosition, int[] listOfCorrectPredictionPosition, String[] labels, int i, int predictedProduct) {
        // loop over the true label list to check if the predicted product is correct
        for (int l = 0; l < labels.length; l++) {
            int label = Integer.parseInt(labels[l]);
            if (predictedProduct == label) {
                correctPredictionByLabelPosition[l] = correctPredictionByLabelPosition[l] + 1;
                listOfCorrectPredictionPosition[i] = listOfCorrectPredictionPosition[i] + 1;
                count++;
            }
        }
        return count;
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

    public static Map<String, String> read_parameters(String[] args) {
        Map<String, String> params = new HashMap<>();
        try {
            if (args.length > 0) {
                params.put("modeler", args[0]);
                params.put("epochs", args[1]);
                params.put("modelName", "conex1");
                if (args[6] != null)
                    if ("Yes".equals(args[6]))
                        saveModel = true;
            } else {
                System.out.println("loading default values...");
                params.put("modeler", "conex");
                params.put("epochs", "1");
                params.put("modelName", "conex1");
            }
            return params;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}