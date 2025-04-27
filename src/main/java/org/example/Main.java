package org.example;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.digitNet.IDXFileReader;
import org.digitNet.IDXLabelReader;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {

        // file paths
        String trainImages = "C:/Users/Zach/Downloads/MNIST_ORG/train-images.idx3-ubyte";
        String trainLabels = "C:/Users/Zach/Downloads/MNIST_ORG/train-labels.idx1-ubyte";
        String testImages = "C:/Users/Zach/Downloads/MNIST_ORG/t10k-images.idx3-ubyte";
        String testLabels = "C:/Users/Zach/Downloads/MNIST_ORG/t10k-labels.idx1-ubyte";


        // network hyperparameters
        int seed          = 123;
        double lr         = 1e-3;
        int batchSize     = 64;
        int nEpochs       = 10;
        int height        = 28;
        int width         = 28;
        int channels      = 1;
        int numClasses    = 10;

        // ---- 1) Build a small CNN (you can switch back to Dense only if you like) ----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .list()
                // conv layer 1
                .layer(new ConvolutionLayer.Builder(5,5)
                        .nIn(channels).nOut(20)
                        .stride(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2).stride(2,2).build())
                // conv layer 2
                .layer(new ConvolutionLayer.Builder(5,5)
                        .nOut(50).stride(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2).stride(2,2).build())
                // fully connected
                .layer(new DenseLayer.Builder()
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // ---- 2) Load train data and labels ----
        IDXFileReader   trainImagesReader = new IDXFileReader(trainImages);
        IDXLabelReader  trainLabelsReader = new IDXLabelReader(trainLabels);
        DataSet         fullTrainData     = new DataSet(
                trainImagesReader.getFeatures(),
                trainLabelsReader.getLabels()
        );

        // shuffle & batch the data
        List<DataSet> listTrain = fullTrainData.asList();
        Collections.shuffle(listTrain, new java.util.Random(seed));
        DataSetIterator trainIter = new ListDataSetIterator<>(listTrain, batchSize);

        // normalize pixel values to [0,1]
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        // TRAIN THE DATA USING MULTIPLE EPOCHS
        System.out.println("Starting training...");
        for (int epoch = 1; epoch <= nEpochs; epoch++) {
            trainIter.reset();
            model.fit(trainIter);
            System.out.println("Completed epoch " + epoch + "/" + nEpochs);
        }

        // TEST THE MODEL
        IDXFileReader   testImagesReader = new IDXFileReader(testImages);
        IDXLabelReader  testLabelsReader = new IDXLabelReader(testLabels);
        DataSet         testData         = new DataSet(
                testImagesReader.getFeatures(),
                testLabelsReader.getLabels()
        );
        DataSetIterator testIter = new ListDataSetIterator<>(testData.asList(), batchSize);
        testIter.setPreProcessor(scaler);

        // EVALUATE THE MODEL
        Evaluation eval = new Evaluation(numClasses);
        while(testIter.hasNext()){
            DataSet batch = testIter.next();
            INDArray output = model.output(batch.getFeatures(), false);
            eval.eval(batch.getLabels(), output);
        }

        System.out.println(eval.stats());
    }
}
