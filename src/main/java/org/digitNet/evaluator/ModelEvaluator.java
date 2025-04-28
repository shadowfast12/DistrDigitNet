package org.digitNet.evaluator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.digitNet.IDXFileReader;
import org.digitNet.IDXLabelReader;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;

public class ModelEvaluator {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println(
                    "Usage: ModelEvaluator <modelFile> <testImages> <testLabels> [batchSize]"
            );
            System.exit(1);
        }

        String modelFile   = args[0];
        String testImages  = args[1];
        String testLabels  = args[2];
        int batchSize      = args.length > 3
                ? Integer.parseInt(args[3])
                : 64;

        // Load the saved model
        MultiLayerNetwork model = ModelSerializer
                .restoreMultiLayerNetwork(modelFile);

        // Load test data
        IDXFileReader imagesReader = new IDXFileReader(testImages);
        IDXLabelReader labelsReader = new IDXLabelReader(testLabels);
        DataSet testData = new DataSet(
                imagesReader.getFeatures(),
                labelsReader.getLabels()
        );

        // Wrap in an iterator and evaluate
        var iter = new ListDataSetIterator<>(testData.asList(), batchSize);
        Evaluation eval = new Evaluation(labelsReader.getLabels().columns());
        while (iter.hasNext()) {
            var ds = iter.next();
            var output = model.output(ds.getFeatures(), false);
            eval.eval(ds.getLabels(), output);
        }

        // Print results
        System.out.println("=== Final Model Evaluation ===");
        System.out.println(eval.stats());
    }
}
