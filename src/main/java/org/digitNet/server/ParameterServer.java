package org.digitNet.server;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.digitNet.DataLoader;
import org.digitNet.DataShard;
import org.digitNet.SerializationUtil;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Synchronous federated averaging:
 *  Each global round:
 *   - Accept one connection per shard
 *   - Send model JSON, hyperparams, global params, shard data
 *   - Receive updated params from each client
 *   - Average them into new global params
 *  At end, save globalModel.zip
 */
public class ParameterServer {
    public static void main(String[] args) throws Exception {
        if (args.length < 9) {
            System.err.println(
                    "Usage: ParameterServer <port> <learningRate> <localEpochs> " +
                            "<batchSize> <numShards> <globalEpochs> <trainImages> <trainLabels>"
            );
            System.exit(1);
        }
        int    port           = Integer.parseInt(args[0]);
        double lr             = Double.parseDouble(args[1]);
        int    localEpochs    = Integer.parseInt(args[2]);
        int    batchSize      = Integer.parseInt(args[3]);
        int    numShards      = Integer.parseInt(args[4]);
        int    globalEpochs   = Integer.parseInt(args[5]);
        String trainImages    = args[6];
        String trainLabels    = args[7];

        // Build a deeper CNN, specifying weightInit per layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(lr))
                .l2(1e-4)
                .list()

                // Conv block 1
                .layer(new ConvolutionLayer.Builder(3,3)
                        .nIn(1).nOut(32)
                        .stride(1,1).padding(1,1)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2).stride(2,2).build())

                // Conv block 2
                .layer(new ConvolutionLayer.Builder(3,3)
                        .nOut(64).stride(1,1).padding(1,1)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2).stride(2,2).build())

                // Conv block 3
                .layer(new ConvolutionLayer.Builder(3,3)
                        .nOut(128).stride(1,1).padding(1,1)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2).stride(2,2).build())

                // Dense + Dropout
                .layer(new DenseLayer.Builder()
                        .nOut(256)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.RELU)
                        .dropOut(0.5)
                        .build())

                // Output
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())

                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();

        // Initialize model once
        MultiLayerNetwork globalModel = new MultiLayerNetwork(conf);
        globalModel.init();

        // Pre-split training data
        List<DataShard> shards = DataLoader.loadShards(trainImages, trainLabels, numShards);

        try (ServerSocket server = new ServerSocket(port)) {
            System.out.println("ParameterServer listening on port " + port);

            // Global training rounds
            for (int round = 1; round <= globalEpochs; round++) {
                System.out.println("=== Global Round " + round + "/" + globalEpochs + " ===");
                List<INDArray> collectedParams = new ArrayList<>(numShards);

                for (int i = 0; i < numShards; i++) {
                    Socket sock = server.accept();
                    System.out.println(" Client connected for shard " + i + ": " +
                            sock.getRemoteSocketAddress());

                    try (DataOutputStream out = new DataOutputStream(sock.getOutputStream());
                         DataInputStream  in  = new DataInputStream(sock.getInputStream()))
                    {
                        // 1) Send model JSON
                        byte[] js = conf.toJson().getBytes(StandardCharsets.UTF_8);
                        out.writeInt(js.length);
                        out.write(js);

                        // 2) Send hyperparameters
                        out.writeInt(localEpochs);
                        out.writeInt(batchSize);

                        // 3) Send global parameters
                        byte[] gp = SerializationUtil.toBytes(globalModel.params());
                        out.writeInt(gp.length);
                        out.write(gp);

                        // 4) Send this shard
                        DataShard shard = shards.get(i);
                        byte[] xb = SerializationUtil.toBytes(shard.getFeatures());
                        out.writeInt(xb.length);
                        out.write(xb);
                        byte[] yb = SerializationUtil.toBytes(shard.getLabels());
                        out.writeInt(yb.length);
                        out.write(yb);

                        out.flush();

                        // 5) Receive updated params
                        int updLen = in.readInt();
                        byte[] updBuf = new byte[updLen];
                        in.readFully(updBuf);
                        INDArray updated = SerializationUtil.fromBytes(updBuf);
                        collectedParams.add(updated);
                    }
                    sock.close();
                }

                // Average all collected parameter arrays
                INDArray sum = collectedParams.get(0).dup();
                for (int j = 1; j < collectedParams.size(); j++) {
                    sum.addi(collectedParams.get(j));
                }
                INDArray avg = sum.div(collectedParams.size());
                globalModel.setParams(avg);
            }
        }

        // Save final model
        File outFile = new File("globalModel.zip");
        ModelSerializer.writeModel(globalModel, outFile, true);
        System.out.println(" Saved final model to " + outFile.getAbsolutePath());
    }
}
