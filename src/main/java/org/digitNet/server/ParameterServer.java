package org.digitNet.server;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.digitNet.DataLoader;
import org.digitNet.DataShard;
import java.io.File;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class ParameterServer {
    public static void main(String[] args) throws Exception {
        if (args.length != 7) {
            System.err.println(
                    "Usage: ParameterServer <port> <learningRate> <localEpochs> " +
                            "<batchSize> <numShards> <trainImages> <trainLabels>"
            );
            System.exit(1);
        }
        int    port        = Integer.parseInt(args[0]);
        double lr          = Double.parseDouble(args[1]);
        int    localEpochs = Integer.parseInt(args[2]);
        int    batchSize   = Integer.parseInt(args[3]);
        int    numShards   = Integer.parseInt(args[4]);
        String trainImgs   = args[5];
        String trainLbls   = args[6];

        // 1) Build your CNN config
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(lr))
                .weightInit(WeightInit.RELU)
                .l2(1e-4)
                .list()
                .layer(new ConvolutionLayer.Builder(3,3)
                        .nIn(1).nOut(32)
                        .stride(1,1).padding(1,1)
                        .activation(Activation.RELU).build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2).stride(2,2).build())
                .layer(new DenseLayer.Builder()
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();

        // 2) Init global model
        MultiLayerNetwork globalModel = new MultiLayerNetwork(conf);
        globalModel.init();

        // 3) Pre-shard the dataset
        List<DataShard> shards = DataLoader.loadShards(trainImgs, trainLbls, numShards);
        ShardManager shardManager = new ShardManager(shards);

        // 4) Progress counter + heartbeat
        AtomicInteger shardsDone = new AtomicInteger(0);
        ScheduledExecutorService hb = Executors.newSingleThreadScheduledExecutor();
        hb.scheduleAtFixedRate(() -> {
            int done = shardsDone.get();
            int left = numShards - done;
            double pct = 100.0 * done / numShards;
            System.out.printf("[heartbeat] %d/%d shards done (%.1f%%), %d remaining%n",
                    done, numShards, pct, left);
        }, 0, 1, TimeUnit.SECONDS);

        // 5) Serve clients
        ExecutorService pool = Executors.newCachedThreadPool();
        try (ServerSocket server = new ServerSocket(port)) {
            server.setSoTimeout(1000);  // 1 s accept timeout
            System.out.println("Server listening on port " + port);

            while (true) {
                if (shardManager.isEmpty()) break;
                try {
                    Socket client = server.accept();
                    pool.submit(new ClientHandler(
                            client, shardManager, globalModel,
                            lr, localEpochs, batchSize,
                            shardsDone, numShards
                    ));
                } catch (SocketTimeoutException e) {
                    // no client this second – loop to check isEmpty()
                }
            }
        }

        // 6) Wait for handlers
        pool.shutdown();
        pool.awaitTermination(1, TimeUnit.HOURS);
        hb.shutdownNow();

        // 7) Save model
        File out = new File("globalModel.zip");
        ModelSerializer.writeModel(globalModel, out, true);
        System.out.println(" Training complete; model saved to " + out.getAbsolutePath());
    }
}
