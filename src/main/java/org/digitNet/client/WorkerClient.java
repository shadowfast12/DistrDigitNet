package org.digitNet.client;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.digitNet.SerializationUtil;
import org.digitNet.server.MessageType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.gradient.Gradient;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;
import java.util.List;

/**
 * WorkerClient for synchronous FedAvg:
 *   Connects once per global round, trains locally, returns updated params.
 */
public class WorkerClient {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println(
                    "Usage: WorkerClient <masterHost> <masterPort> <globalEpochs>"
            );
            System.exit(1);
        }
        String masterHost   = args[0];
        int    masterPort   = Integer.parseInt(args[1]);
        int    globalEpochs = Integer.parseInt(args[2]);

        for (int round = 1; round <= globalEpochs; round++) {
            System.out.println("→ Starting local training for global round " + round);

            try (Socket sock = new Socket(masterHost, masterPort);
                 DataOutputStream out = new DataOutputStream(sock.getOutputStream());
                 DataInputStream  in  = new DataInputStream(sock.getInputStream()))
            {
                // 1) Receive model JSON
                int cLen = in.readInt();
                byte[] cBuf = new byte[cLen];
                in.readFully(cBuf);
                MultiLayerConfiguration conf =
                        MultiLayerConfiguration.fromJson(new String(cBuf, "UTF-8"));
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();

                // 2) Receive hyperparameters
                int localEpochs = in.readInt();
                int batchSize   = in.readInt();

                // 3) Receive global parameters
                int pLen = in.readInt();
                byte[] pBuf = new byte[pLen];
                in.readFully(pBuf);
                model.setParams(Nd4j.fromByteArray(pBuf));

                // 4) Receive shard data
                int xLen = in.readInt();
                byte[] xBuf = new byte[xLen];
                in.readFully(xBuf);
                INDArray X = Nd4j.fromByteArray(xBuf);

                int yLen = in.readInt();
                byte[] yBuf = new byte[yLen];
                in.readFully(yBuf);
                INDArray Y = Nd4j.fromByteArray(yBuf);

                // 5) Local training (FedAvg style via fit)
                List<DataSet> list = new DataSet(X, Y).asList();
                ListDataSetIterator<DataSet> iter =
                        new ListDataSetIterator<>(list, batchSize);

                for (int e = 0; e < localEpochs; e++) {
                    iter.reset();
                    model.fit(iter);
                }

                // 6) Send back updated parameters
                byte[] upd = Nd4j.toByteArray(model.params());
                out.writeInt(upd.length);
                out.write(upd);
                out.flush();
            }
        }

        System.out.println("Worker done all " + globalEpochs + " rounds.");
    }
}
