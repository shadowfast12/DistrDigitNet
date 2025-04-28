package org.digitNet;

import org.digitNet.IDXFileReader;
import org.digitNet.IDXLabelReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

/**
 * Load the full IDX train set once and split into N equal shards.
 */
public class DataLoader {
    public static List<DataShard> loadShards(
            String imagePath,
            String labelPath,
            int numShards
    ) {
        INDArray X = new IDXFileReader(imagePath).getFeatures();   // [N, D]
        INDArray Y = new IDXLabelReader(labelPath).getLabels();    // [N, C]
        int total    = (int) X.size(0);
        int perShard = total / numShards;

        List<DataShard> shards = new ArrayList<>(numShards);
        for (int i = 0; i < numShards; i++) {
            int start = i * perShard;
            int end   = (i == numShards - 1 ? total : start + perShard);
            INDArray Xi = X.get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
            INDArray Yi = Y.get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
            shards.add(new DataShard(Xi, Yi));
        }
        return shards;
    }
}
