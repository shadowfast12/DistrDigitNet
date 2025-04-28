package org.digitNet;

import org.nd4j.linalg.api.ndarray.INDArray;

// HOLDER FOR CLIENT'S SLICE OF THE DATA
public class DataShard {
    private final INDArray features;
    private final INDArray labels;

    public DataShard(INDArray features, INDArray labels) {
        this.features = features;
        this.labels   = labels;
    }

    public INDArray getFeatures() {
        return features;
    }

    public INDArray getLabels() {
        return labels;
    }
}
