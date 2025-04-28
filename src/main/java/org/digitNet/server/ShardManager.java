package org.digitNet.server;

import org.digitNet.DataShard;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.List;

// Thread-safe queue of DataShard objects.
public class ShardManager {
    private final ConcurrentLinkedQueue<DataShard> queue; // SHARED QUEUE BETWEEN EACH PROCESS

    public ShardManager(List<DataShard> shards) {
        this.queue = new ConcurrentLinkedQueue<>(shards);
    }

    // Returns the next shard, or null if none remain
    public DataShard nextShard() {
        return queue.poll();
    }

    public boolean isEmpty(){
        return queue.isEmpty();
    }
}
