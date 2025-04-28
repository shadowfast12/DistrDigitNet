package org.digitNet.server;

/** Codes for server→client control messages. */
public enum MessageType {
    SHARD_DATA(1),
    NO_MORE_SHARDS(2);

    public final int code;
    MessageType(int code) { this.code = code; }

    public static MessageType fromCode(int code) {
        for (MessageType m : values()) {
            if (m.code == code) return m;
        }
        throw new IllegalArgumentException("Unknown MessageType: " + code);
    }
}
