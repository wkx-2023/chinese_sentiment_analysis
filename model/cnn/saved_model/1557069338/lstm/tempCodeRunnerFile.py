    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    _, (cf, hf) = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    _, (cb, hb) = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([hf, hb], axis=-1)
    output = tf.layers.dropout(output, rate=dropout, training=training)