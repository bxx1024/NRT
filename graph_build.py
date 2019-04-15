import tensorflow as tf


class graph_build:
    def __init__(self, para_dict):
        # inputs
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_y_tip = tf.placeholder(tf.int32, [None, para_dict['max_tip_len']], name="input_y_tip")
        self.review_node_idxs = tf.placeholder(tf.int32, [None, para_dict['review_len']])
        self.review_attn_dist = tf.placeholder(tf.float32, [None, para_dict['review_len']])
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")

        self.W_tip = tf.get_variable("W_tip", [para_dict['tip_vocab_size'], para_dict['embedding_tip_size']], dtype=tf.float32)
        self.iidW = tf.get_variable("iidW", [para_dict['item_num'], para_dict['embedding_id_size']], dtype=tf.float32)
        self.uidW = tf.get_variable("uidW", [para_dict['user_num'], para_dict['embedding_id_size']], dtype=tf.float32)

        self.embeded_uid = tf.nn.embedding_lookup(self.uidW, self.input_uid)
        self.embeded_iid = tf.nn.embedding_lookup(self.iidW, self.input_iid)
        self.r = self.rating_prediction(para_dict)
        self.review_prediction, self.hcL = self.review_frequency_prediction(para_dict)
        self.h = self.get_s0(para_dict)

        gru_cell = tf.contrib.rnn.GRUCell(num_units=para_dict['hidden_dimension'], name = 'gru_cell')
        self.captions_in = self.input_y_tip[:, :-1]
        self.captions_out = self.input_y_tip[:, 1:]
        self.mask = tf.to_float(tf.not_equal(self.captions_out, para_dict['tip_vocab_size'] - 1))
        self.x = tf.nn.embedding_lookup(self.W_tip, self.captions_in, name='word_vector')
        self.batch_size = tf.shape(self.input_y_tip)[0]
        loss = 0.0
        with tf.variable_scope('sentence_generate'):
            for t in range(para_dict['max_tip_len'] - 1):
                if t > 0: tf.get_variable_scope().reuse_variables()
                _, self.h = gru_cell(self.x[:,t,:], state=self.h)
                Ws = tf.get_variable('Ws', [para_dict['hidden_dimension'], para_dict['tip_vocab_size']])
                Bs = tf.get_variable('Bs', [para_dict['tip_vocab_size']])
                logits = tf.matmul(self.h, Ws) + Bs
                loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.captions_out[:, t], logits=logits) * self.mask[:, t])

        with tf.variable_scope('loss'):
            tip_loss = loss / tf.to_float(self.batch_size)
            r_loss = tf.reduce_mean(tf.square(self.input_y - self.r))
            self.input_y_review = self.review_preprocess(para_dict)
            review_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y_review, logits = self.review_prediction))
            print(tip_loss, r_loss, review_loss)
            self.loss = tip_loss + r_loss + review_loss

        clipper = 50
        self.learning_rate = tf.Variable(para_dict['learning_rate'], trainable=False, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate )
        tvars = tf.trainable_variables()
        if para_dict['lambda_l2'] > 0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + para_dict['lambda_l2'] * l2_loss
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        tf.get_variable_scope().reuse_variables()
        self.sample_words = self.generate_sentence(para_dict)
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.input_y - self.r)))
        self.mae = tf.reduce_mean(tf.abs(self.input_y - self.r))


    def review_preprocess(self, para_dict):
        batch_nums = tf.range(0, limit = para_dict['batch_size'])  # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, axis = 1)  # shape (batch_size, 1)
        batch_nums = tf.tile(batch_nums, [1, para_dict['review_len']])  # shape (batch_size, review_len)
        indices = tf.stack((batch_nums, self.review_node_idxs), axis = 2)  # shape (batch_size, review_len, 2)
        indices = tf.reshape(indices, [-1, 2])  # [batch_size * review_len, 2]

        attn_dist = tf.reshape(self.review_attn_dist, shape = [-1])  # [batch_size * review_len]

        word_frequency = tf.sparse_to_dense(sparse_indices = indices, sparse_values = attn_dist, output_shape = [para_dict['batch_size'], para_dict['review_vocab_size']], validate_indices=False)  # [batch_size, vsize]
        return word_frequency

    def rating_prediction(self, para_dict):
        with tf.variable_scope("rating_prediction"):
            Wu = tf.get_variable('Wru', [para_dict['embedding_id_size'], para_dict['n_latent']])
            u_feas = tf.matmul(tf.squeeze(self.embeded_uid), Wu)

            Wi = tf.get_variable('Wri', [para_dict['embedding_id_size'], para_dict['n_latent']])
            i_feas = tf.matmul(tf.squeeze(self.embeded_iid), Wi)

            Br = tf.get_variable('Wrb', [para_dict['n_latent']])
            hr = tf.sigmoid(u_feas + i_feas + Br)

            for i in range(0, 4):
                Wr = tf.get_variable('Wr' + str(i), [para_dict['n_latent'], para_dict['n_latent']])
                Br = tf.get_variable('Wb'+ str(i), [para_dict['n_latent']]) #name：新变量或现有变量的名称，这个参数是必须的，函数会根据变量名称去创建或者获取变量
                hrL = tf.sigmoid(tf.matmul(hr, Wr) + Br)

            Wrr = tf.get_variable('Wrr', [para_dict['n_latent'], 1])
            Brr = tf.get_variable('Wbr', [1])
            hr = tf.matmul(hrL, Wrr) + Brr
            return hr


    def review_frequency_prediction(self, para_dict):
        with tf.variable_scope('review_frequency_prediction'):
            Wu = tf.get_variable('Wcu', [para_dict['embedding_id_size'], para_dict['n_latent']])
            u_feas = tf.matmul(tf.squeeze(self.embeded_uid), Wu)

            Wi = tf.get_variable('Wci', [para_dict['embedding_id_size'], para_dict['n_latent']])
            i_feas = tf.matmul(tf.squeeze(self.embeded_iid), Wi)

            Br = tf.get_variable('Wcb', [para_dict['n_latent']])
            hc = tf.sigmoid(u_feas + i_feas + Br)

            for i in range(0, 4):
                Wc = tf.get_variable('Wc' + str(i), [para_dict['n_latent'], para_dict['n_latent']])
                Bc = tf.get_variable('Wbc' + str(i), [para_dict['n_latent']])
                hcL = tf.sigmoid(tf.matmul(hc, Wc) + Bc)

            Wcc = tf.get_variable('Wcc', [para_dict['n_latent'], para_dict['review_vocab_size']])
            Bcc = tf.get_variable('Wbc', [para_dict['review_vocab_size']])
            hc = tf.matmul(hcL, Wcc) + Bcc
            return hc, hcL


    def get_s0(self, para_dict):
        Wu = tf.get_variable('Wu', [para_dict['embedding_id_size'], para_dict['hidden_dimension']])
        u_feas = tf.matmul(tf.squeeze(self.embeded_uid), Wu)

        Wi = tf.get_variable('Wi', [para_dict['embedding_id_size'], para_dict['hidden_dimension']])
        i_feas = tf.matmul(tf.squeeze(self.embeded_iid), Wi)

        r_one_hot = tf.reshape(tf.one_hot(tf.to_int64(self.r), 6), [-1, 6])
        Wr = tf.get_variable('Wr', [6, para_dict['hidden_dimension']])
        r_feas = tf.matmul(r_one_hot, Wr)

        Whc = tf.get_variable('Whc', [para_dict['n_latent'], para_dict['hidden_dimension']])
        hc_feas = tf.matmul(self.hcL, Whc)
        b_feas = tf.get_variable('b_feas', [para_dict['hidden_dimension']])
        feas = tf.nn.tanh(u_feas + i_feas + r_feas + hc_feas + b_feas)
        return feas




    def generate_sentence(self, para_dict):
        gru_cell = tf.contrib.rnn.GRUCell(num_units=para_dict['hidden_dimension'], name='gru_cell')
        sampled_words = []
        with tf.variable_scope('sentence_generate'):
            for t in range(para_dict['max_tip_len'] - 1):
                if t > 0: tf.get_variable_scope().reuse_variables()
                _, self.h = gru_cell(self.x[:, t, :], state=self.h)
                Ws = tf.get_variable('Ws', [para_dict['hidden_dimension'], para_dict['tip_vocab_size']])
                Bs = tf.get_variable('Bs', [para_dict['tip_vocab_size']])
                logits = tf.matmul(self.h, Ws) + Bs
                wordidx_t = tf.argmax(logits, 1)  # [batch_size, 1]
                wordidx_t = tf.reshape(wordidx_t, [-1])  # [batch_size]
                sampled_words.append(wordidx_t)
            sampled_words = tf.stack(sampled_words, axis=1)
            return sampled_words








