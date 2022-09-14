def build_mode(features, mode, params):
  net = fc.input_layer(features, params['feature_columns'])
  # Build the hidden layers, sized according to the 'hidden_units' param.
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
      net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
  # Compute logits
  logits = tf.layers.dense(net, 1, activation=None)
  return logits

def my_model(features, labels, mode, params):
  with tf.variable_scope('ctr_model'):
    ctr_logits = build_mode(features, mode, params)
  with tf.variable_scope('cvr_model'):
    cvr_logits = build_mode(features, mode, params)

  ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
  cvr_predictions = tf.sigmoid(cvr_logits, name="CVR")
  prop = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'probabilities': prop,
      'ctr_probabilities': ctr_predictions,
      'cvr_probabilities': cvr_predictions
    }
    export_outputs = {
      'prediction': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

  y = labels['cvr']
  cvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y, prop), name="cvr_loss")
  ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['ctr'], logits=ctr_logits), name="ctr_loss")
  loss = tf.add(ctr_loss, cvr_loss, name="ctcvr_loss")

  ctr_accuracy = tf.metrics.accuracy(labels=labels['ctr'], predictions=tf.to_float(tf.greater_equal(ctr_predictions, 0.5)))
  cvr_accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prop, 0.5)))
  ctr_auc = tf.metrics.auc(labels['ctr'], ctr_predictions)
  cvr_auc = tf.metrics.auc(y, prop)
  metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
  tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
  tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
  tf.summary.scalar('ctr_auc', ctr_auc[1])
  tf.summary.scalar('cvr_auc', cvr_auc[1])
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN
  optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)