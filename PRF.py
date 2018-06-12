#实现查准率、召回率与F1的计算。
def tf_confusion_metrics(model, actual_classes, session, feed_dict):
    predictions = tf.argmax(model, 1)
    actuals = tf.argmax(actual_classes, 1)

#初始化，将所有的矩阵都初始化为0矩阵或者是单位矩阵
    ones_like_actuals = tf.ones_like(actuals)     #ones_like: create a tensor and set all to 1
    zeros_like_actuals = tf.zeros_like(actuals)    #zeros_like: create a tensor and set all to 0
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
#True positive
    tp_op = tf.reduce_sum(
        tf.cast(                #字符类型转换。
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"             #定义浮点型数据。
        )
    )
#True negative
    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals,zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )
#False negative
    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    tp, tn, fp, fn = \
    session.run(
        [tp_op, tn_op, fp_op, fn_op],
        feed_dict
    )

    tpr = float(tp)/(float(tp)+float(fn))
    fpr = float(fp)/(float(tp)+float(fn))
#准确率
    accuracy = (float(tp)+float(tn))/(float(tp)+float(fp)+float(fn)+float(tn))
#召回率
    recall = tpr
    precision = float(tp)/(float(tn)+float(fp))
#F1_score
    f1_score = (2 * (precision * recall)) / (precision + recall)

    print('Precision = ',precision)
    print('Recall = ',recall)
    print('F1_score = ',f1_score)
    print('Accuracy = ', accuracy)