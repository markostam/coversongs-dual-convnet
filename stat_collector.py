import tensorflow as tf

class StatisticsCollector(object):
    '''
    object based on: https://github.com/brilee/MuGo/blob/master/policy.py#L174
    _______________________________________________________________
    Accuracy and cost cannot be calculated with the full test dataset
    in one pass, so they must be computed in batches. Unfortunately,
    the built-in TF summary nodes cannot be told to aggregate multiple
    executions. Therefore, we aggregate the accuracy/cost ourselves at
    the python level, and push it through the accuracy/cost summary
    nodes to generate the appropriate summary protobufs for writing.
    '''
    with tf.device("/cpu:0"):
        accuracy = tf.placeholder(tf.float32, [])
        cost = tf.placeholder(tf.float32, [])
        accuracy_summary = tf.scalar_summary("accuracy", accuracy)
        cost_summary = tf.scalar_summary("log_likelihood_cost", cost)
        accuracy_summaries = tf.merge_summary([accuracy_summary, cost_summary], name="accuracy_summaries")
    session = tf.Session()

    def __init__(self):
        self.accuracies = []
        self.costs = []

    def collect(self, accuracy, cost):
        self.accuracies.append(accuracy)
        self.costs.append(cost)

    def report(self):
        avg_acc = sum(self.accuracies) / len(self.accuracies)
        batch_cost = sum(self.costs) / len(self.costs)
        self.accuracies = []
        self.costs = [] 
        summary = self.session.run(self.accuracy_summaries,
            feed_dict={self.accuracy:avg_acc, self.cost: batch_cost})
        return avg_acc, batch_cost, summary
