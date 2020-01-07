def sign(x):
    return 1-(x <= 0)


class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        self._weights = {}
        for _ in range(iterations):
            for example, label in examples:
                self.update_weight(example, label)

    def update_weight(self, data_point, label):
        predicted_label = 0
        for feature, value in data_point.items():
            if not feature in self._weights:
                self._weights[feature] = 0

            weight = self._weights[feature]
            predicted_label += weight*value

        if sign(predicted_label) != label:
            if label > 0:
                for feature, value in data_point.items():
                    self._weights[feature] += value
            else:
                for feature, value in data_point.items():
                    self._weights[feature] -= value

    def predict(self, x):
        prediction = 0
        for feature, value in x.items():
            if not feature in self._weights:
                self._weights[feature] = 0

            w = self._weights[feature]
            prediction += w * value

        return sign(prediction)


class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        self.labels_weights = {label: {} for (data_point, label) in examples}

        for __ in range(iterations):
            for data_point, label in examples:
                predicted_label = None
                current_max = -1

                for weight_label, weights in self.labels_weights.items():
                    label_sum = 0

                    for data_point_label, data_point_value in data_point.items():

                        if not data_point_label in weights:
                            weights[data_point_label] = 0

                        label_sum += data_point_value * \
                            weights[data_point_label]

                    if label_sum > current_max:
                        current_max = label_sum
                        predicted_label = weight_label

                if predicted_label != label:
                    correct_weights = self.labels_weights[label]
                    for data_point_label, data_point_value in data_point.items():

                        if data_point_label in correct_weights:
                            correct_weights[data_point_label] += data_point_value
                        else:
                            correct_weights[data_point_label] = data_point_value

                    predicted_weights = self.labels_weights[label]
                    for data_point_label, data_point_value in data_point.items():

                        if data_point_label in correct_weights:
                            predicted_weights[data_point_label] += data_point_value
                        else:
                            predicted_weights[data_point_label] = data_point_value

    def predict(self, x):
        predicted_label = None
        current_max = -1

        for label_weigth in self.labels_weights.keys():
            label_sum = 0

            weights = self.labels_weights.get(label_weigth)
            for data_point_label, data_point_value in x.items():
                weight = weights[data_point_label]
                if weight is not None:
                    label_sum += data_point_value * weight

            if label_sum > current_max:
                current_max = label_sum
                predicted_label = label_weigth

        return predicted_label
