from perceptrons import MulticlassPerceptron
from perceptrons import BinaryPerceptron


def parse_data(data):
    parsed_data = []
    for data_points, label in data:
        data_point = {}
        for i, value in enumerate(data_points, 1):
            data_point[i] = value

        parsed_data.append((data_point, label))

    return parsed_data


def format_predict(instance):
    return {i+1: instance[i] for i, x in enumerate(instance, 0)}


class IrisClassifier(object):

    def __init__(self, data):
        self.classifier = MulticlassPerceptron(parse_data(data), 10)

    def classify(self, instance):
        return self.classifier.predict(format_predict(instance))


class DigitClassifier(object):

    def __init__(self, data):

        self.classifier = MulticlassPerceptron(
            parse_data(data), 3)

    def classify(self, instance):
        return self.classifier.predict(format_predict(instance))


class BiasClassifier(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron(
            [({1: x, 2: 1}, y) for x, y in data], 10)

    def classify(self, instance):
        return self.classifier.predict({1: instance, 2: 1})


class MysteryClassifier1(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron(
            [({1: x[0]**2 + x[1]**2, 2: 1}, y) for x, y in data], 10)

    def classify(self, instance):
        return self.classifier.predict({1: instance[0]**2 + instance[1]**2, 2: 1})


class MysteryClassifier2(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron(
            [({1: x[0] * x[1] * x[2]}, y) for x, y in data], 10)

    def classify(self, instance):
        return self.classifier.predict({1: instance[0] * instance[1] * instance[2]})
