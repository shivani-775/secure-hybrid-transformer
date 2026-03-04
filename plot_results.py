import matplotlib.pyplot as plt


def plot_accuracy(normal_acc, attack_acc, secure_acc):

    scenarios = ["Normal", "Under Attack", "With AES"]
    accuracies = [normal_acc * 100, attack_acc * 100, secure_acc * 100]

    plt.figure()

    plt.bar(scenarios, accuracies)

    plt.xlabel("Scenario")
    plt.ylabel("Accuracy (%)")
    plt.title("Impact of Cyber Attack and AES Protection on Model Accuracy")

    plt.ylim(0, 100)

    plt.show()