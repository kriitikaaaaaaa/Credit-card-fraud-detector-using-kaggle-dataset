import matplotlib.pyplot as plt
import numpy as np

# Data from the image
model_names = ['KNN', 'Naive Bayes', 'CNN']

# Taking the values for %30 training %70 test scenario

accuracies = [0.99, 0.95, 0.97]
precisions = [0.99, 0.85, 0.90]
recalls = [0.96, 0.92, 0.94]
f1s = [0.97, 0.88, 0.92]

# Function to plot performance metrics
def plot_performance_metrics(model_names, accuracies, precisions, recalls, f1s):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    if len(model_names) != 3 or len(accuracies) != 3 or len(precisions) != 3 or len(recalls) != 3 or len(f1s) != 3:
        raise ValueError("There should be exactly three models with four metrics each.")

    ind = np.arange(len(metrics))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 9))

    rects1 = ax.bar(ind - width, [accuracies[0], precisions[0], recalls[0], f1s[0]], width, label=model_names[0])
    rects2 = ax.bar(ind, [accuracies[1], precisions[1], recalls[1], f1s[1]], width, label=model_names[1])
    rects3 = ax.bar(ind + width, [accuracies[2], precisions[2], recalls[2], f1s[2]], width, label=model_names[2])

    ax.set_xlabel('Performance Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Model Performance')
    ax.set_xticks(ind)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend()
    # Move legend to the bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.show()

plot_performance_metrics(model_names, accuracies, precisions, recalls, f1s)