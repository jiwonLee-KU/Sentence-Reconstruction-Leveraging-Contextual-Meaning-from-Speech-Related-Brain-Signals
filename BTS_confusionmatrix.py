import seaborn
import matplotlib.pyplot as plt


def plot_confusion_matrix(data, labels, output_filename):
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(18*3, 12*3))

    plt.title("Words / Confusion Matrix: Imagined Speech")

    seaborn.set(font_scale=2)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
    ax.set_xticklabels(labels=labels, fontsize = 15)
    ax.set_yticklabels(labels=labels, fontsize = 15)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches='tight', dpi=100)
    plt.close()

def plot_cosine_similarities_matrix(data, labels, output_filename):

    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(18*3, 12*3))

    plt.title("Cosine Similarities Matrix")

    seaborn.set(font_scale=2)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
    ax.set_xticklabels(labels=labels, fontsize = 15)
    ax.set_yticklabels(labels=labels, fontsize = 15)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches='tight', dpi=100)
    plt.close()