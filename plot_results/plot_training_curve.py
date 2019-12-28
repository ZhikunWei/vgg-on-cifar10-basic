import matplotlib.pyplot as plt


def plot(filename, figurename):
    batch, loss, acc = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(' ')
            batch.append(int(line[0]))
            loss.append(float(line[1]))
            acc.append(float(line[2]))
    plt.figure(0)
    plt.plot(batch, loss)
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.savefig(figurename)
    plt.show()


def plot_exp_train_layers():
    batches, losses , acces = [[], [], [], []], [[], [], [], []], [[], [], [], []]
    print(batches, losses)
    figure1 = plt.figure(0)
    labels = ['train last layer', 'train last two layers', 'train last three layers', 'train all layers']

    for i in range(1, 5):
        with open('../log/lab/n0/train_log_layer_exp_'+str(i), 'r') as f:
            for line in f:
                line = line.split()
                if len(batches[i-1]) >= 400:
                    break
                if int(line[0]) == 0:
                    for j in range(4):
                        batches[j].clear()
                        losses[j].clear()
                        acces[j].clear()
                batches[i-1].append(int(line[0]))
                losses[i-1].append(float(line[1]))
                acces[i-1].append(float(line[2]))
        plt.plot(batches[i - 1], losses[i - 1], label=labels[i-1])
        # plt.plot(batches[i-1], acces[i-1], label=labels[i-1])
        # plt.title('train accuracy')
        plt.title('loss')
        plt.xlabel('batch')
        plt.legend()
    # plt.savefig('figures/acc_diff_layers.png')
    plt.savefig('figures/lass_diff_layers.png')
    plt.show()


if __name__ == '__main__':
    # plot('../log/lab/n0/train_log_0', 'figures/test.png')
    plot_exp_train_layers()