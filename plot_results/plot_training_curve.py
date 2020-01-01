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


def plot_acc():
    epoches = []
    train_acc = []
    test_acc = []
    with open('../log/n0/train_epoch_log_formal_1229_1_n0') as f:
        for line in f:
            line = line.split()
            epoches.append(int(line[0]))
            acc = line[2].split('(')[1].split(')')[0]
            train_acc.append(float(acc))
    with open('../log/n0/test_log_formal_1229_1_n0') as f:
        for line in f:
            line = line.split()
            acc = line[2].split('(')[1].split(')')[0]
            test_acc.append(float(acc))
    plt.plot(epoches[:30], train_acc[:30], label='train')
    plt.plot(epoches[:30], test_acc[:30], label='test')
    plt.legend()
    plt.title('Training/Testing Accuracy w.r.t Epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('figures/acc_train_test.png')
    plt.show()


def plot_time():
    epoches = []
    times = []
    with open('../log/n0/epoch_time_formal_1229_1_n0') as f:
        for line in f:
            line = line.split()
            epoches.append(int(line[0]))
            times.append(float(line[1])/60)
    average = sum(times[:30])/30
    plt.plot(epoches[:30], times[:30], label='Actual Time')
    plt.plot(epoches[:30], [average]*30, label='average={:.2f}'.format(average))
    plt.legend()
    plt.title('Actual Time per Epoch')
    plt.xlabel('epoches')
    plt.ylabel('Time(mintues)')
    plt.savefig('figures/training_time.png')
    plt.show()




if __name__ == '__main__':
    # plot('../log/lab/n0/train_log_0', 'figures/test.png')
    # plot_exp_train_layers()
    plot_acc()
    plot_time()