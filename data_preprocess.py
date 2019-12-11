import cv2
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def restore_images_cifar10_train():
    if not os.path.exists(os.path.join(root_dir, 'train')):
        os.mkdir(os.path.join(root_dir, 'train'))
    for i in range(1, 6):
        batch= unpickle(root_dir+'data_batch_'+str(i))
        print(batch)
        image_arrays = batch[b'data']
        labels = batch[b'labels']
        filenames = batch[b'filenames']
        for index in range(len(labels)):
            image_array = image_arrays[index]
            red = image_array[:1024]
            green = image_array[1024:2048]
            blue = image_array[2048:]
            red_image = red.reshape(32, 32)
            green_image = green.reshape(32, 32)
            blue_image = blue.reshape(32, 32)
            merged = cv2.merge([blue_image, green_image, red_image])

            label_name = label_names[labels[index]].decode()
            filename = filenames[index].decode()
            if not os.path.exists(os.path.join(root_dir+'train/', label_name)):
                os.mkdir(os.path.join(root_dir+'train/', label_name))
            cv2.imwrite(os.path.join(root_dir+'train/', label_name, filename), merged)


def restore_image_cifar10_val():
    if not os.path.exists(os.path.join(root_dir, 'val')):
        os.mkdir(os.path.join(root_dir, 'val'))
    batch = unpickle(root_dir+'test_batch')
    image_arrays = batch[b'data']
    labels = batch[b'labels']
    filenames = batch[b'filenames']
    for index in range(len(labels)):
        image_array = image_arrays[index]
        red = image_array[:1024]
        green = image_array[1024:2048]
        blue = image_array[2048:]
        red_image = red.reshape(32, 32)
        green_image = green.reshape(32, 32)
        blue_image = blue.reshape(32, 32)
        merged = cv2.merge([blue_image, green_image, red_image])

        label_name = label_names[labels[index]].decode()
        filename = filenames[index].decode()
        if not os.path.exists(os.path.join(root_dir+'val/', label_name)):
            os.mkdir(os.path.join(root_dir+'val/', label_name))
        cv2.imwrite(os.path.join(root_dir+'val/', label_name, filename), merged)


def check_label_names():
    d = unpickle(root_dir+'batches.meta')
    label_names = d[b'label_names']
    return label_names


root_dir = 'dataset/cifar10/'
label_names = check_label_names()


if __name__ == '__main__':
    restore_images_cifar10_train()