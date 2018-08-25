import torch
from torchvision import models
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
from skimage import morphology
from skimage.color import gray2rgb
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import pickle
import os
from datetime import datetime
import pandas as pd
from IPython import display


def main(
    image_path="data/image_samples/cat.jpg",
    kernel_size=3,
    nblocks=3,
    nlayers=3,
    epochs=500,
    lr=0.01,
    l1_coeff=1.0,
    class_index="max",
    init_pert_model=False,
    save_to_dir=".img_pert_temp",
    datetime_subdir=True,
):
    """
    :param image_path: (string) path to image to perturbate.
    :param kernel_size: (int) side of convolution kernel.
    :param nblocks: (int) number of conv. layer blocks.
    Block followed by relu.
    :param nlayers: (int) number of conv. layers in each block.
    :param epochs: (int) number of epochs to train the perturbation.
    :param lr: (float) learning rate.
    :param l1_coeff: (float) weight of the loss term proportional
    to the mean of the absolute difference between the original
    and perturbated images.
    :param class_index: (int, int list, or "max") the (vgg)
    model output(s) to be minimized. "max" minimizes the 5
    maximum classes in the original image.
    :param init_pert_model: (Bool) whether to initialize the
    perturbations generator so as to minimize the initial
    perturbations on the original image. Can't be perfect
    because vgg pre-processing generates inputs in [-0.5, 0.5]
    and the relu layers (in pert_model) zero the
    negative values.
    :param save_to_dir: (string) directory to save the:
    original, perturbated, and differences images, the
    perturbations generator model, the loss history, and a
    csv with the vgg classes, original class scores, and
    perturbated class scores.
    :param datetime_subdir: (Bool) whether create a
    subdirectory within save_to_dir named as the current
    datetime. This is ignored if save_to_dir is the default
    temp directory.
    :return: None
    """
    vgg_model = load_model()
    img = load_input(image_path)
    img = resize_img(img)
    img_tensor = img_to_tensor(img, False)
    # img_tensor = image_to_vgg_input_tensor(img)
    # input_assessment(img_tensor, vgg_model)
    pert_model = PerturbationsGenerator(
        kernel_size, nblocks, nlayers,
    )
    pert_img_tensor, pert_model, losses = get_optimum_perturbation(
        epochs, pert_model, img_tensor,
        vgg_model=vgg_model,
        lr=lr, l1_coeff=l1_coeff,
        class_index=class_index,
        init_pert_model=init_pert_model,
    )

    print_model_evaluations(img_tensor, pert_img_tensor)
    pert_img = tensor_to_img(pert_img_tensor)
    diff = calculate_differences(img, pert_img)
    folder = save_results(
        img, pert_img, diff,
        pert_model=pert_model,
        loss_history=losses,
        save_to_dir=save_to_dir,
        datetime_subdir=datetime_subdir,
    )
    plot_results(folder)

    # diff, proc_img_np, pert_img_np = post_processing(
    #     img_tensor, pert_img_tensor,
    # )
    # plot_results(
    #     proc_img_np, pert_img_np, diff,
    #     indicate_center_of_mass=indicate_center_of_mass,
    # )
    #
    # return proc_img_np, pert_img_np, diff


class PerturbationsGenerator(torch.nn.Module):
    def __init__(self, kernel_size=3, nblocks=3, nlayers=3):
        super(PerturbationsGenerator, self).__init__()
        # build conv layers, implement padding='same':
        if np.mod(kernel_size, 2) == 0: kernel_size += 1
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(
            3, 3, kernel_size = kernel_size,
            padding = padding,
        )
        self.relu = torch.nn.ReLU()
        self.nblocks = nblocks
        self.nlayers = nlayers
        self.padding = padding
        self.kernel_size = kernel_size
        self.net = self.make_layers(
            nblocks, nlayers, kernel_size, padding,
        )

        if use_cuda(): self.cuda()

    def forward(self, x):
        # gather information for scaling
        xmin = torch.min(x)
        Dx = torch.max(x - xmin)

        # perturbate the image:
        x = self.net(x)
        # for __ in range(self.nblocks):
        #     for __ in range(self.nlayers):
        #         x = self.conv(x)
        #     x = self.relu(x)

        # scale to original input range:
        x = x.add(- torch.min(x))  # x: zero to something
        x = x.div(torch.max(x))  # x: zero to 1
        x = x.mul(Dx)  # x: zero to Dx
        x = x.add(xmin)  # x: xmin to xmin + Dx

        if use_cuda(): x.cuda()

        return x

    def make_layers(self, nblocks, nlayers,
                    kernel_size, padding):
        layers = []

        for __ in range(nblocks):
            for __ in range(nlayers):
                conv = torch.nn.Conv2d(
                    3, 3, kernel_size=kernel_size,
                    padding=padding,
                )
                layers.append(conv)

            layers.append(torch.nn.ReLU())

        return torch.nn.Sequential(*layers)

    def initialize_conv_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # initialize to pass the input unchanged:
                torch.nn.init.dirac_(m.weight)

                if m.bias is not None: # conv may be defined without bias (see above)
                        torch.nn.init.constant_(m.bias, 0.)


def get_optimum_perturbation(# todo: fix for categories
        epochs, pert_model, img, vgg_model,
        lr=0.1, l1_coeff=0.01,
        class_index="max",
        init_pert_model=False,
        figname="loss",
):
    # optimizer = torch.optim.RMSprop(
    #     pert_model.parameters(), lr=lr, momentum=0.9,
    # )
    # optimizer = torch.optim.SGD(
    #     pert_model.parameters(), lr=lr, #momentum=0.9,
    # )
    optimizer = torch.optim.Adam(
        pert_model.parameters(), lr=lr
    )

    if init_pert_model:
        pert_model.initialize_conv_weights() # init fiters to delta functions

    if class_index is "max":
        target = torch.nn.Softmax()(vgg_model(img))
        categories = np.argsort(target.cpu().data.numpy()).ravel()[::-1]
        categories = list(categories[:5])
        # category = np.argmax(target.cpu().data.numpy())
        print "Category with highest probability", categories
    else: # scalar or list of ints
        categories = list(np.ravel([class_index]))
        print "minimizing class with index = {}".format(class_index)

    print "Optimizing.. "
    losses = []

    for i in tqdm(range(epochs)):
        pert_img = pert_model(img)
        outputs = torch.nn.Softmax()(vgg_model(pert_img))
        img_diff = img - pert_img
        l1_term = l1_coeff * torch.mean(torch.abs(torch.pow(img_diff, 1)))
        sc_term = torch.sum(outputs[0, categories])
        loss = l1_term + sc_term
        # loss = l1_term + outputs[0, category]

        # live plot:
        losses.append(loss.data[0])
        plot_loss(losses, figname)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print "original score: {}".format(torch.nn.Softmax()(vgg_model(img))[0, categories])
    print "perturbed score: {}".format(torch.nn.Softmax()(vgg_model(pert_img))[0, categories])

    return pert_img, pert_model, losses


def load_input(image_path, graph=False):
    img = io.imread(image_path)

    if graph:
        plt.figure("original image")
        plt.imshow(img)

    return img


def resize_img(img):
    return transform.resize(img, (224, 224))


def load_model():
    model = models.vgg19(pretrained=True)
    model.eval()

    if use_cuda():
        model.cuda()

    for p in model.features.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
            p.requires_grad = False

    return model


def img_to_tensor(img,
                  requires_grad=False):
    # preprocessed_img = transform.resize(img, (224, 224))
    # preprocessed_img = np.float32(preprocessed_img.copy())
    preprocessed_img = np.float32(np.copy(img))
    preprocessed_img = preprocessed_img[:, :, ::-1]

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    for i in range(3):
        preprocessed_img[:, :, i] = \
            preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = \
            preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = np.ascontiguousarray(
        np.transpose(preprocessed_img, (2, 0, 1))
    )

    if use_cuda():
        preprocessed_img_tensor =\
            torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor =\
            torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    preprocessed_img_tensor = Variable(
        preprocessed_img_tensor,
        requires_grad = requires_grad
    )

    return preprocessed_img_tensor


# def image_to_vgg_input_tensor(img):
#     preprocessed_img = transform.resize(img, (224, 224))
#     preprocessed_img = np.float32(preprocessed_img.copy())
#     preprocessed_img = preprocessed_img[:, :, ::-1]
#
#     means=[0.485, 0.456, 0.406]
#     stds=[0.229, 0.224, 0.225]
#
#     for i in range(3):
#         preprocessed_img[:, :, i] =\
#             preprocessed_img[:, :, i] - means[i]
#         preprocessed_img[:, :, i] =\
#             preprocessed_img[:, :, i] / stds[i]
#
#     preprocessed_img = np.ascontiguousarray(
#         np.transpose(preprocessed_img, (2, 0, 1))
#     )
#
#     if use_cuda():
#         preprocessed_img_tensor =\
#             torch.from_numpy(preprocessed_img).cuda()
#     else:
#         preprocessed_img_tensor =\
#             torch.from_numpy(preprocessed_img)
#
#     preprocessed_img_tensor.unsqueeze_(0)
#
#     return Variable(preprocessed_img_tensor, requires_grad = False)


def print_top_5(x):
    y = evaluate_model(x)
    top_5 = get_top_classes(y, 5)
    print "5 top classes identified by the model:"
    print "(class index) class description: model score"

    for index, name, score in top_5:
        print "({}) {}: {}".format(index, name, score)

    print



# def input_assessment(input_tensor, vgg_model):
#     with open("data/imagenet1000_clsid_to_human.pkl", "r") as fp:
#         vgg_class = pickle.load(fp)
#
#     outputs = torch.nn.Softmax()(vgg_model(input_tensor))
#     outputs_np = outputs.data.cpu().numpy()
#     sorted_args = np.argsort(outputs_np[0, :])[::-1]
#
#     print "5 top classes identified by the model:"
#     print "(class index) class description: model score"
#
#     for index in sorted_args[:5]:
#         print "({}) {}: {}".format(index, vgg_class[index], outputs[0, index])
#
#     print
#
#     if outputs_np[0, sorted_args[0]] < 0.5:
#         print "*** Warning ***"
#         print "top category score under 0.5, extracted explanation may not be accurate on not well defined class"
#         print


def evaluate_model(
        x, model=None,
        tensor_output=False,
):
    if model is None: model = load_model()

    if 'torch' not in str(type(x)):
        x_tensor = img_to_tensor(np.array(x))
    else:
        x_tensor = x

    outputs = torch.nn.Softmax()(model(x_tensor))

    if not tensor_output:
        outputs = outputs.data.cpu().numpy()

    return outputs


def get_top_classes(y, n=5):
    vgg_class = get_vgg_classes()

    if 'torch' in str(type(y)):
        ynp = y.data.cpu().numpy()
    else:
        ynp = np.copy(y)

    ynp = ynp.ravel()

    sorted_args = np.argsort(ynp)[::-1]
    sorted_args = sorted_args[:n]
    top_n = zip(sorted_args, vgg_class[sorted_args], ynp[sorted_args])

    return top_n

def get_vgg_classes():
    fname = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(fname, "imagenet1000_clsid_to_human.pkl")

    with open(fname, "r") as fp:
        vgg_class_dir = pickle.load(fp)

    vgg_class = []

    for key, value in vgg_class_dir.iteritems():
        vgg_class.append(value)

    return np.array(vgg_class)


def use_cuda():
    return torch.cuda.is_available()


def tensor_to_img(tensor):
    img = tensor.data.cpu().numpy()[0]
    img = np.transpose(img, (1, 2, 0))

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    means = means[np.newaxis, np.newaxis, :]
    stds = stds[np.newaxis, np.newaxis, :]

    img *= stds
    img += means

    img = img[:, :, ::-1]

    return img


# def post_processing(proc_img_tensor, pert_img_tensor):
#     proc_img_np = tensor_to_img(proc_img_tensor)
#     pert_img_np = tensor_to_img(pert_img_tensor)
#
#     # mean over image channels:
#     proc = np.mean(proc_img_np, axis=2)
#     pert = np.mean(pert_img_np, axis=2)
#
#     # highlighting the differences:
#     diff = (proc - pert) ** 6
#
#     # remove the edges: artifacts due to padding may appear.
#     h, w = np.shape(diff)
#     diff[:int(0.1 * h), :] = 0
#     diff[int(0.9 * h):, :] = 0
#     diff[:, :int(0.1 * w)] = 0
#     diff[:, int(0.9 * w):] = 0
#
#     # dilate the important points left for visibility:
#     square = np.ones((20, 20))
#     diff = morphology.dilation(diff, square)
#
#     return diff, proc_img_np, pert_img_np


def calculate_differences(img, pert):
    diff = np.abs(pert - img)
    return diff


# def calculate_differences(img, pert):
#     # mean over image channels:
#     proc = np.mean(img, axis=2).astype(np.float32)
#     pert = np.mean(pert, axis=2).astype(np.float32)
#
#     # highlighting the differences:
#     diff = (proc - pert) ** 6
#
#     # remove the edges: artifacts due to padding may appear.
#     h, w = np.shape(diff)
#     diff[:int(0.1 * h), :] = 0
#     diff[int(0.9 * h):, :] = 0
#     diff[:, :int(0.1 * w)] = 0
#     diff[:, int(0.9 * w):] = 0
#
#     # dilate the important points left for visibility:
#     square = np.ones((20, 20))
#     diff = morphology.dilation(diff, square)
#     return diff


def save_results(
        img=None, pert=None, diff=None,
        pert_model=None,
        loss_history=None,
        save_to_dir=".img_pert_temp",
        datetime_subdir=True,
):
    folder = save_to_dir
    if (save_to_dir != ".img_pert_temp") and datetime_subdir:
        folder = os.path.join(save_to_dir, str(datetime.now()))

    if not os.path.isdir(folder): os.makedirs(folder)

    print "results saved to:"

    # save images:
    names = [
        "original",
        "perturbated",
        "differences"
    ]

    values = [img, pert, diff]

    for name, value in zip(names, values):
        if value is not None:
            fname = os.path.join(folder, name + ".jpg")
            io.imsave(fname, value)
            print "{} image saved to {}".format(name, fname)
        else:
            print "{} image not saved (None given)".format(name)

    # save the scores:
    if (img is not None) and (pert is not None):
        scores_df = pd.DataFrame()
        scores_df["imagenet_class"] = get_vgg_classes()
        scores_df["original_score"] = np.ravel(evaluate_model(img))
        scores_df["perturbated_score"] = np.ravel(evaluate_model(pert))
        fname = os.path.join(folder, "scores.csv")
        scores_df.to_csv(fname)
        print "scores saved to: {}".format(fname)
    else:
        print "scores not saved, need to input both original and perturbated images"

    # save top 5's:
    # y = evaluate_model(img)
    # or_top5 = get_top_classes(y, 5)
    # y = evaluate_model(pert)
    # pert_top5 = get_top_classes(y, 5)
    # idx = pd.MultiIndex.from_product(
    #     [['or', 'pert'], ['index', 'class', 'score']]
    # )
    # top_ser = pd.Series(or_top5 + pert_top5, index=idx)
    # fname = os.path.join(folder, "top_5.csv")
    # top_ser.to_csv(fname)

    # save pert model:
    if pert_model is not None:
        file_path = os.path.join(folder, "pert_model.pt")
        torch.save(pert_model, file_path)
        print "perturbation generator model saved to: {}".format(file_path)
        print
    else:
        print "model not saved (None given)"

    # save loss history:
    if loss_history is not None:
        file_path = os.path.join(folder, "loss_history")
        np.save(file_path, loss_history)
        print "loss history saved to: {}".format(file_path)
        print
    else:
        print "loss history not saved (None given)"

    return folder


def print_model_evaluations(img_tensor, pert_img_tensor):
    print "original input evaluation:"
    print_top_5(img_tensor)
    print
    print "perturbated input evaluation:"
    print_top_5(pert_img_tensor)
    print



def plot_results(folder=".img_pert_temp"):
    names = [
        "original",
        "perturbated",
        "differences"
    ]

    for k, name in enumerate(names):
        fname = os.path.join(folder, name + ".jpg")
        img = io.imread(fname)
        plt.subplot(1, 3, k + 1)
        io.imshow(img)
        plt.axis('off')

    plt.tight_layout()


def plot_loss(losses, figname):
    plt.figure(figname)
    plt.gca().cla()
    plt.plot(losses)
    # plt.semilogy(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    display.clear_output(wait=True)
    display.display(plt.gcf())



#
# def plot_results(
#     processed_img, pert_img, diff,
#     indicate_center_of_mass=False,
# ):
#     proc = np.mean(processed_img, axis=2)
#     pert = np.mean(pert_img, axis=2)
#     loc = center_of_mass(diff[::-1, :])
#
#     fig, (ax1, ax2, ax3) = plt.subplots(
#         ncols=3, figsize=(15, 5),
#     )
#     fig.canvas.set_window_title("images")
#
#     im1 = ax1.pcolormesh(proc[::-1, :])
#     fig.colorbar(im1, ax=ax1, fraction=0.046)
#     ax1.set_aspect(1)
#     ax1.set_title("processed image")
#
#     im2 = ax2.pcolormesh(pert[::-1, :])
#     fig.colorbar(im2, ax=ax2, fraction=0.046)
#     ax2.set_aspect(1)
#     ax2.set_title("perturbated image")
#
#     im3 = ax3.pcolormesh(diff[::-1, :], cmap='Greys')
#     fig.colorbar(im3, ax=ax3, fraction=0.046)
#     ax3.set_aspect(1)
#     ax3.set_title("differences")
#     if indicate_center_of_mass:
#         ax3.annotate("X: center of mass", loc)
#
#     fig.tight_layout()
#     plt.show()


if __name__ == "__main__":
    main()