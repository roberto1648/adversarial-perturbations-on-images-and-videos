import torch
# from torchvision import models
import i3dpt
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
from skimage import morphology
import cv2
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import pickle
import os
from IPython import display
# import h5py
import pandas as pd
from datetime import datetime

import video_handler
import utils
#todo:test the changes
# - minimize top 5 >>> DONE
# - independent conv3d instance for each layer>>>DONE
# - save orig and pert scores along with class names in a csv.
# - datetime_subdir option>>>DONE

def main(
    input_path="data/video_samples/punching_bag/processed.mp4",
    kernel_size=(3, 3, 3),
    nblocks=3,
    nlayers=3,
    epochs=100,
    lr=0.01,
    l1_coeff=0.5,
    init_pert_model=False,
    save_to_dir=".pert_video_temp",
    datetime_subdir=True,
    graph=False,
    loss_figname="loss",
):
    model = load_model()

    inp = load_input(input_path)
    proc_inp = preprocess_input(inp)
    tensor = input_to_tensor(proc_inp)

    pert_model = PerturbationsGenerator(
        kernel_size, nblocks, nlayers,
    )
    pert_tensor, loss_history = get_optimum_perturbation(
        epochs, pert_model, tensor,
        model=model,
        init_pert_model=init_pert_model,
        lr=lr, l1_coeff=l1_coeff,
        figname=loss_figname,
    )

    print "original input evaluation:"
    input_assessment(tensor, model)
    print
    print "perturbated input evaluation:"
    input_assessment(pert_tensor, model)
    print

    tensor_np = tensor_to_numpy(tensor)
    pert_tensor_np = tensor_to_numpy(pert_tensor)
    diff = calculate_differences(tensor_np, pert_tensor_np)
    # diff, tensor_np, pert_tensor_np = post_processing(
    #     tensor, pert_tensor,
    # )
    save_results(
        save_to_dir, diff, tensor_np, pert_tensor_np,
        pert_model, loss_history, datetime_subdir,
    )
    html_object = None # for embeding videos in notebook
    if graph: html_object = plot_results(save_to_dir)

    return html_object

    # return tensor_np, pert_tensor_np, diff


class PerturbationsGenerator(torch.nn.Module):
    def __init__(self, kernel_size=(3, 3, 3), nblocks=1, nlayers=3):
        super(PerturbationsGenerator, self).__init__()
        # build conv layers, implement padding='same':
        # if np.mod(kernel_size, 2) == 0: kernel_size += 1
        # padding = kernel_size // 2

        # self.conv = i3dpt.Unit3Dpy(
        #     out_channels=3,
        #     in_channels=3,
        #     kernel_size=kernel_size,
        #     stride=(1, 1, 1),
        #     padding='SAME',
        #     activation=None, #'relu',
        #     use_bias= True, # False,
        #     use_bn=True
        # )
        # self.relu = torch.nn.ReLU()
        # self.nblocks = nblocks
        # self.nlayers = nlayers

        self.net = self.make_layers(nblocks, nlayers, kernel_size)

        if use_cuda(): self.cuda()

    def forward(self, x):
        # gather information for scaling
        xmin = torch.min(x)
        Dx = torch.max(x - xmin)

        # perturbate the video:
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

    def make_layers(self, nblocks, nlayers, kernel_size):
        layers = []
        for __ in range(nblocks):
            for __ in range(nlayers):
                conv = i3dpt.Unit3Dpy(
                    out_channels=3,
                    in_channels=3,
                    kernel_size=kernel_size,
                    stride=(1, 1, 1),
                    padding='SAME',
                    activation=None,  # 'relu',
                    use_bias=True,  # False,
                    use_bn=True
                )
                layers.append(conv)

            layers.append(torch.nn.ReLU())

        return torch.nn.Sequential(*layers)

    def initialize_conv_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                # initialize to pass the input unchanged:
                torch.nn.init.dirac_(m.weight)

                # todo: test the conditional:
                if m.bias is not None: # conv may be defined without bias (see above)
                    torch.nn.init.constant_(m.bias, 0.)


def get_optimum_perturbation(
        epochs, pert_model, tensor, model,
        init_pert_model=True,
        lr=0.1, l1_coeff=1., figname="loss",
):
    optimizer = torch.optim.Adam(
        pert_model.parameters(), lr=lr
    )

    if init_pert_model:
        pert_model.initialize_conv_weights() # init fiters to delta functions

    or_outputs = evaluate_model(model, tensor)
    or_vals, or_indxs = sort_flattened_tensor(or_outputs)
    category = list(np.ravel(or_indxs[0:5])) # top 5 categories
    # target = torch.nn.Softmax()(model(tensor))
    # category = np.argmax(target.cpu().data.numpy())
    print "Category with highest probability: {}".format(category)
    print "Optimizing.. "
    losses = []#;print torch.max(torch.abs(tensor))

    for i in tqdm(range(epochs)):
        pert_tensor = pert_model(tensor)
        pert_outputs = evaluate_model(model, pert_tensor)
        # pert_outputs = torch.nn.Softmax()(pert_outputs)
        # outputs = torch.nn.Softmax()(vgg_model(pert_img))
        tensor_diff = tensor - pert_tensor
        l1_term = l1_coeff * torch.mean(torch.abs(tensor_diff))
        l1_term = l1_term / torch.max(torch.abs(tensor)) # so that abs(l1_term) < 1
        sc_term = torch.sum(pert_outputs[0, category])# sum of (originally) top 5 categories
        loss = l1_term + sc_term
        # loss = l1_term + pert_outputs[0, category]

        # live plot:
        losses.append(loss.data[0])
        plot_loss(losses, figname)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.close() # to prevent a duplicate loss plot

    # # plot the loss:
    # plt.figure(figname)
    # plt.plot(losses)
    # plt.xlabel("epoch")
    # plt.ylabel("loss")

    original_score = evaluate_model(model, tensor)[0, category]
    pert_score = evaluate_model(model, pert_tensor)[0, category]
    print "original class score: {}".format(original_score)
    print "perturbed class score: {}".format(pert_score)
    print
    # print "original score: {}".format(torch.nn.Softmax()(model(tensor)[0])[0, category])
    # print "perturbed score: {}".format(torch.nn.Softmax()(model(pert_tensor)[0])[0, category])

    return pert_tensor, losses


def load_model():
    fname = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(fname, 'model_rgb.pth')

    model = i3dpt.I3D(num_classes=400, modality='rgb')
    model.eval()
    model.load_state_dict(torch.load(fname))

    if use_cuda():
        model.cuda()

    # for p in model.features.parameters():
    for p in model.parameters():
        p.requires_grad = False

    # for p in model.classifier.parameters():
    #         p.requires_grad = False

    return model


def load_input(
        input_path="data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy",
        graph=False,
):
    if os.path.splitext(input_path)[1] == ".npy":
        inp = np.load(input_path)
    else:
        inp = video_handler.read_video_frames(
            file_path=input_path, frame_from=0, frame_to=-1,
        )
        # inp = video_to_numpy(input_path)

    if graph:
        video_handler.embed_video_in_notebook(input_path)
        # plot_video_frames(inp, figsize)

    # if graph:
    #     inp_copy = np.copy(inp)
    #     shape = inp_copy.shape
    #     print "input shape: {}".format(shape)
    #
    #     if len(shape) > 4:
    #         inp_copy = inp_copy[0]
    #
    #     shape = inp_copy.shape
    #     side = int(np.ceil(np.sqrt(shape[0])))
    #
    #     plt.figure("original input", figsize=figsize)
    #
    #     for k, img in enumerate(inp_copy):
    #         ax = plt.subplot(side, side, k + 1)
    #         plt.imshow(img, vmax=img.max(), vmin=img.min())
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         plt.title("frame {}".format(k))

    return inp


def preprocess_input(inp):
    # inp is a 4D array: (nt, w, h, ch)
    # or 5D array (nsamples, nt, w, h, ch)
    ndims = len(inp.shape)
    assert (ndims == 4) or (ndims == 5)

    proc_inp = np.copy(inp)
    proc_inp = np.array(proc_inp, dtype=np.float32)

    # convert to 5D: (n_sample, nt, w, h, ch)
    # if needed:
    if len(proc_inp.shape) == 4:
        proc_inp = proc_inp[np.newaxis, ...]

    # resize frames to (224, 224)
    # and rescale to [-1, 1] range:
    for k in range(proc_inp.shape[1]):
        # rescale:
        proc_inp[:, k, ...] -= proc_inp[:, k, ...].min()
        vmax = proc_inp[:, k, ...].max()

        if vmax > 0:
            proc_inp[:, k, ...] /= 0.5 * vmax
            proc_inp[:, k, ...] -= 1.

    # resize:
    proc_inp[0, k, ...] = transform.resize(
        proc_inp[0, k, ...], (224, 224)
    )

    # switch to: (n_sample, ch, nt, w, h)
    proc_inp = proc_inp.transpose(0, 4, 1, 2, 3)

    return proc_inp


def input_to_tensor(inp):
    # tensor = inp.transpose(0, 4, 1, 2, 3)
    tensor = torch.from_numpy(inp)
    if use_cuda(): tensor = tensor.cuda()
    tensor_var = Variable(
        tensor, requires_grad=False
    )
    return tensor_var

    # preprocessed_img = transform.resize(img, (224, 224))
    # preprocessed_img = np.float32(preprocessed_img.copy())
    # preprocessed_img = preprocessed_img[:, :, ::-1]
    #
    # means=[0.485, 0.456, 0.406]
    # stds=[0.229, 0.224, 0.225]
    #
    # for i in range(3):
    #     preprocessed_img[:, :, i] =\
    #         preprocessed_img[:, :, i] - means[i]
    #     preprocessed_img[:, :, i] =\
    #         preprocessed_img[:, :, i] / stds[i]
    #
    # preprocessed_img = np.ascontiguousarray(
    #     np.transpose(preprocessed_img, (2, 0, 1))
    # )
    #
    # if use_cuda():
    #     preprocessed_img_tensor =\
    #         torch.from_numpy(preprocessed_img).cuda()
    # else:
    #     preprocessed_img_tensor =\
    #         torch.from_numpy(preprocessed_img)
    #
    # preprocessed_img_tensor.unsqueeze_(0)
    #
    # return Variable(preprocessed_img_tensor, requires_grad = False)



def input_assessment(
    tensor, model,
):
    top_k = 5
    # classes_path = 'data/kinetic-samples/label_map.txt'
    #
    # kinetics_classes = [x.strip() for x in open(classes_path)]
    kinetics_classes = get_kinetic_classes()

    out_var = evaluate_model(model, tensor)
    top_val, top_idx = sort_flattened_tensor(out_var)
    # out_tensor = out_var.data.cpu()
    # top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

    print 'Top {} classes and associated probabilities: '.format(top_k)
    print "(class index) class description: model score"

    for i in range(top_k):
        index = top_idx[i]
        value = top_val[i]
        print "({}) {}: {}".format(
            index, kinetics_classes[index], value,
        )



# def input_assessment(
#     tensor, model,
# ):
#     top_k = 5
#     classes_path = 'data/kinetic-samples/label_map.txt'
#
#     kinetics_classes = [x.strip() for x in open(classes_path)]
#
#     out_var, out_logit = model(tensor)
#     out_tensor = out_var.data.cpu()
#     top_val, top_idx = torch.sort(out_tensor, 1, descending=True)
#
#     print 'Top {} classes and associated probabilities: '.format(top_k)
#     print "(class index) class description: model score"
#
#     for i in range(top_k):
#         index = top_idx[0, i]
#         value = top_val[0, i]
#         print "({}) {}: {}".format(
#             index, kinetics_classes[index], value,
#         )
        # print '[{}]: {:.6E}'.format(kinetics_classes[top_idx[0, i]],
        #                             top_val[0, i])

    # return top_idx


# def input_assessment(inp, model):
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


def use_cuda():
    return torch.cuda.is_available()


def tensor_to_numpy(tensor):
    sample_np = tensor.data.cpu().numpy()[0] # one sample
    # transform to 0:time, 1,2:coords, 3:channel
    sample_np = np.transpose(sample_np, (1, 2, 3, 0))
    return sample_np


def calculate_differences(original, perturbated):
    orig = np.array(original).astype(np.float32)
    pert = np.array(perturbated).astype(np.float32)
    diff = np.abs(orig - pert)

    # indxs = orig != 0
    # diff[indxs] /= orig[indxs]

    # # remove the edges: artifacts due to padding may appear.
    # nt, h, w = np.shape(diff)[:3]
    #
    # diff[:, :int(0.1 * h), :] = 0
    # diff[:, int(0.9 * h):, :] = 0
    # diff[:, :, :int(0.1 * w)] = 0
    # diff[:, :, int(0.9 * w):] = 0

    # indxs = diff < 0.75 * diff.max()
    # diff[indxs] = 0

    # diff = diff ** 3

    # # dilate the important points left for visibility:
    # square = np.ones((10, 10))
    # diff = np.mean(diff, axis=-1)
    #
    # for j, __ in enumerate(diff):#range(nt):
    #     diff[j, ...] = morphology.dilation(diff[j, ...], square)

    return diff


# def post_processing(tensor, pert_tensor):
#     tensor_np = tensor_to_numpy(tensor)
#     pert_tensor_np = tensor_to_numpy(pert_tensor)
#
#     # # mean over image channels:
#     # proc = np.mean(proc_img_np, axis=2)
#     # pert = np.mean(pert_img_np, axis=2)
#
#     # # highlighting the differences:
#     # diff = (pert_tensor_np - tensor_np) ** 6
#     #
#     # # mean over channels:
#     # diff = np.mean(diff, axis=-1)
#
#     diff = np.abs(pert_tensor_np.astype(np.float32) - tensor_np.astype(np.float32))
#
#     indxs = tensor_np != 0
#     diff[indxs] /= tensor_np.astype(np.float32)[indxs]
#     diff = np.mean(diff, axis=-1)
#     diff = diff ** 5
#
#     # remove the edges: artifacts due to padding may appear.
#     nt, h, w = np.shape(diff)
#
#     diff[:, :int(0.1 * h), :] = 0
#     diff[:, int(0.9 * h):, :] = 0
#     diff[:, :, :int(0.1 * w)] = 0
#     diff[:, :, int(0.9 * w):] = 0
#
#     # dilate the important points left for visibility:
#     square = np.ones((20, 20))
#
#     for j in range(nt):
#         diff[j, ...] = morphology.dilation(diff[j, ...], square)
#
#     return diff, tensor_np, pert_tensor_np


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


# def plot_tensor_as_frames(
#         tensor_np,
#         global_norm=True,
#         figsize=(15, 15),
#         cmap="Greys",
#         figname="tensor_frames",
# ):
#     vid = np.copy(tensor_np)
#     shape = vid.shape
#     print "input shape: {}".format(shape)
#
#     if len(shape) > 4: vid = vid[0]
#
#     nt = vid.shape[0]
#     side = int(np.ceil(np.sqrt(nt)))
#
#     plt.figure(figname, figsize=figsize)
#
#     for k, img in enumerate(vid):
#         if global_norm:
#             vmax, vmin = vid.max(), vid.min()
#         else:
#             vmax, vmin = img.max(), img.min()
#         ax = plt.subplot(side, side, k + 1)
#         plt.imshow(
#             img, vmax=vmax, vmin=vmin, cmap=cmap,
#         )
#         # cb = plt.colorbar()
#         # cb.remove()
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.title("frame {}".format(k))
#
#     plt.tight_layout()
#
#
# def plot_video_frames(
#         tensor_np,
#         figsize=(15, 15),
#         figname="video frames",
# ):
#     vid = np.copy(tensor_np)
#     shape = vid.shape
#     print "input shape: {}".format(shape)
#
#     if len(shape) > 4: vid = vid[0]
#
#     nt = vid.shape[0]
#     side = int(np.ceil(np.sqrt(nt)))
#
#     plt.figure(figname, figsize=figsize)
#
#     for k, img in enumerate(vid):
#         # force img to [0, 255] range:
#         img_norm = img - img.min()
#         img_max = img_norm.max()
#         if img_max > 0: img_norm *= 255. / img_max
#         img_norm = np.array(img_norm, dtype=np.int)
#
#         ax = plt.subplot(side, side, k + 1)
#         plt.imshow(img_norm)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.title("frame {}".format(k))
#
#     plt.tight_layout()


def evaluate_model(model, inp, softmax=False):
    if "torch" not in str(type(inp)):
        tensor = preprocess_input(inp)
        tensor = input_to_tensor(tensor)
    else:
        tensor = inp

    out_var, out_logit = model(tensor)

    if softmax:
        sh = out_var.shape
        out_var = torch.nn.Softmax(dim=0)(out_var.view(-1))
        out_var = out_var.view(sh)

    return out_var


def sort_flattened_tensor(tensor):
    # flatten the tensor:
    x = tensor.view(-1)
    # sort:
    vals, indxs = torch.sort(x, descending=True)
    return vals, indxs


# def video_to_numpy(file_path=""):
#     vc = cv2.VideoCapture()
#     vc.open(file_path)
#     frames = []
#     more_frames = True
#
#     while more_frames:
#         more_frames, img = vc.read()
#
#         if more_frames:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             frames.append(img)
#
#     frames = np.array(frames, dtype=np.float32)
#
#     return frames


def get_kinetic_classes():
    fname = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(fname, 'label_map.txt')

    kinetics_classes = [x.strip() for x in open(fname)]

    return np.array(kinetics_classes)


def save_results(
        save_to_dir=".pert_video_temp",
        diff=None,
        tensor_np=None,
        pert_tensor_np=None,
        pert_model=None,
        loss_history=None,
        datetime_subdir=True,
):
    vids = [diff, tensor_np, pert_tensor_np]
    names = [
        "differences",
        "original_224x224",
        "perturbated"
    ]

    folder = save_to_dir
    if (save_to_dir != ".pert_video_temp") and datetime_subdir:
        folder = os.path.join(save_to_dir, str(datetime.now()))

    if not os.path.isdir(folder): os.makedirs(folder)

    print "results saved to:"

    for vid, name in zip(vids, names):
        if vid is not None:
            # file_path = os.path.join(save_to_dir, name)
            # np.save(file_path, vid)#takes too much disk space
            # print file_path

            file_path = os.path.join(folder, name + ".mp4")
            video_handler.save_np_array_as_video(
                file_path=file_path, np_array=vid,
            )
            print file_path
        else:
            print "{} not saved (None given)".format(name)
    print

    if pert_model is not None:
        file_path = os.path.join(folder, "pert_model.pt")
        torch.save(pert_model, file_path)
        print "perturbation generator model saved to:"
        print file_path
        print
    else:
        print "model not saved (None given)"

    if loss_history is not None:
        file_path = os.path.join(folder, "loss_history")
        np.save(file_path, loss_history)
        print "loss history saved to:"
        print file_path
        print
    else:
        print "loss history not saved (None given)"

    # save the original and pert scores:
    if (tensor_np is not None) and (pert_tensor_np is not None):
        scores_df = pd.DataFrame()
        scores_df["kinetics_class"] = get_kinetic_classes()
        model = load_model()
        scores_df["original_score"] = np.ravel(
            evaluate_model(model, tensor_np)
        )
        scores_df["perturbated_score"] = np.ravel(
            evaluate_model(model, pert_tensor_np)
        )
        fname = os.path.join(folder, "scores.csv")
        scores_df.to_csv(fname)
        print "scores saved to: {}".format(fname)
    else:
        print "scores not saved, need to input both original and perturbated images"



def plot_results(save_to_dir):
    names = ["original_224x224.mp4", "perturbated.mp4", "differences.mp4"]
    file_paths = [os.path.join(save_to_dir, name) for name in names]
    html_object = video_handler.embed_videos(file_paths)
    # need to return to insert to notebook html:
    return html_object


def plot_loss(losses, figname):
    plt.figure(figname)
    plt.gca().cla()
    plt.plot(losses)
    # plt.semilogy(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    display.clear_output(wait=True)
    display.display(plt.gcf())


if __name__ == "__main__":
    main()