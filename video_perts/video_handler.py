import skvideo.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
# from IPython.display import HTML

import utils


def read_video_between_times(
        file_path="",
        time_from=0, # seconds
        time_to=10, # seconds
):
    fps = get_video_frames_per_second(file_path)
    frame_from = int(np.round(time_from * fps))
    frame_to = int(np.round(time_to * fps))
    frames = read_video_frames(file_path, frame_from, frame_to)

    return frames


def read_video_frames(
        file_path="",
        frame_from=0,
        frame_to=10, # -1 for last frame
):
    videogen = skvideo.io.vreader(file_path)
    frames = []

    for k, frame in enumerate(videogen):
        if k >= frame_from:
            if (frame_to == -1) or (k < frame_to):
                frames.append(frame)

    return np.array(frames)


# def read_video_frames(
#         file_path="",
#         frame_from=0,
#         frame_to=10,
# ):
#     videogen = skvideo.io.vreader(file_path)
#     frames = []
#
#     for k, frame in enumerate(videogen):
#         if (k >= frame_from) and (k < frame_to): frames.append(frame)
#
#     return np.array(frames)


def get_video_metadata(file_path=""):
    metadata = skvideo.io.ffprobe(file_path)
    return metadata


def get_video_frames_per_second(file_path=""):
    metadata = get_video_metadata(file_path)
    vid_meta = metadata["video"]
    frame_rate = vid_meta["@avg_frame_rate"] # e.g., "25/1"
    num = float(frame_rate.split("/")[0])
    den = float(frame_rate.split("/")[1])
    fps = num / den
    return fps


def save_video(file_path, video):
    utils.create_directory_if_needed(file_path)
    vid = np.array(video)
    skvideo.io.vwrite(file_path, vid.astype(np.uint8))


def save_np_array_as_video(file_path, np_array):
    video = np_array_to_video_range(np_array)
    save_video(file_path, video)


def np_array_to_video_range(np_array):
    vid = np.copy(np_array).astype(np.float32)
    vid -= vid.min()
    if vid.max() > 0: vid *= 255. / vid.max()
    vid = vid.astype(np.uint8)
    return vid


def video_to_minus_one_to_one(video):
    vid = np.copy(video).astype(np.float32)
    vid -= vid.min()
    if vid.max() > 0: vid /= vid.max()
    return vid


def resize_video_frames(video, shape=(224, 224)):
    vid = video_to_minus_one_to_one(video)
    new_frames = []

    for frame in vid:
        res_frame = skimage.transform.resize(
            frame.astype(np.float32), shape
        )
        new_frames.append(res_frame)

    new_frames = np_array_to_video_range(new_frames)

    return new_frames


def plot_as_video_frames(
        video,
        figsize=(20, 20),
        figname="video frames",
):
    vid = np.copy(video)
    shape = vid.shape
    print "input shape: {}".format(shape)

    if len(shape) > 4:
        vid = utils.get_subarray_from_last_ndims(vid, 4)

    # to video range:
    vid = np_array_to_video_range(vid)

    nt = vid.shape[0]
    side = int(np.ceil(np.sqrt(nt)))

    plt.figure(figname, figsize=figsize)

    for k, img in enumerate(vid):
        ax = plt.subplot(side, side, k + 1)
        plt.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("frame {}".format(k))

    plt.tight_layout()


def plot_frames_with_cmap(
        tensor_np,
        global_norm=True,
        figsize=(15, 15),
        cmap="Greys",
        figname="tensor_frames",
):
    vid = np.copy(tensor_np)
    shape = vid.shape
    print "input shape: {}".format(shape)

    if len(shape) > 4:
        vid = utils.get_subarray_from_last_ndims(vid, 4)

    nt = vid.shape[0]
    side = int(np.ceil(np.sqrt(nt)))

    plt.figure(figname, figsize=figsize)

    for k, img in enumerate(vid):
        if global_norm:
            vmax, vmin = vid.max(), vid.min()
        else:
            vmax, vmin = img.max(), img.min()
        ax = plt.subplot(side, side, k + 1)
        plt.imshow(
            img, vmax=vmax, vmin=vmin, cmap=cmap,
        )
        # cb = plt.colorbar()
        # cb.remove()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("frame {}".format(k))

    plt.tight_layout()


def build_video_html(
        file_path="",
        autoplay=True,
        loop=True,
):
    video = io.open(file_path, 'r+b').read()
    encoded = base64.b64encode(video)

    if autoplay:
        aut = "autoplay"
    else:
        aut = ""

    if loop:
        lp = "loop"
    else:
        lp = ""

    data_html = '''<video alt="test" controls {1} {2}>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'), aut, lp)

    return data_html


# def embed_html_in_notebook(html=""):
#     return HTML(data=html)


def embed_video_in_notebook(file_path="", autoplay=True, loop=True):
    data_html = build_video_html(file_path, autoplay, loop)
    return utils.embed_html_in_notebook(data_html)


def embed_videos(
        file_paths=[""],
        autoplay=True,
        loop=True,
        on_row=True,
):
    html = ""

    for file_path in file_paths:
        html += build_video_html(file_path, autoplay, loop)
        if not on_row: html += "<br>"

    return utils.embed_html_in_notebook(html)


def embed_video_grid(
        video_files=[[]],
        autoplay=True,
        loop=True,
        label_rows=True,
):
    html = ""

    for row in video_files:
        if label_rows:
            html += "<a>{}</a><br>".format(row)
        for vid_path in row:
            html += build_video_html(
                    file_path=vid_path,
                    autoplay=autoplay,
                    loop=loop,
            )
        html += "<br>"

    return utils.embed_html_in_notebook(html)

