import numpy as np
import pandas as pd
from pytube import YouTube
import time
import os

import utils
import video_handler


def main(
        samples_dir="data/samples",
        nsamples=20,
        random_state=4,
):
    samples_df = pick_samples(nsamples, random_state)

    for k, (index, row) in enumerate(samples_df.iterrows()):
        row.name = index
        print "sample {} of {}".format(k+1, nsamples)

        try:
            process_sample(row, samples_dir)
        except Exception as e:
            print "error: ", e

        print


def pick_samples(nsamples=10, random_state=4):
    fname = "data/kinetics-400/kinetics_train/kinetics_train.csv"
    df = pd.read_csv(fname)
    samples_df = df.sample(nsamples, random_state=random_state)
    return samples_df


def process_sample(
        sample_ser=pd.Series(),
        samples_dir="data/samples"
):
    label, id, time_start, time_end = unpack_params(sample_ser)
    print label
    sample_dir = os.path.join(samples_dir, label)

    if not os.path.isdir(sample_dir):
        save_metadata(sample_ser, sample_dir)
        download_stream(id, sample_dir)

    process_image(sample_dir, time_start, time_end)

    # slow down to avoid upsetting youtube:
    time.sleep(10 + 10 * np.random.rand())


def unpack_params(row=pd.Series()):
    label = row["label"]
    id = row["youtube_id"]
    time_start = row["time_start"]
    time_end = row["time_end"]
    return label, id, time_start, time_end


def save_metadata(
        sample_ser=pd.Series(),
        sample_dir="",
):
    # save the row metadata to the sample folder
    fname = os.path.join(sample_dir, "sample_description.csv")
    utils.create_directory_if_needed(fname)
    sample_ser.to_csv(fname)


def download_stream(id, stream_dir):
    # get the related youtube streams:
    url = 'http://youtube.com/watch?v={}'.format(id)
    yt = YouTube(url)

    stream = pick_best_stream(yt)
    print "downloading stream from {}".format(url)
    stream.download(
        output_path=stream_dir,
        filename="original"
    )
    file_name = find_named_file("original", stream_dir)
    print "saved to: {}".format(os.path.join(stream_dir, file_name))


def pick_best_stream(yt):
    # pick only mp4:
    valid_streams = yt.streams.filter(file_extension='mp4').all()
    # # pick only the streams with video:
    # all_streams = yt.streams.all()
    # only_audio = yt.streams.filter(only_audio=True).all()
    # valid_streams = [x for x in all_streams if x not in only_audio]
    res_diff = 1e6
    best_stream = valid_streams[0]

    # pick a stream closer to 224p resolution, prefer mp4:
    for stream in valid_streams:
        if hasattr(stream, "resolution"):
            res = stream.resolution
            res = res.replace("p", "").strip()
            res = int(res)

            if np.abs(res - 224) < res_diff:
                best_stream = stream
                res_diff = np.abs(res - 224)

                # if hasattr(stream, "video_codec"):
                #     if "mp4" in stream.video_codec:
                #         best_stream = stream
                #         res_diff = np.abs(res - 224)

    return best_stream


def process_image(sample_dir, time_start, time_end):
    file_name = find_named_file("original", sample_dir)
    file_path = os.path.join(sample_dir, file_name)
    # reduced time video:
    vid = video_handler.read_video_between_times(
            file_path, time_start, time_end,
        )
    # resize to 224x224:
    vid = video_handler.resize_video_frames(
            vid, (224, 224)
        )
    # subsample to avoid runtime error (in my gpu)
    # when perturbating. 80 frames work:
    nt = vid.shape[0]
    nsub = int(np.round(nt / 80.))
    if nsub > 0: vid = vid[::nsub, ...]

    # save:
    file_name = os.path.join(sample_dir, "processed.mp4")
    video_handler.save_video(file_name, vid)
    print "processed video saved to: {}".format(file_name)


def find_named_file(name="original",
                    directory="data/samples"):
    files = os.listdir(directory)
    file_found = ""

    for file in files:
        file_name, termination = os.path.splitext(file)
        if file_name == name:
            file_found = file
            break

    return file_found


if __name__ == "__main__":
    main()