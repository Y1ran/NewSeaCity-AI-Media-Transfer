import cv2
import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip, AudioFileClip
import logging
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

logging.basicConfig(level=logging.INFO)

video_file = 'D:\\桌面\\创作\\MAAS\\AI-2Dstyle-Transfer\\media\\other_1.mp4'
audio_file = 'D:\\桌面\\创作\\MAAS\\AI-2Dstyle-Transfer\\media\\audio_tmp.mp3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")


class VideoTransBase(object):
    """
    content:make video transfer to 2-Dims style
    @:parameter: model,video,output_file
    @:type: ModelScope AI demo
    """

    def __init__(self):
        self.model = None
        self.out_file = 'D:\\桌面\\创作\\MAAS\\AI-2Dstyle-Transfer\\output\\other_1'
        self.chunk = 500

    def __enter__(self):
        print("test started...")

    def model_load(self):
        self.model = pipeline('image-portrait-stylization', model='damo/cv_unet_person-image-cartoon_compound-models')

    def read_parse_video(self, video_file: str, audio_file: str):
        # if you have BGM inside source video, uncomment below code
        # self.split_audio_src(video_file, audio_file)

        logging.info(f'load video {video_file}')
        video = cv2.VideoCapture(video_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        if not video.isOpened():
            logging.info("Error reading video file")

        frames = self.load_frames(video)
        # When everything done, release the video capture object
        video.release()
        logging.info('loading video done.')

        for idx in range(0, len(frames), self.chunk):
            results = self.model(frames[idx:idx + self.chunk])
            logging.info('model result has has been done: ' + str(idx))

            result_frames = [r['output_img'] for r in results]
            # We need to set resolutions for writing video and  convert them from float to integer.
            frame_height, frame_width, _ = result_frames[0].shape
            size = (frame_width, frame_height)

            #  r, _, _, _ = lstsq(X, U)
            for pos in range(len(result_frames)):
                result_frames[pos] = result_frames[pos].astype(np.uint8)

            filename = self.out_file + "_" + str(idx + 1) + ".mp4"
            logging.info(f'saving video to file {filename}')
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)

            for f in result_frames:
                out.write(f)
                # cv2.imshow('frame: ', f)
            out.release()
            logging.info(f'saving {idx/1000}th video done')

        # save to gif
        logging.info('finished!')

    def load_frames(self, video) -> list:
        frames = []
        i = 0
        while video.isOpened():
            i += 1
            if i % 10:
                logging.info(f'loading {i} frames')
            ret, frame = video.read()
            if ret:
                frames.append(frame)
            else:
                break
        return frames

    @staticmethod
    def split_audio_src(video_file, audio_file):
        my_clip = VideoFileClip(video_file)
        my_clip.audio.write_audiofile(audio_file)
        logging.info('save audio file done')


def test():
    video_trans = VideoTransBase()
    video_trans.model_load()
    video_trans.read_parse_video(video_file, audio_file)


if __name__ == "__main__":
    test()
