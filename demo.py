import cv2
import numpy as np
import tensorflow as tf
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from moviepy.editor import VideoFileClip, AudioFileClip
import logging

logging.basicConfig(level=logging.INFO)

video_file = 'D:\\桌面\\创作\\MAAS\\AI-2Dstyle-Transfer\\media\\beijing.mp4'
audio_file = 'D:\\桌面\\创作\\MAAS\\AI-2Dstyle-Transfer\\media\\audio_tmp.mp3'


class VideoTransBase(object):
    """
    content:make video transfer to 2-Dims style
    @:parameter: model,video,output_file
    @:type: ModelScope AI demo
    """

    def __init__(self):
        self.out_file = 'D:\\桌面\\创作\\MAAS\\AI-2Dstyle-Transfer\\output\\beijing.mp4'
        self.out_tmp_file = 'D:\\桌面\\创作\\MAAS\\AI-2Dstyle-Transfer\\output\\video_tmp.mp4'

    def __repr__(self):
        print("test")

    def model_load(self):
        self.model = pipeline('image-portrait-stylization', model='damo/cv_unet_person-image-cartoon_compound-models')

    def read_parse_video(self, video_file: str, audio_file: str) -> str:
        self.split_audio_src(video_file, audio_file)
        logging.info(f'load video {video_file}')
        video = cv2.VideoCapture(video_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        if not video.isOpened():
            logging.info("Error reading video file")

        frames = self.load_frames(video)
        # When everything done, release the video capture object
        video.release()
        logging.info('loading video done.')
        results = self.model(frames)

        result_frames = [r['output_img'] for r in results]
        # We need to set resolutions for writing video and  convert them from float to integer.
        frame_height, frame_width, _ = result_frames[0].shape
        size = (frame_width, frame_height)

        #  r, _, _, _ = lstsq(X, U)
        for idx in range(len(result_frames)):
            result_frames[idx] = result_frames[idx].astype(np.uint8)
        logging.info(f'saving video to file {self.out_tmp_file}')
        out = cv2.VideoWriter(self.out_tmp_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)

        for f in result_frames:
            out.write(f)
            cv2.imshow('frame: ', f)
        out.release()
        logging.info(f'saving video done')
        logging.info(f'merging audio and video')

        # loading video dsa gfg intro video
        clip = VideoFileClip(self.out_tmp_file)
        audioclip = AudioFileClip(audio_file)

        # adding audio to the video clip
        videoclip = clip.set_audio(audioclip)
        videoclip.write_videofile(self.out_file)
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
