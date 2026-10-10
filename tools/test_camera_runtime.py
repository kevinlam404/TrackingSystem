from time import sleep

from software.camera import Camera, LatestFrameCamera


class EndedCamera(Camera):
    def start(self):
        pass

    def read(self):
        return None

    def stop(self):
        pass


def test_latest_frame_camera_reports_ended_stream():
    camera = LatestFrameCamera(EndedCamera(), read_timeout_seconds=0.01)
    camera.start()
    try:
        for _ in range(20):
            if camera.ended:
                break
            sleep(0.005)
        assert camera.ended is True
        assert camera.read() is None
    finally:
        camera.stop()
