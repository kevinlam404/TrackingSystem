from time import sleep

from software.camera import Camera, LatestFrameCamera


def test_application_config_selects_picamera_without_opening_hardware():
    from software.camera import PiCamera
    from software.config import load_config
    from software.main import build_camera
    config = load_config()['camera']
    assert config['source'] == 'picamera'
    camera = build_camera(config)
    assert isinstance(camera, LatestFrameCamera)
    assert isinstance(camera._camera, PiCamera)
    assert (camera._camera.width, camera._camera.height, camera._camera.fps) == (640, 480, 30)


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
