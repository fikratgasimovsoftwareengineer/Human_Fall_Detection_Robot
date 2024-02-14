#VideoStreaming On Remote PC, based on Communication Protocol of UDP with Embedded Board TX2
### Install First Gstreamer
Installation is derived from [github](https://github.com/GStreamer/gstreamer.git)
_Source Installation is as follows_

1. Install from Internet: $ wget https://gstreamer.freedesktop.org/src/gst-rtsp-server/gst-rtsp-server-1.13.91.tar.xz
2. Extract from tar: $ tar xf gst-rtsp-server-1.13.91.tar.xz
3. Go to directy: $ cd /data/gst-rtsp-server-1.13.91
4. Execute: $ ./autogen.sh --disable-gtk-doc
5. $ make -j4

### Istallation of JetPack4.5.1
_This version of Jetpack, being more stable, providing these version of compatibility_
1. OpenCV 4.1.1
2. TensorRT
3. CudaDNN8
4. Cuda10.2
### Install Json from Source 
1. $ sudo apt install nlohmann-json-dev

### Generate SDP Video File
_Then This video file has to be set in client side_

### NOTE! If you want to go throught all systems that have been developed, please have a look below link:

[google drive](https://docs.google.com/document/d/12MvDksdUVVAfGLSFn3dUlcQ6uEvmTkz47jZcsHXkG0s/edit?usp=sharing)
