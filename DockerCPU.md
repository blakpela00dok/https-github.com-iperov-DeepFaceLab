# For Mac Users
if you just have a MacBook.deepfacelab gpu mode does not works. however,it can also works with cpu mode.

### 1. Install Docker

[Docker Desktop for Mac] (https://hub.docker.com/editions/community/docker-ce-desktop-mac)

### 2. Build Docker Image For

```
$ docker build -t deepfacelab-cpu -f Dockerfile.cpu .
```

### 3. Mount deepfacelab volume and Run it

```
$ docker run -p 8888:8888  --hostname deepfacelab-cpu --name deepfacelab-cpu  -v **your source path**:/srv  deepfacelab-cpu
# for example
$ docker run -p 8888:8888  --hostname deepfacelab-cpu --name deepfacelab-cpu  -v /Users/plucky/own/DeepFaceLab:/srv  deepfacelab-cpu
```

then you will see the log:

```
The Jupyter Notebook is running at:
http://(deepfacelab-cpu or 127.0.0.1):8888/?token=your token
```

### 4. Open a new terminal to run deepfacelab in /srv

```
$ docker exec -it deepfacelab-cpu bash
$ cd ../srv/
```

### 5. Use jupyter in deepfacelab-cpu bash

```
$ jupyter notebook list
```
or just open it on your browser `http://127.0.0.1:8888/?token=your_token`

### 6. Close or Kill Docker Container

```
$ docker kill deepfacelab-cpu
```

### 7. Start Docker Container

```
$ docker start -i deepfacelab-cpu
$ docker exec -it deepfacelab-cpu bash
```

### 8. enjoy it

```
$ cd ../srv/
$ chmod +x cpu.sh
$ ./cpu.sh
```

####  some error

```
1. localization.py system_locale is NoneType
# system_locale may be nil
system_language = system_locale[0:2] if system_locale is not None else "en"


```

[Install NVIDIA Driver](http://www.linuxandubuntu.com/home/how-to-install-latest-nvidia-drivers-in-linux)
[安装N卡驱动](https://linuxstory.org/how-to-install-latest-nvidia-drivers-in-linux/)
