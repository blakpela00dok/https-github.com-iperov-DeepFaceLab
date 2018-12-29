# For Mac Users
If you just have a **MacBook**.DeepFaceLab **GPU** mode does not works. However,it can also works with **CPU** mode.Follow the Steps below will help you build the **DRE** (DeepFaceLab Runtime Environment) Easier.

### 1. Open a new terminal and Clone DeepFaceLab with git
```
$ git clone git@github.com:Pluckypan/DeepFaceLab.git
```

### 2. Change the directory to DeepFaceLab
```
$ cd DeepFaceLab
```

### 3. Install Docker

[Docker Desktop for Mac] (https://hub.docker.com/editions/community/docker-ce-desktop-mac)

### 4. Build Docker Image For DeepFaceLab

```
$ docker build -t deepfacelab-cpu -f Dockerfile.cpu .
```

### 5. Mount DeepFaceLab volume and Run it

```
$ docker run -p 8888:8888  --hostname deepfacelab-cpu --name deepfacelab-cpu  -v $PWD:/notebooks  deepfacelab-cpu
```

PS: Because your current directory is `DeepFaceLab`,so `-v $PWD:/notebooks` means Mount `DeepFaceLab` volume to `notebooks` in **Docker**

And then you will see the log below:

```
The Jupyter Notebook is running at:
http://(deepfacelab-cpu or 127.0.0.1):8888/?token=your token
```

### 6. Open a new terminal to run DeepFaceLab in /notebooks

```
$ docker exec -it deepfacelab-cpu bash
$ ls -A
```

### 7. Use jupyter in deepfacelab-cpu bash

```
$ jupyter notebook list
```
or just open it on your browser `http://127.0.0.1:8888/?token=your_token`

PS: You can run python with jupyter.However,we just run our code in bash.It's simpler and clearer.Now the **DRE** (DeepFaceLab Runtime Environment) almost builded.

### 8. Stop or Kill Docker Container

```
$ docker stop deepfacelab-cpu
$ docker kill deepfacelab-cpu
```

### 9. Start Docker Container

```
# start docker container
$ docker start deepfacelab-cpu
# open bash to run deepfacelab
$ docker exec -it deepfacelab-cpu bash
```

PS: `STEP 8` or `STEP 9` just show you the way to stop and start **DRE**.

### 10. enjoy it

```
# make sure you current directory is `/notebooks`
$ pwd
# make sure all `DeepFaceLab` code is in current path `/notebooks`
$ ls -a
# read and write permission
$ chmod +x cpu.sh
# run `DeepFaceLab`
$ ./cpu.sh
```

### Details with `DeepFaceLab`
