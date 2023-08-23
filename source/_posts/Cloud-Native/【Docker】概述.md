---
title: 【Docker】概述
date: 2023-07-23 15:23:01
tag: 
   - docker
   - Cloud Native
category:
   - Docker 
---

### 什么是容器？

简而言之，容器是计算机的沙盒进程，与主机上的所有其他进程隔离，这种隔离利用了`内核命名空间和cgroup`，这些功能在`linux`中已经存在很长时间了。`Docker`正是利用了这些能力。总而言之，容器包括以下这些能力：

- 是一个可运行的镜像实例，你可以使用`Docker API`或`CLI`创建、启动、停止、移动或删除容器。
- 可以在本地机器、虚拟机、云上部署和运行。
- 是跨平台并且可移植的。
- 与其他容器隔离运行自己的应用、二进制文件以及配置。

### 什么是容器镜像？

运行容器时，它使用隔离的文件系统。该自定义文件系统由容器镜像提供。由于镜像包含容器的文件系统，因此它必须包含运行应用程序所需的所有内容-所有依赖项、配置、脚本、二进制文件等。镜像还包含容器的其它配置，例如环境变量、要运行的默认命令和其它元数据。

### 将应用程序在容器中部署

#### 准备工作

- 在机器上下载&安装`Docker`
- 在机器上下载&安装`git`客户端
- 在机器上安装一个编辑文件的IDE或文本编辑器，这里建议下载&安装`Visual Studio Code`

<!-- more -->

#### 下载应用程序

在运行应用程序之前，你需要将应用程序源代码下载的你的机器里。

1. 使用以下命令下载应用程序源码：

    ```
    $ git clone git@github.com:docker/getting-started.git
    ```
2. 查看源码，在`getting-started/app`目录中，你应该看到`package.json`和两个子目录（`src`和`spec`）。

{% asset_img ide-screenshot.png %}

#### 构建应用程序的容器镜像

1. 在应用的`app`目录下创建一个`Dockerfile`文件，你可以使用以下命令创建`Dockerfile`文件

    ```
    $ touch Dockerfile
    ```
2. 使用文本编辑器或IDE，讲一下内容添加到`Dockerfile`文件中

    ```
    # syntax=docker/dockerfile:1
   
    FROM node:18-alpine
    WORKDIR /app
    COPY . .
    RUN yarn install --production
    CMD ["node", "src/index.js"]
    EXPOSE 3000
    ```

3. 使用以下命令创建容器镜像：

    ```
    $ cd getting-started/app
    ```

    构建容器镜像

    ```
    $ docker build -t getting-started .
    ```
    `docker build`命令使用 `Dockerfile` 构建新的容器镜像。您可能注意到`Docker`下载了很多"层"。只是因为你指定构建器要从`node:18-alpine`镜像开始。但我们的机器上没有此镜像。因此`Docker`需要下载该镜像。

    `Docker`下载该镜像后，`Dockerfile`中的指令会被复制到你的应用程序中并用`yarn`安装应用程序的依赖项。该`CMD`指令指定了从镜像启动容器时要运行的默认命令。

    最后`-t`参数标记你的镜像名称为：`getting-started`, 因此你可以在运行容器时引用该镜像。命令末尾的`.`是告诉`Docker`执行`docker build`命令应该在当前目录中查找`Dockerfile`文件。

#### 在容器中启动应用程序

现在你已经有了镜像，你可以在容器中运行应用程序。为此，你将使用`docker run`命令。

1. 使用以下命令启动容器`docker run` 并指定刚刚创建的镜像名称：

    ```
    $ docker run -dp 127.0.0.1:3000:3000 getting-started
    ```
    该`-d`参数（缩写：`--detach`）在后台运行容器。该`-p`参数（`--publish`）在主机和容器之间创建端口映射。该`-p`参数采用的格式是字符串值`HOST:CONTAINER`，其中`HOST`是主机上的地址，`CONTAINER`是容器的端口，此处的命令是将容器的端口`3000`发布到主机上`127.0.0.1:3000`。`127.0.0.1:3000`如果没有端口映射，则无法从主机访问应用。

2. 几秒钟后，打开`web`浏览器访问`http://localhost:3000`，您应该会看到你的应用程序。

如果你想快速查看一下容器，你应该会看到至少有一个在使用的`getting-started`镜像，是在`port:3000`上运行的容器，要查看容器可以使用`CLI`或 `Docker Desktop`图形界面。

`CLI`, 在终端中运行 `docker ps` 命令，列出你的容器：
    
```
$ docker ps 
```
应该输出以下内容：
     
```
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                      NAMES
df784548666d        getting-started     "docker-entrypoint.s…"   2 minutes ago       Up 2 minutes        127.0.0.1:3000->3000/tcp   priceless_mcclintock
```

#### 更新应用程序

**更新源码**

在下面的步骤中，您将在没有任何待办事项列表项时将“空文本”更改为“您还没有待办事项！上面加一个！

1. 在`src/static/js/app.js`文件中，更新第56行使用新的空文本。

    ```
    - <p className="text-center">No items yet! Add one above!</p>
    + <p className="text-center">You have no todo items yet! Add one above!</p>
    ```

2. `docker build`使用您第2部分中使用相同命令构建镜像的更新版本

    ```
    $ docker build -t getting-started .
    ```
3. 使用更新的代码启用一个新容器。
    
    ```
    $ docker run -dp 127.0.0.1:3000:3000 getting-started
    ```
你可能会看到这样一个错误（ID会不同）

    ```
    docker: Error response from daemon: driver failed programming external connectivity on endpoint laughing_burnell 
            (bb242b2ca4d67eba76e79474fb36bb5125708ebdabd7f45c8eaf16caaabde9dd): Bind for 127.0.0.1:3000 failed: port is already allocated.
    ```
发生错误的原因是您无法在旧容器仍在运行时启动新容器。原因是旧容器已经在使用主机的3000端口，并且机器上只有一个进程（包括容器）可以监听特定端口。要解决此问题，你需要删除此容器。

**删除旧容器**

要删除容器，你首先要停止它。一旦停止，你就可以将其删除。你可以使用`CLI`或`Docker Desktop`图形界面来删除旧容器

1. 使用命令获取容器ID `docker ps`。

    ```
    $ docker ps
    ```
2. 使用docker stop命令停止容器，将 `<the-container-id>` 替换为 中的 `ID docker ps`。

    ```
    $ docker stop <the-container-id>
    ```
3. 容器停止后，你可以使用`docker rm`命令将其删除

    ```
    $ docker rm <the-container-id>
    ```
你可以用一条命令来停止&删除容器`docker rm -f <the-container-id>`

**启动更新后的应用程序容器**

1. 现在，使用命令启动更新后的应用程序`docker run`。

    ```
    $ docker run -dp 127.0.0.1:3000:3000 getting-started
    ```
2. 在http://localhost:3000上刷新浏览器，你应该会看到更新后的帮助文本。

#### 共享应用程序

现在你已经构建了镜像，你可以共享它，要共享`Docker`镜像，你必须使用`Docker`注册表。默认注册表是`Docker Hub`，你使用的所有镜像都来自于此。

> Docker ID 允许您访问 Docker Hub，这是世界上最大的容器镜像库和社区。如果您没有Docker ID，请免费创建一个。

**创建一个仓库**

要推送镜像，首先要在 `Docker Hub` 创建一个存储库

1. 注册/登录[Docker Hub](https://hub.docker.com/)。

2. 选择创建仓库按钮。

3. 对于仓库的名称，请使用 `getting-started` 确保可见性，请选择`Public`。

4. 选择创建按钮。

**推送**

1. 在命令行中，运行你在`Docker Hub`上看到的命令。请注意，你的命令将使用你的命名空间，而不是`docker`

    ```
    docker push docker/getting-started
    The push refers to repository [docker.io/docker/getting-started]
    An image does not exist locally with the tag: docker/getting-started
    ```
    为什么报错了？ Push命令正在寻找名为`docker/getting-started` 的镜像，但没有找到，要解决此问题，你需要“标记”你构建的现有镜像，为其指定另一个名称。

2. 使用命令登录`Docker Hub` `docker login -u YOUR-USER-NAME`。

3. 使用该`docker tag` 命令为镜像指定 `getting-started` 新名称。请务必更换 `YOUR-USER-NAME`为您的 `Docker ID`。

    ```
    $ docker tag getting-started YOUR-USER-NAME/getting-started
    ```

4. 现在再次尝试推送命令。如果您从`Docker Hub` 复制该值，则可删除 `tagname` 部分，因为你没有想镜像添加标签。如果你不指定标签，`Docker`将使用名为`latest`的标签。

    ```
    $ docker push YOUR-USER-NAME/getting-started
    ```

**在新实例上运行镜像**

现在你的镜像已构建并推送到注册表中，请尝试在从未见过此容器映像的全新实例上运行您的应用程序。为此，您将使用 `Play with Docker`。

1. 打开浏览器访问[Docker](https://labs.play-with-docker.com/)

2. 选择“登录”，然后从下拉列表中选择“docker” 。

3. 连接您的 `Docker Hub` 帐户。

4. 登录后，选择左侧栏上的“添加新实例”选项。如果您没有看到它，请将您的浏览器设置得更宽一些。几秒钟后，浏览器中将打开一个终端窗口。

    {% asset_img pwd-add-new-instance.png %}

5. 在终端中，启动新推送的应用程序。

    ```
    $ docker run -dp 0.0.0.0:3000:3000 YOUR-USER-NAME/getting-started
    ```
6. 当 3000 徽章出现时，选择它，您应该会看到经过修改的应用程序。如果 3000 徽章未显示，您可以选择“打开端口”按钮并输入 3000。

#### 容器的文件系统

当容器运行时，它会使用镜像中的各个层作为其文件系统。每个容器还拥有自己的“临时空间”来创建/更新/删除文件。即使它们使用了相同的镜像。任何更改都不会再另一个容器中生效。

要查看实际效果，我们启动两个容器并在每个容器中创建一个文件。你将看到的是，一个容器中创建的文件在另一个容器中不可用。

1. 启动一个`ubuntu`容器，该容器将创建一个1到10000之间的随机数命名的文件`/data.txt`。

    ```
    $ docker run -d ubuntu bash -c "shuf -i 1-10000 -n 1 -o /data.txt && tail -f /dev/null"
    ```
    如果你对该命令感到好奇，你可以启动`bash shell`并调用两个命令 (为什么有`&&`) 。第一部分选择一个随机数将其写入`/data.txt`，第二个命令只是监视一个文件以保持容器运行。

2. 验证你是否可以通过访问容器中的终端来查看输出。为此，你可以使用CLI 或 Docker Desktop 的图形界面。

在命令行上可以使用`docker exec`命令访问容器，你需要获取容器的ID（使用`docker ps`来获取）。在MAC或Linux终端中，或者Windows命令提示符或PowerShell中，使用以下命令获取内容。

    ```
    $ docker exec <container-id> cat /data.txt
    ```

3. 现在启动一个`ubuntu`（相同的镜像），你会发现没有相同的文件

    ```
    $ docker run -it ubuntu ls /
    ```
    这种情况下，该命令列出容器根目录下的文件，那里没有`data.txt`！这是因为它们仅写入第一个容器的暂存空间。

4. 继续使用`docker rm -f <container-id>` 命令删除第一个容器。

通过之前的实验，您看到每个容器每次启动时都从图像定义开始。虽然容器可以创建、更新和删除文件，但当您删除容器时，这些更改将会丢失，并且 Docker 会隔离对该容器的所有更改。有了卷，你就可以改变这一切。

卷提供了将容器的特定文件系统路径连接回主机的能力。如果在容器中挂载目录，则该目录中的更改也会在主机上看到。如果您在容器重新启动时挂载相同的目录，您将看到相同的文件。卷有两种主要类型。您最终将使用两者，但您将从卷安装开始。

**保存数据**

默认情况下，`todo` 应用程序将其数据存储在 `/etc/todos/todo.db`容器文件系统的 `SQLite` 数据库中。如果您不熟悉 `SQLite`，不用担心！它只是一个将所有数据存储在单个文件中的关系数据库。虽然这对于大型应用程序来说不是最好的，但它适用于小型演示。稍后您将了解如何将其切换到不同的数据库引擎。

由于数据库是单个文件，如果您可以将该文件保留在主机上并将其可供下一个容器使用，那么它应该能够从上一个容器停止的地方继续。通过创建卷并将其附加（通常称为“安装”）到存储数据的目录，您可以保留数据。当容器写入文件时`todo.db`，它将数据保存到卷中的主机。您将使用卷安装。将卷挂载视为不透明的数据桶。Docker 完全管理卷，包括磁盘上的存储位置。您只需要记住卷的名称即可。

**创建一个卷并启动容器**

您可以使用 CLI 或 Docker Desktop 的图形界面创建卷并启动容器。

1. 使用`docker volume create`命令创建卷。

    ```
    $ docker volume create todo-db
    ```
2. 再次停止并删除待办事项应用程序容器`docker rm -f <id>`，因为它仍在运行而不使用持久卷。

3. 启动 todo 应用程序容器，但添加--mount指定卷安装的选项。为卷命名，并将其安装到/etc/todos容器中，该容器捕获在该路径中创建的所有文件。在 Mac 或 Linux 终端中，或者在 Windows 命令提示符或 PowerShell 中，运行以下命令：

    ```
    $ docker run -dp 127.0.0.1:3000:3000 --mount type=volume,src=todo-db,target=/etc/todos getting-started
    ```

**验证数据是否持续存在**

1. 容器启动后，打开应用程序并将一些项目添加到您的待办事项列表中。

2. 停止并删除待办事项应用程序的容器。使用 Docker Desktop 或`docker ps`获取 ID，然后`docker rm -f <id>`将其删除。

3. 使用与上面相同的步骤启动一个新容器。

4. 打开应用程序。您应该会看到您的项目仍在列表中。

5. 检查完清单后，请继续移除容器。

很多人经常问“当我使用卷时，`Docker` 将我的数据存储在哪里？” 如果你想知道，可以使用`docker volume inspect`命令。

```
$ docker volume inspect todo-db
[
    {
        "CreatedAt": "2019-09-26T02:18:36Z",
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/todo-db/_data",
        "Name": "todo-db",
        "Options": {},
        "Scope": "local"
    }
]
```
这Mountpoint是磁盘上数据的实际位置。请注意，在大多数计算机上，您需要具有 root 访问权限才能从主机访问此目录。但是，那就是它所在的地方。

> 在 Docker Desktop 中运行时，Docker 命令实际上是在计算机上的小型虚拟机内运行。如果您想查看挂载点目录的实际内容，则需要查看该虚拟机的内部。

#### 绑定挂载

绑定挂载是另一种类型的挂载，它允许你将主机文件系统中的目录共享到容器当中。在处理应用程序时，你可以使用绑定挂载将源代码挂载到容器中。一旦你保存文件，容器就会立即看到你所做的更改。这意味着你可以在容器中运行进程来监视文件系统更改并对其作出响应。卷挂载与绑定挂载之间的区别：

|                     | Name volumes             | Bind mounts                      |
|:--------------------|:-------------------------|:---------------------------------|
|Host location        |Docker chooses            |You decide                        |
|Mount example (using --mount) |type=volume,src=my-volume,target=/usr/local/data |type=bind,src=/path/to/data,target=/usr/local/data|
|Populates new volume with container contents |Yes |No |
|Supports Volume Drivers |Yes |No |

1. 打开终端并将目录更改为`app` 入门存储库的目录。

2. 运行以下命令以bash在ubuntu具有绑定挂载的容器中启动。

    ```
    $ docker run -it --mount type=bind,src="$(pwd)",target=/src ubuntu bash
    ```
    该--mount选项告诉 Docker 创建绑定挂载，其中`src`是主机上的当前工作目录 ( `getting-started/app`)， target也是该目录应出现在容器内的位置 ( `/src`)。

3. 运行命令后，Docker 将`bash`在容器文件系统的根目录中启动交互式会话。

4. 将目录更改为该src目录。这是启动容器时安装的目录。列出此目录的内容将显示与 `getting-started/app`主机上的目录中相同的文件。

    ```
    root@ac1237fad8db:/# pwd
    /
    root@ac1237fad8db:/# ls
    bin   dev  home  media  opt   root  sbin  srv  tmp  var
    boot  etc  lib   mnt    proc  run   src   sys  usr
    ```
5. 创建一个名为 的新文件`myfile.txt`。

    ```
    root@ac1237fad8db:/src# touch myfile.txt
    root@ac1237fad8db:/src# ls
    Dockerfile  myfile.txt  node_modules  package.json  spec  src  yarn.lock
    ```
6. 打开app主机上的目录，观察该`myfile.txt`目录下有文件。

    ```
    ├── app/
    │ ├── Dockerfile
    │ ├── myfile.txt
    │ ├── node_modules/
    │ ├── package.json
    │ ├── spec/
    │ ├── src/
    │ └── yarn.lock
    ```

7. 从主机中删除该`myfile.txt`文件。

8. `app`在容器中，再次列出目录的内容。观察到该文件现在已经消失了。

    ```
    root@ac1237fad8db:/src# ls
    Dockerfile  node_modules  package.json  spec  src  yarn.lock
    ```
9. `Ctrl + D` 停止交互式容器会话

**在容器中运行应用**

以下步骤描述了如何使用执行以下操作的绑定安装来运行开发容器：
- 将源代码挂载到容器中
- 安装所有依赖项
- 开始`nodemon`监视文件系统更改

1. 确保当前没有任何`getting-started`容器正在运行。

2. 从目录运行以下命令`getting-started/app`。

    ```
    $ docker run -dp 127.0.0.1:3000:3000 \
    -w /app --mount type=bind,src="$(pwd)",target=/app \
    node:18-alpine \
    sh -c "yarn install && yarn run dev"
    ```
    以下是该命令的细分：
    - -`dp 127.0.0.1:3000:3000`- 和之前一样。以分离（后台）模式运行并创建端口映射
    - -`w /app`- 设置“工作目录”或命令将从中运行的当前目录
    - --`mount type=bind,src="$(pwd)"`,`target=/app`- 将当前目录从主机绑定挂载到/app容器中的目录
    - `node:18-alpine`- 要使用的镜像。请注意，这是来自 Dockerfile 的应用程序的基础镜像
    - `sh -c "yarn install && yarn run dev"`- 命令。sh您正在使用（`alpine`没有`bash`）启动`shell`并运行`yarn install`以安装软件包。然后运行`yarn run dev`以启动开发服务器。如果您查看`package.json`，您将看到`dev`脚本启动`nodemon`。

3. 您可以使用查看日志`docker logs <container-id>`。当您看到以下内容时，您就会知道您已准备好出发：

    ```
    $ docker logs -f <container-id>
    nodemon src/index.js
    [nodemon] 2.0.20
    [nodemon] to restart at any time, enter `rs`
    [nodemon] watching dir(s): *.*
    [nodemon] starting `node src/index.js`
    Using sqlite database at /etc/todos/todo.db
    Listening on port 3000
    ```
    查看完日志后，按`Ctrl+C`退出。

4. 在`src/static/js/app.js`文件的第 109 行，将“添加项目”按钮更改为简单地说“添加”：

    ```js
    - {submitting ? 'Adding...' : 'Add Item'}
    + {submitting ? 'Adding...' : 'Add'}
    ```

5. 刷新网络浏览器中的页面，您应该会立即看到更改的反映。节点服务器可能需要几秒钟才能重新启动。如果出现错误，请尝试在几秒钟后刷新。

6. 请随意进行您想要进行的任何其他更改。每次进行更改并保存文件时，该`nodemon`过程都会自动重新启动容器内的应用程序。完成后，停止容器并使用以下命令构建新映像：

    ```
    $ docker build -t getting-started .
    ```

#### 多容器应用

到目前为止，您一直在使用单容器应用程序。但是，现在您将把 MySQL 添加到应用程序堆栈中。经常会出现以下问题：“MySQL 将在哪里运行？安装在同一个容器中还是单独运行？” 一般来说，每个容器应该做一件事，并且做好。以下是单独运行容器的几个原因：

- 您很有可能必须以不同于数据库的方式扩展 API 和前端。
- 单独的容器允许您隔离版本和更新版本。
- 虽然您可以在本地使用数据库容器，但您可能希望在生产中使用数据库托管服务。那么您不想将数据库引擎与您的应用程序一起提供。
- 运行多个进程将需要一个进程管理器（容器只启动一个进程），这增加了容器启动/关闭的复杂性。

如下图所示，最好在多个容器中运行您的应用程序。

    {% asset_img multi-app-architecture.png %}

**容器网络**

默认情况下，容器是独立运行的，并且不了解同一台计算机上的其他进程或容器。那么，如何允许一个容器与另一个容器通信呢？答案是网络。如果将两个容器放在同一网络上，它们就可以相互通信。

**启动mysql**

将容器放到网络上有两种方法:
- 启动容器时分配网络。
- 将已运行的容器连接到网络。

在以下步骤中，您将首先创建网络，然后在启动时附加 MySQL 容器。

1. 创建网络。

    ```
    $ docker network create todo-app
    ```
2. 启动 MySQL 容器并将其连接到网络。您还将定义数据库将用于初始化数据库的一些环境变量。要了解有关 MySQL 环境变量的更多信息，请参阅MySQL Docker Hub 列表中的“环境变量”部分。

    ```
    $ docker run -d \
     --network todo-app --network-alias mysql \
     -v todo-mysql-data:/var/lib/mysql \
     -e MYSQL_ROOT_PASSWORD=secret \
     -e MYSQL_DATABASE=todos \
     mysql:8.0
    ```
    在上面的命令中，您将看到该`--network-alias`标志。在后面的部分中，您将了解有关此标志的更多信息。

> 您会注意到上面命令中命名的卷todo-mysql-data安装在/var/lib/mysql，这是 MySQL 存储其数据的位置。但是，您从未运行过docker volume create命令。Docker 识别出您想要使用命名卷并自动为您创建一个。

3. 要确认数据库已启动并正在运行，请连接到数据库并验证其是否已连接。

    ```
    $ docker exec -it <mysql-container-id> mysql -u root -p
    ```
    当出现密码提示时，输入`secret`。在 MySQL `shell` 中，列出数据库并验证您是否看到该`todos`数据库。
    ```sql
    mysql> SHOW DATABASES;
    ```
    您应该看到如下所示的输出：
    ```
    +--------------------+
    | Database           |
    +--------------------+
    | information_schema |
    | mysql              |
    | performance_schema |
    | sys                |
    | todos              |
    +--------------------+
    5 rows in set (0.00 sec)
    ```
4. 退出 MySQL `shell` 以返回到您计算机上的 `shell`。

    ```sql
    mysql> exit
    ```

**连接mysql**

现在您知道 MySQL 已启动并正在运行，您可以使用它了。但是，你如何使用它呢？如果在同一网络上运行另一个容器，如何找到该容器？请记住，每个容器都有自己的 IP 地址。
为了回答上述问题并更好地理解容器网络，您将使用`nicolaka/netshoot`容器，它附带了许多可用于排除或调试网络问题的工具。

1. 使用 `nicolaka/netshoot` 镜像启动一个新容器。确保将其连接到同一网络。
    ```
    $ docker run -it --network todo-app nicolaka/netshoot
    ```
2. 在容器内，您将使用该`dig命`令，这是一个有用的 DNS 工具。您将查找主机名的 IP 地址`mysql`。
    ```
    dig mysql
    ```
    您应该得到如下所示的输出。
    ```
    ; <<>> DiG 9.18.8 <<>> mysql
    ;; global options: +cmd
    ;; Got answer:
    ;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 32162
    ;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 0

    ;; QUESTION SECTION:
    ;mysql.				IN	A

    ;; ANSWER SECTION:
    mysql.			600	IN	A	172.23.0.2

    ;; Query time: 0 msec
    ;; SERVER: 127.0.0.11#53(127.0.0.11)
    ;; WHEN: Tue Oct 01 23:47:24 UTC 2019
    ;; MSG SIZE  rcvd: 44
    ```
    在“答案部分”中，您将看到解析为的A记录 （您的 IP 地址很可能具有不同的值）。虽然通常不是有效的主机名，但 Docker 能够将其解析为具有该网络别名的容器的 IP 地址。请记住，您使用的是 较早的。`mysql 172.23.0.2 mysql --network-alias` 这意味着您的应用程序只需要连接到名为 的主机`mysql`，它就会与数据库通信。


**使用 MySQL 运行您的应用程序**

todo 应用程序支持设置一些环境变量来指定 MySQL 连接设置。他们是：

- MYSQL_HOST- 正在运行的 MySQL 服务器的主机名
- MYSQL_USER- 用于连接的用户名
- MYSQL_PASSWORD- 用于连接的密码
- MYSQL_DB- 连接后使用的数据库

1. 指定上面的每个环境变量，并将容器连接到您的应用程序网络。`getting-started/app`运行此命令时请确保您位于该目录中。

    ```
    docker run -dp 127.0.0.1:3000:3000 \
    -w /app -v "$(pwd):/app" \
    --network todo-app \
    -e MYSQL_HOST=mysql \
    -e MYSQL_USER=root \
    -e MYSQL_PASSWORD=secret \
    -e MYSQL_DB=todos \
    node:18-alpine \
    sh -c "yarn install && yarn run dev"
    ```
2. 如果您查看容器 ( ) 的日志`docker logs -f <container-id>`，您应该会看到类似于以下内容的消息，这表明它正在使用 `mysql` 数据库。

    ```
    nodemon src/index.js
    [nodemon] 2.0.20
    [nodemon] to restart at any time, enter `rs`
    [nodemon] watching dir(s): *.*
    [nodemon] starting `node src/index.js`
    Connected to mysql db at host mysql
    Listening on port 3000
    ```
3. 在浏览器中打开应用程序，然后将一些项目添加到您的待办事项列表中。

4. 连接到 `mysql` 数据库并证明项目正在写入数据库。请记住，密码是`secret`。
    ```
    $ docker exec -it <mysql-container-id> mysql -p todos
    ```
    在 mysql shell 中，运行以下命令：
    ```
    mysql> select * from todo_items;
    +--------------------------------------+--------------------+-----------+
    | id                                   | name               | completed |
    +--------------------------------------+--------------------+-----------+
    | c906ff08-60e6-44e6-8f49-ed56a0853e85 | Do amazing things! |         0 |
    | 2912a79e-8486-4bc3-a4c5-460793a575ab | Be awesome!        |         0 |
    +--------------------------------------+--------------------+-----------+
    ```

#### 使用 Docker Compose

Docker Compose是一款旨在帮助定义和共享多容器应用程序的工具。使用 Compose，我们可以创建一个 YAML 文件来定义服务，并且使用单个命令就可以启动或拆除所有内容。

使用 Compose 的一大优势是您可以在文件中定义应用程序堆栈，将其保存在项目存储库的根目录中（现在是版本控制的），并轻松的让其他人为您的项目做出贡献。有人只需要克隆你的存储库并启动编写应用程序。

**安装 Docker Compose**

1. 更新包索引，并安装最新版本的 `Docker Compose`：
    - 对于 Ubuntu 和 Debian，运行：
    ```
    $ sudo apt-get update
    $ sudo apt-get install docker-compose-plugin
    ```
    - 对于基于 RPM 的发行版，运行：
    ```
    $ sudo yum update
    $ sudo yum install docker-compose-plugin
    ```
2. 通过检查版本来验证 Docker Compose 是否正确安装。
    ```
    $ docker compose version
    Docker Compose version vN.N.N
    ```
**创建 Compose 文件**

1. 在文件夹的根目录下`/getting-started/app`，创建一个名为`docker-compose.yml`.

2. 在撰写文件中，我们首先定义要作为应用程序一部分运行的服务（或容器）列表。
    ```
    services:
    ```
**定义应用服务**

请记住，这是我们用来定义应用程序容器的命令。

```
$ docker run -dp 127.0.0.1:3000:3000 \
  -w /app -v "$(pwd):/app" \
  --network todo-app \
  -e MYSQL_HOST=mysql \
  -e MYSQL_USER=root \
  -e MYSQL_PASSWORD=secret \
  -e MYSQL_DB=todos \
  node:18-alpine \
  sh -c "yarn install && yarn run dev"
```
1. 首先，我们定义容器的服务入口和镜像。我们可以为该服务选择任何名称。该名称将自动成为网络别名，这在定义我们的 MySQL 服务时非常有用。

    ```
    services:
        app:
            image: node:18-alpine
    ```
2. 通常，您会看到`command`接近`image`定义的内容，但没有顺序要求。那么，让我们继续将其移至我们的文件中。

    ```
    services:
        app:
            image: node:18-alpine
            command: sh -c "yarn install && yarn run dev"
    ```
3. 让我们通过定义服务来迁移`-p 127.0.0.1:3000:3000`命令的一部分。`ports`我们将在这里使用 短语法，但也有更详细的 长语法可用。

    ```
    services:
        app:
            image: node:18-alpine
            command: sh -c "yarn install && yarn run dev"
            ports:
                - 127.0.0.1:3000:3000
    ```
4. 接下来，我们将使用和定义迁移工作目录 ( `-w /app`) 和卷映射 ( ) 。`Volumes` 也有短语法和长语法。`-v "$(pwd):/app"working_dirvolumes`
    
    Docker Compose 卷定义的优点之一是我们可以使用当前目录的相对路径。
    ```
    services:
        app:
            image: node:18-alpine
            command: sh -c "yarn install && yarn run dev"
            ports:
                - 127.0.0.1:3000:3000
            working_dir: /app
            volumes:
                - ./:/app
    ```
5. 我们需要使用`environment`密钥迁移环境变量定义。

    ```
    services:
        app:
            image: node:18-alpine
            command: sh -c "yarn install && yarn run dev"
            ports:
                - 127.0.0.1:3000:3000
            working_dir: /app
            volumes:
                - ./:/app
            environment:
                MYSQL_HOST: mysql
                MYSQL_USER: root
                MYSQL_PASSWORD: secret
                MYSQL_DB: todos
    ```

**定义 MySQL 服务**

现在，是时候定义 MySQL 服务了。我们用于该容器的命令如下：
```
$ docker run -d \
  --network todo-app --network-alias mysql \
  -v todo-mysql-data:/var/lib/mysql \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=todos \
  mysql:8.0
```
1. 我们将首先定义新服务并为其命名，`mysql`以便它自动获取网络别名。我们将继续指定要使用的镜像。

    ```
    services:
        app:
            # The app service definition
        mysql:
            image: mysql:8.0
    ```
2. 接下来，我们将定义卷映射。当我们使用 运行容器时`docker run`，会自动创建命名卷。但是，使用 Compose 运行时不会发生这种情况。我们需要在顶级 `volumes:`部分定义卷，然后在服务配置中指定挂载点。只需仅提供卷名称，即可使用默认选项。不过还有更多的选择。

    ```
    services:
        app:
            # The app service definition
        mysql:
            image: mysql:8.0
        volumes:
            - todo-mysql-data:/var/lib/mysql

    volumes:
        todo-mysql-data:
    ```
3. 最后，我们只需要指定环境变量即可。

    ```
    services:
        app:
            # The app service definition
        mysql:
            image: mysql:8.0
        volumes:
            - todo-mysql-data:/var/lib/mysql
        environment:
            MYSQL_ROOT_PASSWORD: secret
            MYSQL_DATABASE: todos

    volumes:
        todo-mysql-data:
    ```
4. 此时，我们的完整内容`docker-compose.yml`应该是这样的：

    ```
    services:
        app:
            image: node:18-alpine
            command: sh -c "yarn install && yarn run dev"
            ports:
                - 127.0.0.1:3000:3000
            working_dir: /app
            volumes:
                - ./:/app
            environment:
                MYSQL_HOST: mysql
                MYSQL_USER: root
                MYSQL_PASSWORD: secret
                MYSQL_DB: todos

        mysql:
            image: mysql:8.0
            volumes:
                - todo-mysql-data:/var/lib/mysql
            environment:
            MYSQL_ROOT_PASSWORD: secret
            MYSQL_DATABASE: todos

    volumes:
        todo-mysql-data:
    ```
**运行应用程序**

现在我们有了`docker-compose.yml`文件，我们可以启动它了！

1. 确保应用程序/数据库的其他副本没有首先运行（`docker ps`和`docker rm -f <ids>`）。

2. 使用命令启动应用程序`docker compose up`。我们将添加`-d`标志以在后台运行所有内容。

    ```
    $ docker compose up -d
    ```
    当我们运行它时，我们应该看到如下输出：
    ```
    Creating network "app_default" with the default driver
    Creating volume "app_todo-mysql-data" with default driver
    Creating app_app_1   ... done
    Creating app_mysql_1 ... done
    ```
    您会注意到卷和网络都已创建！默认情况下，`Docker Compose`` 会自动创建一个专门用于应用程序堆栈的网络（这就是我们没有在 `compose` 文件中定义网络的原因）。

3. 让我们使用命令查看日志`docker compose logs -f`。您将看到每个服务的日志交织到单个流中。当您想要观察与计时相关的问题时，这非常有用。该`-f`标志“跟随”日志，因此会在生成时为您提供实时输出。

    如果您已经运行该命令，您将看到如下所示的输出：
    ```
    mysql_1  | 2019-10-03T03:07:16.083639Z 0 [Note] mysqld: ready for connections.
    mysql_1  | Version: '8.0.31'  socket: '/var/run/mysqld/mysqld.sock'  port: 3306  MySQL Community Server (GPL)
    app_1    | Connected to mysql db at host mysql
    app_1    | Listening on port 3000
    ```
    服务名称显示在行的开头（通常是彩色的）以帮助区分消息。如果要查看特定服务的日志，可以将服务名称添加到日志命令的末尾（例如 `docker compose logs -f app`）。

4. 此时，您应该能够打开应用程序并看到它正在运行。

当您准备好将其全部卸载时，只需运行`docker compose down`整个应用程序或点击 Docker 仪表板上的垃圾桶即可。容器将停止，网络将被删除。一旦卸载，您可以切换到另一个项目，运行`docker compose up`并准备好为该项目做出贡献！真的没有比这更简单的了！

> 默认情况下，运行时不会删除 compose 文件中的命名卷docker compose down。如果要删除卷，则需要添加该--volumes标志。

#### 最佳实践

**镜像分层**

您知道您可以查看镜像的组成部分吗？使用该`docker image history` 命令，您可以看到用于在图像中创建每个镜像层的命令。

1. 使用该`docker image history`命令查看`getting-started`您在本教程前面创建的图像中的镜像层。
    ```
    $ docker image history getting-started
    ```
    您应该得到如下所示的输出（日期/ID 可能不同）
    ```
    IMAGE               CREATED             CREATED BY                                      SIZE                COMMENT
    a78a40cbf866        18 seconds ago      /bin/sh -c #(nop)  CMD ["node" "src/index.j…    0B                  
    f1d1808565d6        19 seconds ago      /bin/sh -c yarn install --production            85.4MB              
    a2c054d14948        36 seconds ago      /bin/sh -c #(nop) COPY dir:5dc710ad87c789593…   198kB               
    9577ae713121        37 seconds ago      /bin/sh -c #(nop) WORKDIR /app                  0B                  
    b95baba1cfdb        13 days ago         /bin/sh -c #(nop)  CMD ["node"]                 0B                  
    <missing>           13 days ago         /bin/sh -c #(nop)  ENTRYPOINT ["docker-entry…   0B                  
    <missing>           13 days ago         /bin/sh -c #(nop) COPY file:238737301d473041…   116B                
    <missing>           13 days ago         /bin/sh -c apk add --no-cache --virtual .bui…   5.35MB              
    <missing>           13 days ago         /bin/sh -c #(nop)  ENV YARN_VERSION=1.21.1      0B                  
    <missing>           13 days ago         /bin/sh -c addgroup -g 1000 node     && addu…   74.3MB              
    <missing>           13 days ago         /bin/sh -c #(nop)  ENV NODE_VERSION=12.14.1     0B                  
    <missing>           13 days ago         /bin/sh -c #(nop)  CMD ["/bin/sh"]              0B                  
    <missing>           13 days ago         /bin/sh -c #(nop) ADD file:e69d441d729412d24…   5.59MB
    ```
    每条线代表镜像中的一个层。这里的显示显示底部位于底部，最新层位于顶部。使用它，您还可以快速查看每层的大小，帮助诊断大镜像。

2. 您会注意到有几行被截断。如果添加该`--no-trunc`标志，您将获得完整的输出.
    ```
    $ docker image history --no-trunc getting-started
    ```

**层缓存**

现在您已经了解了分层的实际效果，接下来需要学习一个重要的课程，以帮助减少容器映像的构建时间。一旦层发生变化，所有下游层也必须重新创建.

让我们再看一次我们使用的 Dockerfile...
```
# syntax=docker/dockerfile:1
FROM node:18-alpine
WORKDIR /app
COPY . .
RUN yarn install --production
CMD ["node", "src/index.js"]
```
回到镜像历史输出，我们看到 Dockerfile 中的每个命令都成为镜像中的一个新层。您可能还记得，当我们对图像进行更改时，必须重新安装纱线依赖项。有没有办法来解决这个问题？每次构建时都传递相同的依赖项没有多大意义，对吧？

为了解决这个问题，我们需要重构 Dockerfile 以帮助支持依赖项的缓存。对于基于节点的应用程序，这些依赖项在文件中定义`package.json`。那么，如果我们首先只复制该文件，安装依赖项，然后复制其他所有内容会怎么样？然后，如果`package.json` ？

1. 更新 Dockerfile 以首先复制`package.json`，安装依赖项，然后复制其他所有内容。

    ```
    # syntax=docker/dockerfile:1
    FROM node:18-alpine
    WORKDIR /app
    COPY package.json yarn.lock ./
    RUN yarn install --production
    COPY . .
    CMD ["node", "src/index.js"]
    ```
2. `.dockerignore`在与 Dockerfile 相同的文件夹中创建一个包含以下内容的文件。
    ```
    node_modules
    ```
3. `.dockerignore`文件是有选择地仅复制图像相关文件的简单方法。您可以在此处阅读有关此内容的更多信息 。在这种情况下，`node_modules`应在第二步中省略该文件夹`COPY`，否则可能会覆盖该RUN步骤中命令创建的文件。

4. 使用 构建新图像`docker build`。
    ```
    $ docker build -t getting-started .
    ```
    你应该看到这样的输出......
    ```
    [+] Building 16.1s (10/10) FINISHED
    => [internal] load build definition from Dockerfile
    => => transferring dockerfile: 175B
    => [internal] load .dockerignore
    => => transferring context: 2B
    => [internal] load metadata for docker.io/library/node:18-alpine
    => [internal] load build context
    => => transferring context: 53.37MB
    => [1/5] FROM docker.io/library/node:18-alpine
    => CACHED [2/5] WORKDIR /app
    => [3/5] COPY package.json yarn.lock ./
    => [4/5] RUN yarn install --production
    => [5/5] COPY . .
    => exporting to image
    => => exporting layers
    => => writing image     sha256:d6f819013566c54c50124ed94d5e66c452325327217f4f04399b45f94e37d25
    => => naming to docker.io/library/getting-started
    ```
    您会看到所有镜像层都已重建。非常好，因为我们对 Dockerfile 做了很多修改。

5. 现在，对文件进行更改`src/static/index.html`（例如将其更改`<title>`为“The Awesome Todo App”）。

6. 现在再次使用构建 Docker 映像`docker build -t getting-started .`。这次，您的输出应该看起来有点不同。

    ```
    [+] Building 1.2s (10/10) FINISHED
    => [internal] load build definition from Dockerfile
    => => transferring dockerfile: 37B
    => [internal] load .dockerignore
    => => transferring context: 2B
    => [internal] load metadata for docker.io/library/node:18-alpine
    => [internal] load build context
    => => transferring context: 450.43kB
    => [1/5] FROM docker.io/library/node:18-alpine
    => CACHED [2/5] WORKDIR /app
    => CACHED [3/5] COPY package.json yarn.lock ./
    => CACHED [4/5] RUN yarn install --production
    => [5/5] COPY . .
    => exporting to image
    => => exporting layers
    => => writing image     sha256:91790c87bcb096a83c2bd4eb512bc8b134c757cda0bdee4038187f98148e2eda
    => => naming to docker.io/library/getting-started
    ```
    首先，您应该注意到构建速度快得多！而且，您会看到有几个步骤正在使用以前缓存的图层。我们正在使用构建缓存。推送和拉取此映像及其更新也会快得多。

**多阶段构建**

虽然我们不会在本教程中深入探讨它，但多阶段构建是一个非常强大的工具，可以帮助使用多个阶段来创建镜像。他们有几个优点：

- 将构建时依赖项与运行时依赖项分开
- 通过仅传送应用程序需要运行的内容来减小整体镜像大小

Maven/Tomcat 示例

构建基于 Java 的应用程序时，需要 `JDK` 将源代码编译为 Java 字节码。但是，生产中不需要该 JDK。此外，您可能会使用 `Maven` 或 `Gradle` 等工具来帮助构建应用程序。我们的最终图像中也不需要这些。多阶段构建有帮助。

```
# syntax=docker/dockerfile:1
FROM maven AS build
WORKDIR /app
COPY . .
RUN mvn package

FROM tomcat
COPY --from=build /app/target/file.war /usr/local/tomcat/webapps
```
在此示例中，我们使用一个阶段（称为`build`）来使用 Maven 执行实际的 Java 构建。在第二阶段（从 开始`FROM tomcat`），我们从阶段复制文件`build`。最终图像只是创建的最后一个阶段（可以使用`--target`标志覆盖）。

React 示例

在构建 React 应用程序时，我们需要一个 Node 环境来将 `JS` 代码（通常是 `JSX`）、`SASS` 样式表等编译为静态 `HTML、JS` 和 `CSS`。如果我们不进行服务器端渲染，我们甚至不需要 Node 环境来进行生产构建。为什么不在静态 `nginx` 容器中传送静态资源？

```
# syntax=docker/dockerfile:1
FROM node:18 AS build
WORKDIR /app
COPY package* yarn.lock ./
RUN yarn install
COPY public ./public
COPY src ./src
RUN yarn run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
```
在这里，我们使用`node:18`镜像来执行构建（最大化层缓存），然后将输出复制到 `nginx` 容器中。