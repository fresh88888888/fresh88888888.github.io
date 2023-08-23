---
title: 构建Rust镜像
date: 2023-07-28 15:23:01
tag: 
   - docker
   - Cloud Native
category:
   - Docker 
---

##### 获取示例应用

克隆示例应用程序以与本指南一起使用。打开终端，将目录更改为您要工作的目录，然后运行以下命令来克隆存储库：
```
$ git clone https://github.com/docker/docker-rust-hello
```
##### 为 Rust 创建 Dockerfile

现在您已经有了一个应用程序，您可以`docker init`为其创建一个 Dockerfile。在`docker-rust-hello`目录中，运行`docker init`命令。请参阅以下示例来回答 中的提示`docker init`。
<!-- more -->
```
$ docker init
Welcome to the Docker Init CLI!

This utility will walk you through creating the following files with sensible defaults for your project:
  - .dockerignore
  - Dockerfile
  - compose.yaml

Let's get started!

? What application platform does your project use? Rust
? What version of Rust do you want to use? 1.70.0
? What port does your server listen on? 8000
```
您的目录中现在应该有以下 3 个新文件`docker-rust-hello` ：

- Dockerfile
- .dockerignore
- compose.yaml

为了构建镜像，只需要 Dockerfile。在您最喜欢的 IDE 或文本编辑器中打开 Dockerfile 并查看它包含的内容。要了解有关 Dockerfile 的更多信息。当您运行时docker init，它还会创建一个.dockerignore文件。使用该.dockerignore文件指定您不想复制到图像中的模式和路径，以使图像尽可能小。

##### 创建镜像

现在您已经创建了 Dockerfile，您可以构建镜像了。为此，请使用该`docker build`命令。该`docker build`命令从 Dockerfile 和上下文构建 Docker 镜像。构建的上下文是位于指定 PATH 或 URL 中的文件集。Docker 构建过程可以访问位于此上下文中的任何文件。

构建命令可以选择使用一个--`tag`标志。该标签设置图像的名称和格式中的可选标签`name:tag`。如果您不传递标签，Docker 将使用“latest”作为其默认标签。构建 Docker 镜像。
```
$ docker build --tag docker-rust-image .
```
您应该看到如下所示的输出。
```
[+] Building 62.6s (14/14) FINISHED
 => [internal] load .dockerignore                                                                                                    0.1s
 => => transferring context: 2B                                                                                                      0.0s 
 => [internal] load build definition from Dockerfile                                                                                 0.1s
 => => transferring dockerfile: 2.70kB                                                                                               0.0s 
 => resolve image config for docker.io/docker/dockerfile:1                                                                           2.3s
 => CACHED docker-image://docker.io/docker/dockerfile:1@sha256:39b85bbfa7536a5feceb7372a0817649ecb2724562a38360f4d6a7782a409b14      0.0s
 => [internal] load metadata for docker.io/library/debian:bullseye-slim                                                              1.9s
 => [internal] load metadata for docker.io/library/rust:1.70.0-slim-bullseye                                                         1.7s 
 => [build 1/3] FROM docker.io/library/rust:1.70.0-slim-bullseye@sha256:585eeddab1ec712dade54381e115f676bba239b1c79198832ddda397c1f  0.0s
 => [internal] load build context                                                                                                    0.0s 
 => => transferring context: 35.29kB                                                                                                 0.0s 
 => [final 1/3] FROM docker.io/library/debian:bullseye-slim@sha256:7606bef5684b393434f06a50a3d1a09808fee5a0240d37da5d181b1b121e7637  0.0s 
 => CACHED [build 2/3] WORKDIR /app                                                                                                  0.0s
 => [build 3/3] RUN --mount=type=bind,source=src,target=src     --mount=type=bind,source=Cargo.toml,target=Cargo.toml     --mount=  57.7s 
 => CACHED [final 2/3] RUN adduser     --disabled-password     --gecos ""     --home "/nonexistent"     --shell "/sbin/nologin"      0.0s
 => CACHED [final 3/3] COPY --from=build /bin/server /bin/                                                                           0.0s
 => exporting to image                                                                                                               0.0s
 => => exporting layers                                                                                                              0.0s
 => => writing image sha256:f1aa4a9f58d2ecf73b0c2b7f28a6646d9849b32c3921e42adc3ab75e12a3de14                                         0.0s
 => => naming to docker.io/library/docker-rust-image
```

##### 查看本地镜像

要查看本地计算机上的映像列表，您有两种选择。一种是使用 Docker CLI，另一种是使用Docker Desktop。由于您已经在终端中工作，请查看使用 CLI 列出图像。要列出图像，请运行`docker images`命令。

```
$ docker images
REPOSITORY                TAG               IMAGE ID       CREATED         SIZE
docker-rust-image         latest            8cae92a8fbd6   3 minutes ago   123MB
```
您应该看到至少列出了一个镜像，包括您刚刚构建的镜像`docker-rust-image:latest`。

##### 标记镜像

镜像名称由斜杠分隔的名称组件组成。名称组件可以包含小写字母、数字和分隔符。分隔符可以包括句点、一个或两个下划线、或者一个或多个破折号。名称不能以分隔符开头或结尾。镜像由清单和层列表组成。此时不要太担心清单和层，除了指向这些工件的组合的“标签”之外。一张图片可以有多个标签。为您构建的镜像，创建第二个标签并查看其层。要为您构建的镜像创建新标签，请运行以下命令。

```
$ docker tag docker-rust-image:latest docker-rust-image:v1.0.0
```
该`docker tag`命令为镜像创建一个新标签。它不会创建新镜像。标签指向同一个镜像，只是引用镜像的另一种方式。现在，运行`docker images`命令以查看本地镜像的列表。
```
$ docker images
REPOSITORY                TAG               IMAGE ID       CREATED         SIZE
docker-rust-image         latest            8cae92a8fbd6   4 minutes ago   123MB
docker-rust-image         v1.0.0            8cae92a8fbd6   4 minutes ago   123MB
rust                      latest            be5d294735c6   4 minutes ago   113MB
```
您可以看到两个镜像以 开头`docker-rust-image`。您知道它们是相同的镜像，因为如果您查看该IMAGE ID列，您可以看到两个镜像的值相同。删除您刚刚创建的标签。为此，请使用该`rmi`命令。该rmi命令代表删除镜像。
```
$ docker rmi docker-rust-image:v1.0.0
Untagged: docker-rust-image:v1.0.0
```
请注意，Docker 的镜像说明Docker 并未删除该镜像，而只是“取消标记”它。您可以通过运行`docker images`命令来检查这一点。
```
$ docker images
REPOSITORY               TAG               IMAGE ID       CREATED         SIZE
docker-rust-image        latest            8cae92a8fbd6   6 minutes ago   123MB
rust                     latest            be5d294735c6   6 minutes ago   113MB
```
Docker 删除了标记为 的映像`:v1.0.0`，但该`docker-rust-image:latest`标记在您的计算机上可用。

##### 运行镜像

用于运行您在构建 Rust 镜像`docker run`中构建的镜像。

```
$ docker run docker-rust-image
```

运行此命令后，您会发现没有返回到命令提示符。这是因为您的应用程序是一个在循环中运行的服务器，等待传入的请求，而不将控制权返回给操作系统，直到您停止容器。打开一个新终端，然后使用命令向服务器发出请求`curl`。
```
$ curl http://localhost:8000
```
您应该看到如下所示的输出。
```
curl: (7) Failed to connect to localhost port 8000 after 2236 ms: Couldn't connect to server
```
正如您所看到的，您的curl命令失败了。这意味着您无法连接到端口 8000 上的本地主机。这是正常的，因为您的容器是独立运行的，其中包括网络。停止容器并使用本地网络上发布的端口 8000 重新启动。要停止容器，请按 `ctrl-c`。这将使您返回到终端提示符。要为容器发布端口，您将在命令中使用标志`--publish`（`-p`简称）`docker run`。命令的格式`--publish为[host port]:[container port]`. 因此，如果您想将容器内的端口 `8000` 公开到容器外的端口 `3001`，则需要传递`3001:8000`给`--publish`标志。在容器中运行应用程序时，您没有指定端口，默认为 `8000`。如果您希望之前发送到端口 `8000` 的请求能够正常工作，可以将主机的端口 `3001` 映射到容器的端口 `8000`：
```
$ docker run --publish 3001:8000 docker-rust-image
```
现在，重新运行`curl` 命令。记得打开一个新终端。
```
$ curl http://localhost:3001
```
您应该看到如下所示的输出。
```
Hello, Docker!
```

##### 分离模式运行

到目前为止这很棒，但是您的示例应用程序是一个 Web 服务器，您不必连接到容器。Docker 可以在分离模式或后台运行容器。为此，可以简称为`--detach`或`-d`。Docker 与以前一样启动容器，但这次将从容器“分离”并将您返回到终端提示符。
```
$ docker run -d -p 3001:8000 docker-rust-image
ce02b3179f0f10085db9edfccd731101868f58631bdf918ca490ff6fd223a93b
```
Docker 在后台启动容器并在终端上打印容器 ID。再次确保我们的容器正常运行。运行与上面相同的curl 命令。
```
$ curl http://localhost:3001
```

##### 列出容器

由于您在后台运行容器，因此如何知道您的容器是否正在运行或者您的计算机上正在运行哪些其他容器？那么，要查看计算机上运行的容器列表，请运行`docker ps`。这类似于在 Linux 中使用 `ps` 命令查看进程列表的方式。
```
CONTAINER ID   IMAGE                   COMMAND         CREATED         STATUS         PORTS                    NAMES
3074745e412c   docker-rust-image       "/bin/server"   8 seconds ago   Up 7 seconds   0.0.0.0:3001->8000/tcp   wonderful_kalam
```
该`docker ps`命令提供了有关正在运行的容器的大量信息。您可以查看容器 ID、容器内运行的映像、用于启动容器的命令、创建容器的时间、状态、公开的端口以及容器的名称。您可能想知道容器的名称来自哪里。由于启动时没有为容器提供名称，因此 Docker 生成了一个随机名称。您很快就会解决这个问题，但首先您需要停止容器。要停止容器，请运行`docker stop`停止容器的命令。您需要传递容器的名称，也可以使用容器 ID。

```
$ docker stop wonderful_kalam
wonderful_kalam
```
现在，重新运行该`docker ps`命令以查看正在运行的容器的列表。
```
$ docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```

##### 停止、启动和命名容器

您可以启动、停止和重新启动 Docker 容器。当您停止容器时，它不会被删除，但状态会更改为已停止，并且容器内的进程也会停止。当您在上一个模块中运行`docker ps`命令时，默认输出仅显示正在运行的容器。当您通过`--all`或`-a`简称时，您会看到计算机上的所有容器，无论它们的启动或停止状态如何。

```
$ docker ps -a
CONTAINER ID   IMAGE                   COMMAND                  CREATED          STATUS                      PORTS                       
     NAMES
3074745e412c   docker-rust-image       "/bin/server"            3 minutes ago    Exited (0) 6 seconds ago                                
     wonderful_kalam
6cfa26e2e3c9   docker-rust-image       "/bin/server"            14 minutes ago   Exited (0) 5 minutes ago                                
     friendly_montalcini
4cbe94b2ea0e   docker-rust-image       "/bin/server"            15 minutes ago   Exited (0) 14 minutes ago                               
     tender_bose
```

您现在应该看到列出了几个容器。这些是您启动和停止但尚未删除的容器。重新启动刚刚停止的容器。找到您刚刚停止的容器的名称，并在以下重新启动命令中替换该容器的名称。
```
$ docker restart wonderful_kalam
```
现在使用该命令再次列出所有容器`docker ps`。
```
$ docker ps --all
CONTAINER ID   IMAGE                   COMMAND                  CREATED          STATUS                      PORTS                       
     NAMES
3074745e412c   docker-rust-image       "/bin/server"            6 minutes ago    Up 4 seconds                0.0.0.0:3001->8000/tcp           wonderful_kalam
6cfa26e2e3c9   docker-rust-image       "/bin/server"            16 minutes ago   Exited (0) 7 minutes ago                                
     friendly_montalcini
4cbe94b2ea0e   docker-rust-image       "/bin/server"            18 minutes ago   Exited (0) 17 minutes ago                               
     tender_bose
```
请注意，您刚刚重新启动的容器已以分离模式启动。另外，观察容器的状态为“Up X 秒”。当您重新启动容器时，它将以最初启动时相同的标志或命令启动。现在，停止并删除所有容器，然后看看修复随机命名问题。停止刚刚启动的容器。找到正在运行的容器的名称，并将以下命令中的名称替换为系统上容器的名称。
```
$ docker stop wonderful_kalam
wonderful_kalam
```
现在您已停止所有容器，请将其删除。当你删除一个容器时，它不再运行，也不是停止状态，但容器内的进程已经停止，容器的元数据也被删除。要删除容器，请`docker rm`使用容器名称运行命令。您可以使用单个命令将多个容器名称传递给该命令。再次，将以下命令中的容器名称替换为您系统中的容器名称。
```
$ docker rm wonderful_kalam friendly_montalcini tender_bose
wonderful_kalam
friendly_montalcini
tender_bose
```

再次运行该`docker ps --all`命令可以看到 Docker 删除了所有容器。现在，是时候解决随机命名问题了。标准做法是为容器命名，原因很简单，因为可以更轻松地识别容器中运行的内容以及与它关联的应用程序或服务。要命名容器，您只需将`--name`标志传递给`docker run`命令即可。

```
docker run -d -p 3001:8000 --name docker-rust-container docker-rust-image
1aa5d46418a68705c81782a58456a4ccdb56a309cb5e6bd399478d01eaa5cdda
docker ps
CONTAINER ID   IMAGE                   COMMAND         CREATED         STATUS         PORTS                    NAMES
c68fa18de1f6   docker-rust-image       "/bin/server"   7 seconds ago   Up 6 seconds   0.0.0.0:3001->8000/tcp   docker-rust-container
```

##### 在容器中运行数据库

您可以使用 PostgreSQL 的 Docker 官方映像并在容器中运行它，而不是下载 PostgreSQL、安装、配置然后将 PostgreSQL 数据库作为服务运行。在容器中运行 PostgreSQL 之前，创建一个 Docker 可以管理的卷来存储持久数据和配置。使用 Docker 提供的命名卷功能，而不是使用绑定安装。运行以下命令来创建您的卷。
```
$ docker volume create db-data
```
现在创建一个网络，您的应用程序和数据库将使用该网络相互通信。该网络称为用户定义的桥接网络，为您提供良好的 DNS 查找服务，您可以在创建连接字符串时使用该服务。
```
$ docker network create postgresnet
```
现在，您可以在容器中运行 PostgreSQL 并附加到创建的卷和网络。Docker 从 Hub 中提取映像并在本地运行它。在以下命令中，选项`--mount`用于启动带有卷的容器。
```
$ docker run --rm -d --mount \
  "type=volume,src=db-data,target=/var/lib/postgresql/data" \
  -p 5432:5432 \
  --network postgresnet \
  --name db \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -e POSTGRES_DB=example \
  postgres
```
现在，确保您的 PostgreSQL 数据库正在运行并且可以连接到它。连接到容器内正在运行的 PostgreSQL 数据库。
```
$ docker exec -it db psql -U postgres
```
您应该看到如下所示的输出。
```
psql (15.3 (Debian 15.3-1.pgdg110+1))
Type "help" for help.

postgres=#
```
`psql`在上一个命令中，您通过将命令传递给容器来登录到 PostgreSQL 数据库db。按 `ctrl-d` 退出 PostgreSQL 交互式终端。

##### 获取并运行示例应用程序

1. 使用以下命令克隆示例应用程序存储库。
```
$ git clone https://github.com/docker/docker-rust-postgres
```
2. 在克隆存储库的目录中，运行`docker init`以创建必要的 Docker 文件。请参阅以下示例来回答 中的提示`docker init`。
```
$ docker init
Welcome to the Docker Init CLI!

This utility will walk you through creating the following files with sensible defaults for your project:
  - .dockerignore
  - Dockerfile
  - compose.yaml

Let's get started!

? What application platform does your project use? Rust
? What version of Rust do you want to use? 1.70.0
? What port does your server listen on? 8000
```
3. 在克隆存储库的目录中，Dockerfile在 IDE 或文本编辑器中打开以更新它。

`docker init`处理了 Dockerfile 中大部分指令的创建，但您需要针对您独特的应用程序更新它。除了`src`目录之外，该应用程序还包括一个`migrations`用于初始化数据库的目录。将目录的绑定挂载添加`migrations`到 Dockerfile 中的构建阶段。以下是更新后的 Dockerfile。
```
# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/
   
################################################################################
# Create a stage for building the application.
   
ARG RUST_VERSION=1.70.0
ARG APP_NAME=react-rust-postgres
FROM rust:${RUST_VERSION}-slim-bullseye AS build
ARG APP_NAME
WORKDIR /app
   
# Build the application.
# Leverage a cache mount to /usr/local/cargo/registry/
# for downloaded dependencies and a cache mount to /app/target/ for 
# compiled dependencies which will speed up subsequent builds.
# Leverage a bind mount to the src directory to avoid having to copy the
# source code into the container. Once built, copy the executable to an
# output directory before the cache mounted /app/target is unmounted.
RUN --mount=type=bind,source=src,target=src \
    --mount=type=bind,source=Cargo.toml,target=Cargo.toml \
    --mount=type=bind,source=Cargo.lock,target=Cargo.lock \
    --mount=type=cache,target=/app/target/ \
    --mount=type=cache,target=/usr/local/cargo/registry/ \
    --mount=type=bind,source=migrations,target=migrations \
    <<EOF
set -e
cargo build --locked --release
cp ./target/release/$APP_NAME /bin/server
EOF
   
################################################################################
# Create a new stage for running the application that contains the minimal
# runtime dependencies for the application. This often uses a different base
# image from the build stage where the necessary files are copied from the build
# stage.
#
# The example below uses the debian bullseye image as the foundation for    running the app.
# By specifying the "bullseye-slim" tag, it will also use whatever happens to    be the
# most recent version of that tag when you build your Dockerfile. If
# reproducability is important, consider using a digest
# (e.g.,    debian@sha256:ac707220fbd7b67fc19b112cee8170b41a9e97f703f588b2cdbbcdcecdd8af57).
FROM debian:bullseye-slim AS final
   
# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/   #user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser
USER appuser
   
# Copy the executable from the "build" stage.
COPY --from=build /bin/server /bin/
   
# Expose the port that the application listens on.
EXPOSE 8000
   
# What the container should run when it is started.
CMD ["/bin/server"]
```
4. 在克隆存储库的目录中，运行`docker build`以构建映像。
```
$ docker build -t rust-backend-image .
```
5. 使用以下选项运行`docker run`，将映像作为容器在与数据库相同的网络上运行。
```
$ docker run \
  --rm -d \
  --network postgresnet \
  --name docker-develop-rust-container \
  -p 3001:8000 \
  -e PG_DBNAME=example \
  -e PG_HOST=db \
  -e PG_USER=postgres \
  -e PG_PASSWORD=mysecretpassword \
  -e ADDRESS=0.0.0.0:8000 \
  -e RUST_LOG=debug \
  rust-backend-image
```
6. 应用程序以验证它是否连接到数据库。
```
$ curl http://localhost:3001/users
```
您应该得到如下所示的响应。
```
[{"id":1,"login":"root"}]
```
##### 使用 Compose 进行本地开发

当您运行`docker init`时，除了Dockerfile，它还会创建一个`compose.yaml`文件。这个 Compose 文件非常方便，因为您不必键入所有参数来传递给命令`docker run`。您可以使用 Compose 文件以声明方式执行此操作。在克隆存储库的目录中，`compose.yaml`在 IDE 或文本编辑器中打开文件。`docker init`处理了大部分指令的创建，但您需要针对您独特的应用程序对其进行更新。

您需要更新文件中的以下项目`compose.yaml`：
- 取消所有数据库指令的注释。
- 在server服务下添加环境变量。
以下是更新后的compose.yaml文件。
```
# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Docker compose reference guide at
# https://docs.docker.com/compose/compose-file/

# Here the instructions define your application as a service called "server".
# This service is built from the Dockerfile in the current directory.
# You can add other services your application may depend on here, such as a
# database or a cache. For examples, see the Awesome Compose repository:
# https://github.com/docker/awesome-compose
services:
  server:
    build:
      context: .
      target: final
    ports:
      - 8000:8000
    environment:
      - PG_DBNAME=example
      - PG_HOST=db
      - PG_USER=postgres
      - PG_PASSWORD=mysecretpassword
      - ADDRESS=0.0.0.0:8000
      - RUST_LOG=debug
# The commented out section below is an example of how to define a PostgreSQL
# database that your application can use. `depends_on` tells Docker Compose to
# start the database before your application. The `db-data` volume persists the
# database data between container restarts. The `db-password` secret is used
# to set the database password. You must create `db/password.txt` and add
# a password of your choosing to it before running `docker compose up`.
    depends_on:
      db:
        condition: service_healthy
  db:
    image: postgres
    restart: always
    user: postgres
    secrets:
      - db-password
    volumes:
      - db-data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=example
      - POSTGRES_PASSWORD_FILE=/run/secrets/db-password
    expose:
      - 5432
    healthcheck:
      test: [ "CMD", "pg_isready" ]
      interval: 10s
      timeout: 5s
      retries: 5
volumes:
  db-data:
secrets:
  db-password:
    file: db/password.txt
```
请注意，该文件没有为这两个服务指定网络。当您使用 Compose 时，它​​会自动创建一个网络并将服务连接到该网络。在使用 Compose 运行应用程序之前，请注意此 Compose 文件指定了一个`password`.txt文件来保存数据库的密码。您必须创建此文件，因为它不包含在源存储库中。在克隆存储库的目录中，创建一个名为 的新目录db，并在该目录内创建一个名为 的文件`password.txt`，其中包含数据库的密码。使用您喜欢的 IDE 或文本编辑器，将以下内容添加到文件中`password.txt`。
```
mysecretpassword
```
如果您有任何其他容器在前面的部分中运行，请立即停止它们。现在，运行以下`docker compose up`命令来启动您的应用程序。
```
$ docker compose up --build
```
该命令传递`--build`标志，以便 Docker 编译您的映像，然后启动容器。现在测试您的 API 端点。打开一个新终端，然后使用`curl`命令向服务器发出请求：
```
$ curl http://localhost:8000/users
```
您应该收到以下回复：
```
[{"id":1,"login":"root"}]
```
##### 应用程序配置 CI/CD

完成设置和使用 Docker GitHub Actions 来构建 Docker 映像以及将映像推送到 Docker Hub 的过程。您将完成以下步骤：
- 在 GitHub 上创建一个新存储库。
- 定义 GitHub Actions 工作流程。
- 运行工作流程。

创建 GitHub 存储库并配置 Docker Hub 密钥。
1. 创建新的 GitHub 存储库 。该存储库包含一个简单的 Dockerfile，仅此而已。如果您愿意，可以随意使用另一个包含可用 Dockerfile 的存储库。
2. `Open the repository Settings, and go to Secrets and variables > Actions`.
3. 创建一个名为的新密钥`DOCKERHUB_USERNAME`，并将您的 Docker ID 作为值。
4. 为 Docker Hub创建新的 个人访问令牌 (PAT) 。您可以命名该令牌`clockboxci`。
5. 将 PAT 添加为 GitHub 存储库中的`second secret`，名称为 DOCKERHUB_TOKEN。

设置 GitHub Actions 工作流程以构建映像并将其推送到 Docker Hub。
1. 转到 GitHub 上的存储库，然后选择“Actions”选项卡。
2. 选择自己设置工作流程。这将带您进入一个页面，用于在存储库中创建新的 GitHub 操作工作流程文件（.github/workflows/main.yml默认情况下）。
3. 在编辑器窗口中，复制并粘贴以下 YAML 配置。
   ```
   name: ci

   on:
   push:
      branches:
         - "main"

   jobs:
   build:
      runs-on: ubuntu-latest
   ```
   - `name`：此工作流程的名称。
   - `on.push.branches`：指定此工作流应在列表中分支的每个推送事件上运行。
   - `jobs`：创建作业 ID ( `build`) 并声明作业应运行的机器类型。

   有关此处使用的 YAML 语法的更多信息，请参阅 [GitHub Actions 的工作流语法](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)。

定义工作流程步骤:
   ```
   jobs:
      build:
         runs-on: ubuntu-latest
         steps:
            -
            name: Checkout
            uses: actions/checkout@v3
            -
            name: Login to Docker Hub
            uses: docker/login-action@v2
            with:
               username: ${{ secrets.DOCKERHUB_USERNAME }}
               password: ${{ secrets.DOCKERHUB_TOKEN }}
            -
            name: Set up Docker Buildx
            uses: docker/setup-buildx-action@v2
            -
            name: Build and push
            uses: docker/build-push-action@v4
            with:
               context: .
               file: ./Dockerfile
               push: true
               tags: ${{ secrets.DOCKERHUB_USERNAME }}/clockbox:latest
   ```
   前面的 YAML 片段包含一系列步骤：
   1. 检查构建机器上的存储库。
   2. 使用Docker 登录操作和您的 Docker Hub 凭据登录 Docker Hub 。
   3. 使用Docker Setup Buildx操作创建 BuildKit 构建器实例 。
   4. 使用Build and push Docker images构建容器镜像并将其推送到 Docker Hub 存储库 。

   该`with`键列出了配置步骤的许多输入参数：
   - `context`：构建上下文。
   - `file`：Dockerfile 的文件路径。
   - `push`：告诉在构建镜像后将镜像上传到注册表的操作。
   - `tags`：指定将镜像推送到何处的标签。

将这些步骤添加到您的工作流程文件中。完整的工作流程配置应如下所示：
```
name: ci

on:
  push:
    branches:
      - "main"

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      -
        name: Checkout
        uses: actions/checkout@v3
      -
        name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      -
        name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      -
        name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: ${{ secrets.DOCKERHUB_USERNAME }}/clockbox:latest
```

运行工作流程: 保存工作流程文件并运行作业。
1. 选择提交更改...并将更改推送到main分支。推送提交后，工作流程将自动启动。
2. 转到“Actions”选项卡。它显示工作流程。--选择工作流程会显示所有步骤的细分。
3. 工作流程完成后，转到 Docker Hub 上的存储库。如果您在该列表中看到新的存储库，则意味着 GitHub Actions 已成功将镜像推送到 Docker Hub！

