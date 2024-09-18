# gammaflash-env

`gammaflash-env` is a project that sets up a Python virtual environment and provides a Docker-based development setup for the DL0-to-DL2 conversion process.

## Install and run the environment

Follow these steps to install and configure the Python environment.

1. **Install Python3**  
   Ensure you have Python 3 installed on your system. The supported version for this project is 3.8.8.

2. **Create the virtual environment**  
   Use the following command to create a new Python virtual environment at the desired location.

   ```bash
   python3 -m venv /path/to/new/virtual/environment
   ```

3. Install the required dependencies
   Once the environment is set up, install all necessary Python packages by running:
   
   ```bash
   pip install -r venv/requirements.txt
   ```

4. Activate the environment
   Activate the virtual environment with this command:

   ```bash
   source /path/to/new/virtual/environment/bin/activate
   ```

This process will set up a clean Python environment, isolated from your system’s global packages, allowing you to run the project dependencies securely.

=============

## Docker Image

You can also use Docker for a containerized environment. Docker ensures a consistent setup across different platforms.

1. Clean up Docker system
   Before building the images, you may want to free up space by pruning unused Docker resources.


   ```bash
   docker system prune
   ```

   __Building Docker Images__

   * On MacOS

     For Mac users, you can build Docker images for different architectures by running:

     ```bash
     docker build --platform linux/amd64 -t dl0todl2:1.5.0 -f ./Dockerfile.amd .
     docker build --platform linux/arm64 -t dl0todl2:1.5.0 -f ./Dockerfile.arm .
     ```
     
   * On Linux
     
     For Linux users, you can build the Docker image using:
     
     ```bash
     docker build -t dl0todl2:1.5.0 -f ./Dockerfile.amd .
     ```
     
     Alternatively, you can use the default Docker build:
     
     ```bash
     docker build -t dl0todl2:1.5.0 .
     ```

2. Run the bootstrap script
   Once the image is built, you can initialize the Docker container with the following command:


   ```bash
   ./bootstrap.sh dl0todl2:1.5.0 <USERNAME>
   ```

3. Start a Docker container
   To run a container with your source code and data linked, use:
   
   ```bash
   docker run -it -d -v <PATH/TO/SOURCE_CODE/TO/LINK>:<PATH/TO/WORKSPACE> -v <PATH/TO/DATA_FOLDER/TO/LINK>:<PATH/TO/DATA_FOLDER> -p <PORT>:<PORT> --name <NAME_CONTAINER> dl0todl2:1.5.0_<USERNAME> /bin/bash
   ```
   
4. Access the container
   Once the container is running, you can access it by executing:
   
   ```bash
   docker exec -it <NAME_CONTAINER> /bin/bash
   ```
   
## Inside the container

After accessing the container, navigate to the home directory and start the project’s entry point:

```bash
cd
./entrypoint.sh
```

To launch a Jupyter Lab instance within the container, run:

```
jupyter-lab --ip="*" --port <PORT> --no-browser --autoreload --NotebookApp.token='dl0todl2024#' --notebook-dir=/home/usergamma/workspace --allow-root
```

## Connecting to Jupyter from your local machine

Once Jupyter Lab is running inside the container, you can access it from your local machine by forwarding the container’s port with SSH:

```bash
ssh -L <PORT>:localhost:<PORT> <REMOTE_USER>@<REMOTE_HOST>
```

Now you can open your browser and access Jupyter Lab at `localhost:<PORT>`.









