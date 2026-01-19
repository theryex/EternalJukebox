# EternalJukebox

This repository is a fork of [floriegl's EternalJukebox](https://github.com/floriegl/EternalJukebox), customized by Ryex for use with the Floppa Analyzer.

You can visit the hosted instance of this fork at [Ryex's website](https://www.ryex.xyz).

The source files for the EternalJukebox, a rehosting of the Infinite Jukebox.
This repo contains everything you need to host the EternalJukebox on your own server!

## 🎷 Ryex's Fork Customizations

This fork includes the following enhancements:

### Floppa Analyzer Integration

Instead of relying on the deprecated Spotify Audio Analysis API (which required a bookmarklet workaround), this instance uses the **Floppa Analyzer** - a custom GPU-accelerated audio analysis service. When a song isn't found in the database, the analyzer automatically processes the track using:

- **librosa** for beat detection and rhythm analysis
- **nnAudio** with CUDA acceleration for spectral feature extraction
- Custom algorithms for segment detection and structural analysis

The analyzer is hosted at `https://www.ryex.xyz/tools/analyze` and provides real-time progress updates during processing.

### Dual Song Picker Tabs

- **Loaded Songs**: Browse songs that have already been analyzed and are available in the system
- **Search Songs**: Search for new songs to analyze (uses Spotify search API)

### Analyzer Reference Files

The `/analyzer/` directory contains the source code for the Floppa Analyzer service:
- `api.py` - FastAPI endpoints for analysis requests and status polling
- `app.py` - Streamlit web interface for manual analysis
- `analyzer.py` - GPU-accelerated audio analysis engine
- `grabber.py` - Audio acquisition from various sources

## Credits

- **Paul Lamere** ([@plamere](https://twitter.com/plamere)) - Original creator of The Infinite Jukebox at Music Hack Day @ MIT
- **floriegl** ([GitHub](https://github.com/floriegl)) - Maintainer of the EternalJukebox fork and Docker setup
- **Claude** (Anthropic) - AI assistance for this customization

---


# Docker Install

## Prerequisites

You need to install [docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/)

## Configuration

To configure, grab `.env.example` from this repository, rename it to `.env` and change the appropriate values.

You'll also need `envvar_config.yaml`, but only edit this if you want some advanced configuration.

## Running

You can use the following `docker-compose.yaml` as a starting point to run the application without needing a database running, if you want to use a db refer to the main `docker-compose.yaml`.

```yaml
version: "3"

services:
  main:
    image: daviirodrig/eternaljukebox
    ports:
      - 8080:8080
    env_file:
      - .env
    volumes:
      - "./envvar_config.yaml:/EternalJukebox/envvar_config.yaml"
```

To start, run `docker compose up -d` in the folder containing `envvar_config.yaml`, `.env` and `docker-compose.yaml`. To stop, run `docker compose down`.

If you want to you can upgrade the image by pulling the newest with `docker pull daviirodrig/eternaljukebox` and then restart with `docker compose down` and `docker compose up -d`

If you want to change the port from 8080, edit `docker-compose.yml` port, to be `- <your port>:8080`

# Manual Install

## Prerequisites

### Java:

##### Windows

Download and install Java from https://www.java.com/en/download/

##### Debian-based Linux distributions

For Ubuntu or Debian-based distributions execute `sudo apt-get install default-jre` in the terminal

##### Fedora and CentOS

There is a tutorial for installing java on Fedora and CentOS at https://www.digitalocean.com/community/tutorials/how-to-install-java-on-centos-and-fedora

### Yt-dlp (a more up-to-date fork of Youtube-dl):

##### Windows

Download the .exe at https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe and place it in `C:\Windows\`, or in another folder on the PATH.

##### Linux

Use these commands in the terminal to install youtube-dl on Linux:
`sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp`
`sudo chmod a+rx /usr/local/bin/yt-dlp`

### ffmpeg:

##### Windows

Download the exe from https://ffmpeg.zeranoe.com/builds/ and place it in `C:\Windows\`, or in another folder on teh PATH.

##### Linux

ffmpeg is available to download in most distributions using `sudo apt-get install ffmpeg` or equivalent

## Getting the project files:

The whole process of obtaining project files is much easier now, as the build process is streamlined through Jenkins.

The project site is over [here](https://jenkins.abimon.org/job/EternalJukebox/), and contains the individual files to download, or an all-in-one zip for all the files. Alternatively, the files can be found over at a permanent server [here](https://abimon.org/eternal_jukebox)

## Configuring

First thing to do is create a new file called either `config.yaml` or `config.json` (YAML tends to be easier to write, but takes up slightly more space), then open it with notepad/notepad++ on Windows and whatever text editor you like on Linux (for example nano: `nano config.json`)

Now you should go to https://developer.spotify.com/my-applications/ and log in to your spotify account.
Then click the "Create an app" button and a new page should popup.
There give it a name and description and click create.
It should send you to the new app's page, the only thing you need from here is your Client ID and Client Secret
(Note: Never share these with anyone!)

You will also need a Youtube Data API key, which you can find about how to obtain [here](https://developers.google.com/youtube/v3/getting-started).

There are a variety of config options (documentation coming soon) that allow most portions of the EternalJukebox to be configured, and these can be entered here.

## Starting the server:

First you need to open the Terminal or Command Prompt.
Then make sure its running in the folder that your EternalJukebox.jar is in, once again to do this use the `cd` command.
Then execute the jar with `java -jar EternalJukebox.jar`

If everything went right it should say `Listening at http://0.0.0.0:11037`

you should now be able to connect to it with a browser through http://localhost:11037

Congrats you did it!

## Manually Building

This is not recommended unless you're making some modifications, and as such should only be performed by more advanced users

You'll need to obtain a copy of [Gradle](https://gradle.org/install/), likely a [JDK](https://www.oracle.com/java/technologies/javase/jdk11-archive-downloads.html), and [Jekyll](https://jekyllrb.com/). You'll also need the project files in some capacity, be it `git clone` or downloading the archive from GitHub.

From there, building in Gradle is simple; just run `gradle clean shadowJar` from the project file directory. That should produce a jar file in `build/libs` that will work for you. In addition, you'll need to build the Jekyll webpages, which can be done by running `jekyll build --source _web --destination web`
