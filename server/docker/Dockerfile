# Instructions from the app developer
# - you should use the 'node' official image, with the alpine 6.x branch
FROM gw000/keras:2.0.8-py3-tf-cpu
# - this app listens on port 3000, but the container should launch on port 80
EXPOSE 5000
  #  so it will respond to http://localhost:80 on your computer
# - then it should use alpine package manager to install tini: 'apk add --update tini'
RUN mkdir -p /usr/src/app
# - then it should create directory /usr/src/app for app files with 'mkdir -p /usr/src/app'

# - Node uses a "package manager", so it needs to copy in package.json file
WORKDIR /usr/src/app

COPY ./requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-index
# - then it needs to run 'npm install' to install dependencies from that file
# - to keep it clean and small, run 'npm cache clean --force' after above

# - then it needs to copy in all files from current directory
COPY ./. .
# - then it needs to start container with command 'tini -- node ./bin/www'
# - in the end you should be using FROM, RUN, WORKDIR, COPY, EXPOSE, and CMD commands

CMD ["python", "application.py"]

# Bonus Extra Credit
# this will not have you setting up a complete image useful for local develement, test, and prod
# it's just meant to get you started with basic Dockerfile concepts and not focus too much on
# proper Node.js use in a container. **If you happen to be a Node.js Developer**, then
# after you get through more of this course, you should come back and use my
# Node Docker Good Defaults sample project on GitHub to change this Dockerfile for
# better local development with more advanced topics
# https://github.com/BretFisher/node-docker-good-defaults
