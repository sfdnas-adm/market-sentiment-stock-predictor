# Use an official Python base image
FROM python:3.9

# Create a working directory inside the container
WORKDIR /app

# Copy your requirements file (if you have one)
# If you don't have a requirements.txt, skip these two lines
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Otherwise, you can install packages directly:
# RUN pip install torch==2.0.1 pandas==2.0.0 numpy==1.23.0 ...

# Copy your code into the container
COPY . /app

# Expose any port you plan to run on (if using Gradio or a web server)
EXPOSE 7860

# For example, if you have an gradio_app.py script, set it as the default command:
CMD ["python", "gradio_app.py"]
