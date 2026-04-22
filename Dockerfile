FROM python:3.10

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Install dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

EXPOSE 7860

# Start FastAPI 
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
