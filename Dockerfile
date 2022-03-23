FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Set bash as default shell for conda operations
SHELL ["/bin/bash", "-c"] 

# Copy requirements
COPY requirements.txt requirements.txt

# Install requirements (pytorch has already been installed)
RUN pip install -r requirements.txt

# Move everything to the folder
COPY . .

ENTRYPOINT [ "python", "-m", "scripts.train", "--model", "poly-encoder" , "--evaluate", "--orgs_to_embed", "data/cleaned_all_orgs_with_linkedin.json" ] 
