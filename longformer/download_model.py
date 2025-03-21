import urllib.request
import urllib
import os

def downlaod_model(path="../longformer-base/longformer-base-4096.tar.gz"):
    folder = os.path.dirname(path)
    os.makedirs(folder,exist_ok=True)
# download zip
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz",
        path)
    

if __name__=='__main__':
    downlaod_model()