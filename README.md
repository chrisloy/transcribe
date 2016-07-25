Install Instructions
---

```
# Download and install Yaafe (create a virtual env, move it to the right place, fix broken links)
workon music
pip install -r requirements.txt
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py2-none-any.whl
pip install --upgrade $TF_BINARY_URL
```
