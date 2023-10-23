# text-embedding-inference-server-edd

We take the following approach to explore the text-embeddings-inference server:
* Install the text-embeddings-inference server on a local CPU and run evaluations to compare performance between two embedding models: inference server's bge-large-en-v1.5 versus OpenAI's text-embedding-ada-002.
* Install the text-embeddings-inference server on an AWS GPU EC2 instance. We again run the evaluations to compare the performance between the same embedding models from the inference server and that from OpenAI.

Follow the steps in the notebook for detailed implementation.

The key difference between connecting to a CPU inference server and a GPU inference server is in the parameter `base_url` when constructing `TextEmbeddingsInference`, see the code snippet below.  Toggle the value of `base_url` based on local CPU vs cloud GPU to test in your inference server implementation.

```
from llama_index.embeddings import TextEmbeddingsInference

embed_model = TextEmbeddingsInference(
    model_name="BAAI/bge-large-en-v1.5",
    base_url = "http://127.0.0.1:8080",
    #base_url = "http://ec2-##-##-##-##.compute-1.amazonaws.com:8080",
    timeout=60,  # timeout in seconds
    embed_batch_size=10,  # batch size for embedding
)
```
