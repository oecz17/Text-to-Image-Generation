## Demonstration and discussion

#Solution description:

The solution implemented uses a model called Parti. It combines techniques from NLP (Natural Language Processing) and Computer Vision. The task of generating images from text is not straight forward for computers. To accomplish this task, tons of data, training time and a large models are required.  

During training, the model learns how images and text are related to each other. For example, it learns that the visual representation of the word "apple" is rounded and can be of colors red, green or yellow. All these features that we humans, visualize very quickly when thinking of a word are learnt by the model. 

I selected this model in particular for these reasons:
- It is smaller than most architectures which means less training time and less cost
- Parti is very powerful, it allows very thorough descriptions as input and in the paper it is shown that it can generate images with all the details inputed to it
- It can include text into images without misspelled words. A very simple task for people but very hard for most text to image generation model.

# Is it better to deploy the model on a server and all devices consume it using an API? Or ot deploy it on each device? How?

Generally it is more efficient to deploy the model on a server and have every device consume it using an API. It is more efficient because it will allow updating and maintenance of the model to be a centralized process ensuring consistency across all devices. 

# How to scale our solution?
By scaling the resources of the host of the model. With auto-scaling, we could ensure we will be able to handle increased demand and optimal performance. 

Another strategy could be load balancing. Where we make use all the instances receive some amount of requests to avoid saturating a single server.

From the model's perpective, using GPU's, quantization techniques, distillation or converting the model into onnx, are techniques that help reduce the inference time and thus, the amount of instances needed when scaling the solution.

# The specs of the hosting machine, Cloud, etc.
It depends on the scale, expected workload, the models used, etc. Some of the things that need to be considered are:
- CPU or GPU. Most text to image generation models will require tons of computational power to generate an image relatively quick. Otherwise, the user would need to wait some minutes to receive it.
- Memory. Loading and processing the dataset, plus the model will need a sufficient amount of memory.
- Storage and availability are other factors to consider.
- Budget

Most popular clouds are AWS, GCP and Azure right now. Each of them provide different features on top of hosting machines that should be also consider when selecting a cloud. For example, Azure nowadays offers OpenAI models capabilites. 

# Tools and frameworks that could simplify the deployment process
Some frameworks that help are: Tensorflow Serving, Pytorch Lightning, FastAPI, Flask, Nvidia Triton, Docker, Kubernetes, Infrastructure-as-Code tools such as Terraform, CI/CD pipelines. 

Tensorflow Serving, Pytorch Lightning, Nvidia Triton are designed to serve a model in production and simplify the deployment process of the model.

FastAPI, Flask are used to build the APIs with python. 

Docker helps package the application, dependencies and model into a container. To deploy anywhere without worrying about operating systems or dependencies.

Kubernetes orchestrates the deployment, scaling and management of containerized applications.

Infrastructure-as-Code tools let you deinfe the cloud infrastructure programmatically.

# Suppose we got more data batches post-production for model tuning. How could we avoid bi-asing the new data in favor of the original one?

Some techniques that could be used to prevent this issue are:

- Random sampling
- Stratified sampling
- Using a validation set
- Monitor data distribution
- Fine tune on a full combination of old + new data
- Use a diverse dataset. Through techniques like clustering, find the most diverse and representative data to fine tune the model.