# Cloud computing

Machine learning workflows often require significant computational resources. Toy problems and demos can be constructed to work on typical workstations and laptops. But many workflows such as model training quickly hit bottlenecks either with data management or GPU resources to obtain results in a reasonable amount of time.

Here we provide and overview of several options for researchers to utilize cloud computing services for hackweek projects. We focus on pre-configured services that offer Jupyter servers to connect and run code on remote machines.

We limit discussion to 3 major commercial cloud providers: Microsoft Azure, Amazon Web Services (AWS), and Google Cloud. You can consider "cloud computing" simply as renting computers from these 3 companies!

```{note}
This is a fast-evolving space and services and tech specs change rapidly! To the best of our knowledge this information is correct as of September 2023
```

## Data-proximate computing

ML workflows often require huge volumes of training data. Rather than having to download and store that data, Cloud providers often host large public archives. You will see better performance and have reduced costs if you make sure that your computation runs in the same Cloud as where your data is stored.


## Geoscience community-supported cyberinfrastructure

All participating of {{ hackweek }} have access to a computing environment provided by the [CryoCloud project](https://book.cryointhecloud.com/intro.html). CryoCloud operates a JupyterHub in the AWS us-west-2 data center (where NASA is storing many public remote sensing datasets). We encourage you to use CryoCloud but also list other options below:


| Service | Max vCPU | Max RAM (GB) | Storage (GB) | Datacenter |
| - | - | - | - | - |
| [CryoCloud](https://book.cryointhecloud.com/intro.html) | 4 | 32 | 10 | AWS us-west-2 |
| [Pangeo JupyterHub](https://pangeo.io/cloud.html) | 16 | 32 | 10 | GCP us-central-1b |
| [ASF Open Science Lab](https://opensciencelab.asf.alaska.edu) | 8 | 16 | 500 | AWS us-west-2 |


## Free GPUs

Many leading machine learning libraries (e.g. tensorflow, pytorch) are designed to take advantage of Graphical Processing Units (GPUs). Typically, using a machine with a GPU on the cloud costs ~$1/hr, but there are some pre-configured services to try things out for free (usually with a time cap). Also, free services have no guarantee on current or future availability. Nevertheless, these are great for experimenting!

| Service | vCPU | RAM (GB) | GPU | GPU RAM (GB) | Storage (GB) | Max Session (hr) | Datacenter |
| - | - | - | - | - | - |  - | - |
| [Google Colab](https://colab.research.google.com) | 2 | 12 | T4 | 16 | 40 | 12 | random! |
| [AWS Sagemaker Studio Lab](https://aws.amazon.com/sagemaker/studio-lab/) | 4 | 12 | T4 | 16 | 15 | 4 | us-east-2 |
| [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com) | 4 | 32 | T4 | 16 | 150 | 12 | eu-west-2 |


## Free CPUs

If you don't need a GPU (maybe you are just visualizing results), you can access machines that allow longer sessions. As a rough rule of thumb you can expect a machine with a single CPU to cost an order of magnitude less (~$0.1/hr). And once again, there are free options to get started:

| Service | Max vCPU | Max RAM (GB) | Storage (GB) | Session (vCPU hr/mo) | Datacenter |
| - | - | - | - | - | - |
| [GitHub Codespaces](https://github.com/features/codespaces) | 16 | 32 | 15 | 120 | Azure |
| [BinderHub](https://github.com/features/codespaces) | 2 | 4 | 10 | n/a | Various |

## Guaranteed Access

If your workflow requires resources or time limits exceding what is offered by the free services listed above you'll need your own Cloud account. Configuring Cloud resources and keeping track of costs is non-trivial. Fortunately for researchers, Cloud providers offer generous credit programs.

* AWS: https://aws.amazon.com/earth/research-credits/
* Azure: https://www.microsoft.com/en-us/azure-academic-research/
* GCP: https://edu.google.com/intl/ALL_us/programs/credits/research/
