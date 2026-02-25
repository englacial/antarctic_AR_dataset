# Antarctic Atmospheric River (AR) Dataset Tutorial

Welcome! This repo contains a tutorial notebook showcasing a fully cloud-based workflow for extracting landfalling characteristics and impacts from a catalog of atmospheric river storms in Antarctica, as mentioned in the following AGU25 talks:

+ [A12D-01 Linking Antarctic Atmospheric River Characteristics with Their Landfalling Impacts](https://agu.confex.com/agu/agu25/meetingapp.cgi/Paper/1970828), [slides](https://zenodo.org/records/17926794)
+ [IN23A-06 Cloud-based Workflows for Antarctic Atmospheric Rivers: Successes and Challenges](https://agu.confex.com/agu/agu25/meetingapp.cgi/Paper/1971552), [slides](https://zenodo.org/records/17926811)

## Background

Much of the existing work on Antarctic ARs is based on an Eulerian threshold catalog, which identifies individual AR pixels based on if poleward integrated vapor transport is above some extreme threshold at that particular time. However, this catalog does not distinguish between individual storm events. We first cluster the groups of AR pixels identified in this existing catalog, creating storm masks for individual AR storms. This is the catalog which we refer to, a subset of which is stored in the `catalog` directory.

To learn about the behavior of AR events on a storm-by-storm basis, we will want to extract relevant quantities associated with each storm,
such as landfalling temperature, duration of landfall, and many more. However, much of these quantities necessitate the use of reanalysis datasets, such as MERRA-2, which can be very costly to store in the cloud. To overcome this challenge, we implement a workflow that uses
the `earthaccess` software to stream data from MERRA-2 directly into memory, without the need to host these huge datasets. As such, we can loop through each individual storm, stream the relevant dataset days, compute quantities of interest, and then move onto the next storm.

## Getting Started

A smaller-scale version of the full workflow we are developing can be found in `dataset_construction.ipynb`. If you'd like to try it out, feel free to clone this repository and run through the notebook: everything should be ready-to-run.

We implemented this workflow on [CryoCloud](https://cryointhecloud.com/), a cloud-hosted JupyterHub whose mission is to facilitate open, collaborative, and reproducible science. We recommend running through this notebook on CryoCloud to be able to replicate our exact workflow for yourself. However, if you'd like to run this notebook on some other computing service or JupyterHub, go ahead! Just make sure whichever platform you use is in the AWS `us-west-2` region, or else you won't be able to stream the MERRA-2 reanalysis data from the AWS S3 buckets.

If you need to create an account, doing to is very easy! See [here](https://book.cryointhecloud.com/getting-started).

Once you have an account and log in, follow these steps to start up a server you can use to run through the workflow!

<img width="1065" height="864" alt="cryo_walkthrough" src="https://github.com/user-attachments/assets/c771d5ea-e811-49f9-9722-42f8d2482489" />

## Repo Contents
+ `environment.yml`: file specifying environment to run this workflow in
+ `dataset_construction.ipynb`: the notebook with the workflow implemented
+ `utils`: collection of helper modules to compute charcateristics and impacts of storms, given a streamed dataset and storm masks
+ `output`: directory containing output from the notebook
+ `data`: directory containing (1) a mask for the Antarctic Ice Sheet and (2) a dataset mapping lat/lon pixels to its area.
+ `catalog`: directory containing a subset of the full AR catalog we constructed (only first 250 storms, out of ~3000 total)
